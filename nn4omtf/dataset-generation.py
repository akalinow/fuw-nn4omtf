"""
    OMTF dataset generation and balancing.
    Copyright (C) 2018 Jacek Łysiak
    MIT License
"""

import numpy as np
import os

class NPZ_DATASET:
    HITS_REDUCED = 'hits_reduced'
    PROD = 'prod'
    OMTF = 'omtf'

DATASET_TYPES = ['TRAIN', 'VAILD', 'TEST']

DATASET_FIELDS = ['HITS', 'HITS_TYPE', 'PT', 'SIGN', 'IS_NULL']

class HIST_TYPES:
    AVG = 'AVG'
    VALS = 'VALS'

class DATA_TYPES:
    ORIG = 'ORIG'
    TRANS = 'TRANS'

HIST_TYPES = [HIST_TYPES.AVG, HIST_TYPES.VALS]
DATA_TYPES = [DATA_TYPES.ORIG, DATA_TYPES.TRANS]

class OMTFDataset:
    """
    Balanced dataset preparation + statistical analysis.

    Datasets are generated from given set of `*.npz` files.
    It's assumed that:
    - each source file was created by ROOT-to-numpy converter and 
      has proper inner structure,
    - each source file has unique (muon pt code, muon charge sign) signature

    # Dataset types 

    Datasets generated: TRAIN, VALID, TEST.
    All datasets are created at once in same time thus contain unique examples.
    Size of each dataset can be set using: `train_n`, `valid_n`, `test_n`.
    
    # Treshold 

    `treshold` parameter (default=5400) is used to mark more event examples as
    those which HITS array doesn't contain sufficient amout of data. 
    (Looks like no hit was registered by sensor in detector.)
    `treshold` parameter acts on averaged HITS array for each single event.

    # HITS values transformation

    For better training conditions values in HITS array can be transformed.
    `transform` must be None or tuple (`null value`, `shift`) 
    which is used as follows:
    - all elements equals 5400 are mapped on `null value`
    - for the rest `+ shift` is applied

    # Balancing dataset

    Dataset generator collects statistcs of:
    - values in HITS array - used for validating dataset transformation 
    - averaged values in HITS array for single event - used for balancing dataset
    """
    
    def __init__(self, files, train_n, valid_n, test_n, treshold=5400., 
            transform=(0, 600)):
        """
        Args:
            files: list of paths to files created by ROOT-TO-NUMPY converter
            train_n: number of events in train dataset
            valid_n: number of events in valid dataset
            test_n: number of events in test dataset
            treshold: filter threshold applied on mean over hits array before transformation
            transform: (null value, shift value)
            
        """
        self.files = files
        
        self.names = DATASET_TYPES
        self.phase_n = [train_n, valid_n, test_n]
        
        self.treshold = treshold
        
        self.hist_types = HIST_TYPES 
        self.data_types = DATA_TYPES
            
        # Distribution of mean HITS_ARR per example
        # It shows how many events should be considered as null examples
        self.bins = np.linspace(0, 5400, 50)
        
        self.histograms = dict()
        for dtype in self.data_types:
            hists = dict()
            for htype in self.hist_types:
                hists[htype] = {'TOTAL': np.zeros(self.bins.shape[0] - 1), 'CODE': []}
            self.histograms[dtype] = hists
        
        self.transform = transform
 

    def moving_avg(self, arr, wnd_size=32):
        weights = np.ones(wnd_size) / wnd_size
        return np.convolve(arr, weights, mode='valid')

    
    def get_partition(self):
        """
        Get per data-file partitioning to get wanted distribution.
        """
        files_n = len(self.files)
        M = files_n + 1
        null_frac = 1. / M
        
        g_tot = 0
        n_tot = 0
        partition = []
        for N, name in zip(self.phase_n, self.names):
            events_per_file = int(N / files_n)
            nulls_per_file = int(events_per_file * null_frac)
            if nulls_per_file == 0:
                print("WARNING! Zero events will be taken as NULL examples in %s phase!" % name)
            good_per_file = events_per_file - nulls_per_file
            partition.append((g_tot, g_tot + good_per_file, n_tot, n_tot + nulls_per_file))
            n_tot += nulls_per_file
            g_tot += good_per_file
        return partition
    

    def add_histograms_for_file(self, hits, htype='VALS', dtype='ORIG'):
        """
        Get hits and hits averaged over single event and add histograms data.
        """
        hist, _ = np.histogram(hits, bins=self.bins)
        self.histograms[dtype][htype]['CODE'].append(hist)
        self.histograms[dtype][htype]['TOTAL'] += hist
        

    def dataset_validator(self, data, shuffled=False):
        """
        Validate shuffled dataset by checking whether examples match.
        Creates 2D histograms of pairs:
        - (hits_avg, pt_value)
        - (hits_avg, omtf_pt_value)
        - (pt_value, omtf_pt_value)
        """
        hist_avg = np.mean(data[0], axis=(1,2))
        ps = data[1][:,0] # Get muon PT value
        os = data[2][:,1] # Get OMTF PT value
        hists = (np.histogram2d(hist_avg, ps)[0], 
            np.histogram2d(hist_avg, os)[0],
            np.histogram2d(ps, os)[0])
        if not shuffled:
            self.valid_hists = hists
            return
        orig_hists = self.valid_hists
        self.valid_hists = {
            'ORIG': self.valid_hists,
            'SHUF': hists}
        for orig, shuf in zip(orig_hists, hists):
            if not np.all(np.equal(orig, shuf)):
                return False
        return True
    
    
    def save_train_examples_ordering(self, train_dataset, shuffled=False):
        """
        Apply moving average on mouns PT value arrays and check
        if examples are shuffled well.
        """
        ps_avg = moving_avg(train_dataset[1][:, 0])
        if not shuffled:
            self.train_examples_order = {'ORIG': ps_avg}
        else:
            self.train_examples_order['SHUF'] = ps_avg
        

    def generate(self):
        """
        Generate balanced datasets.
        """
        partition = self.get_partition()
        dataset = dict(zip(self.names, [None] * 3))
        for fn in self.files:
            data = np.load(fn)
            hits = data[NPZ_DATASET.HITS_REDUCED]
            prod = data[NPZ_DATASET.PROD]
            omtf = data[NPZ_DATASET.OMTF]
            file_data = [hits, prod, omtf]
            
            # Calc histogram of original input values
            hits_avg = np.mean(hits, axis=(1,2))
            self.add_histograms_for_file(hits, htype='VALS', dtype='ORIG')
            self.add_histograms_for_file(hits_avg, htype='AVG', dtype='ORIG')
            
            good_mask = hits_avg < self.treshold
            null_mask = hits_avg >= self.treshold
            
            if self.transform is not None:
                # Apply data transformation
                # Set new NULL value and shift others
                hits = np.where(hits >= 5400, self.transform[0], hits + self.transform[1])
                hits_avg = np.mean(hits, axis=(1,2))
                self.add_histograms_for_file(hits, htype='VALS', dtype='TRANS')
                self.add_histograms_for_file(hits_avg, htype='AVG', dtype='TRANS')
            
            good_data = [t[good_mask] for t in file_data]
            null_data = [t[null_mask] for t in file_data]
            
            file_data = []
            for (gb, ge, nb, ne), name  in zip(partition, self.names):
                _good_data = [_data[gb:ge] for _data in good_data]
                _null_data = [_data[nb:ne] for _data in null_data]
                if dataset[name] is None:
                    dataset[name] = [np.concatenate((_gd, _nd), axis=0) for _gd, _nd in zip(_good_data, _null_data)]
                else:
                    dataset[name] = [np.concatenate((_d, _gd, _nd), axis=0) for _d, _gd, _nd in zip(dataset[name], _good_data, _null_data)]
            data.close()
        
        self.save_train_examples_ordering(dataset['TRAIN'])
        
        for name in self.names:
            rng_state = np.random.get_state()
            self.dataset_validator(dataset[name])
            for i in range(3):
                # Shuffle each array in same way
                np.random.shuffle(dataset[name][i])
                np.random.set_state(rng_state)
            assert self.dataset_validator(dataset[name], shuffled=True), "%s dataset shuffle failed! Histograms don't match!"
        
        self.save_train_examples_ordering(dataset['TRAIN'], shuffled=True)
        self.dataset = dataset
        
        
    def save_dataset(self, prefix):
        data = dict()
        data_suffix = ['_HITS', '_PROD', '_OMTF']
        for k, v in self.dataset.items():
            for suff, d in zip(data_suffix, v):
                data[k + suff] = d            
        np.savez_compressed(path, **data)
    

    def save_stats(self, path):
        stats = {"TYPE": "DATASET_STATISTICS"}
        stats['TRAIN_EXAMPLES_ORDERING'] = self.train_examples_order
        np.savez_compressed(path, **stats)
        

DIR = "./orig-npz-datasets/"
fs = os.listdir(DIR)[:4]
files = [os.path.join(DIR, fn) for fn in fs]

ds = OMTFDataset(files, 10000, 2000, 1000, transform=(0, 600))
ds.generate()
ds.save_dataset("dataset-test")
ds.save_stats("dataset-stats")
