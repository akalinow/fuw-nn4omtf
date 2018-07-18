# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    OMTF dataset generation and balancing.
"""

import numpy as np
import os

from nn4omtf.const_files import FILE_TYPES
from nn4omtf.const_dataset import DATASET_TYPES, HIST_TYPES, DATA_TYPES,\
    HIST_SCOPES, ORD_TYPES, NPZ_DATASET, DATASET_FIELDS, DSET_STAT_FIELDS


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

    Let `N` be the number of examples in whole dataset and `F` number of 
    files paths in `files` list. Under assumption that each file has unique 
    (pt code, sign) signature, `F` is equal to number of correct classes in
    dataset.
    `events_per_file = N / F`
    To incorrect class / unknown / null belong examples which HITS array has
    not sufficient amout of information to propperly guess their correct class.
    (And network MUST learn to classify those examples as incorrent ones.)

    Number of all classes in dataset: `C = F+1`
    Fraction of all examples which must be in correct class: `(C - 1) / C`
    The rest `1 / C` of examples must come from null class.

    Examples taken from each source file:
    - correct: `events_per_file * (C - 1) / C`
    - incorrect: `events_per_files * 1 / C`

    # Examples ordering in TRAIN dataset
    
    Selected examples from each file are stored in arrays in grouped manner.
    TRAIN data must be shuffled to mix all examples and reach more uniform 
    distribution of classes over all dataset.
    Original and final dataset distributions are available as 
    `train_examples_order`.

    # Dataset statistics available
     
    - TRAIN examples ordering `ORIG` and 'SHUF`
    - HITS array input data histograms (per file and global - 
            containing data from all files):
      - averaged HITS per example for original and transformed input
      - HITS values for original and transformed input
    """
    
    def __init__(self, files, train_n, valid_n, test_n, treshold=5400., 
            transform=(0, 600), hist_bins=(-800, 5400, 80)):
        """
        Args:
            files: list of paths to files created by ROOT-TO-NUMPY converter
            train_n: number of events in train dataset
            valid_n: number of events in valid dataset
            test_n: number of events in test dataset
            treshold: filter threshold applied on mean over hits array before transformation
            transform: (null value, shift value)
            
        """
        self.hist_types = [HIST_TYPES.AVG, HIST_TYPES.VALS]
        self.data_types = [DATA_TYPES.ORIG, DATA_TYPES.TRANS]
        self.names = [DATASET_TYPES.TRAIN,
                DATASET_TYPES.VALID,
                DATASET_TYPES.TEST]

        self.files = files
        self.phase_n = [train_n, valid_n, test_n]
        self.treshold = treshold
            
        # Distribution of mean HITS_ARR per example
        # It shows how many events should be considered as null examples
        self.bins = np.linspace(*hist_bins)
        self.histograms = dict()
        for dtype in self.data_types:
            hists = dict()
            for htype in self.hist_types:
                hists[htype] = {
                    HIST_SCOPES.TOTAL: np.zeros(self.bins.shape[0] - 1), 
                    HIST_SCOPES.CODE: []}
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
        print("Files N: %d" % files_n)
        print("Divisions: %d" % M)
        print("Events fraction per division: %f" % null_frac)
        
        g_tot = 0
        n_tot = 0
        partition = []
        print("Dataset partition:")
        for N, name in zip(self.phase_n, self.names):
            events_per_file = int(N / files_n)
            nulls_per_file = int(events_per_file * null_frac)
            if nulls_per_file == 0:
                print("WARNING! Zero events will be taken as NULL examples \
                        in %s phase!" % name)
            good_per_file = events_per_file - nulls_per_file
            print("name: {}, events N: {}".format(name, N))
            print("events per file: %d" % events_per_file)
            print("null events per file: %d" % nulls_per_file)
            print("good events per file: %d" % good_per_file)
            partition.append((g_tot, g_tot + good_per_file, 
                n_tot, n_tot + nulls_per_file))
            n_tot += nulls_per_file
            g_tot += good_per_file
        print("Events taken from single file:")
        print("total: {}, good: {}, null: {}".format(g_tot+n_tot, g_tot, n_tot))
        return partition
    

    def add_histograms_for_file(self, hits, htype=HIST_TYPES.VALS, 
            dtype=DATA_TYPES.ORIG):
        """
        Get hits and hits averaged over single event and add histograms data.
        """
        hist, _ = np.histogram(hits, bins=self.bins)
        self.histograms[dtype][htype][HIST_SCOPES.CODE].append(hist)
        self.histograms[dtype][htype][HIST_SCOPES.TOTAL] += hist
        

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
            ORD_TYPES.ORIG: self.valid_hists,
            ORD_TYPES.SHUF: hists}
        for orig, shuf in zip(orig_hists, hists):
            if not np.all(np.equal(orig, shuf)):
                return False
        return True
    
    
    def save_train_examples_ordering(self, train_dataset, shuffled=False):
        """
        Apply moving average on mouns PT value arrays and check
        if examples are shuffled well.
        """
        ps_avg = self.moving_avg(train_dataset[1][:, 0])
        if not shuffled:
            self.train_examples_order = {ORD_TYPES.ORIG: ps_avg}
        else:
            self.train_examples_order[ORD_TYPES.SHUF] = ps_avg
        

    def generate(self):
        """
        Generate balanced datasets.
        """
        partition = self.get_partition()
        dataset = dict(zip(self.names, [None] * 3))
        signatures = []
        for fn in self.files:
            print('Reading data from: %s' % fn)
            data = np.load(fn)
            hits = data[NPZ_DATASET.HITS_REDUCED]
            prod = data[NPZ_DATASET.PROD]
            omtf = data[NPZ_DATASET.OMTF]
            code = data[NPZ_DATASET.PT_CODE]
            sign = data[NPZ_DATASET.SIGN]
            signatures.append((code, sign))

            
            # Calc histogram of original input values
            hits_avg = np.mean(hits, axis=(1,2))
            self.add_histograms_for_file(hits, htype=HIST_TYPES.VALS, 
                    dtype=DATA_TYPES.ORIG)
            self.add_histograms_for_file(hits_avg, htype=HIST_TYPES.AVG, 
                    dtype=DATA_TYPES.ORIG)
            
            good_mask = hits_avg < self.treshold
            good_n = np.sum(good_mask)
            null_mask = hits_avg >= self.treshold
            null_n = np.sum(null_mask)
            
            if self.transform is not None:
                # Apply data transformation
                # Set new NULL value and shift others
                hits = np.where(hits >= 5400, self.transform[0], 
                        hits + self.transform[1])
                hits_avg = np.mean(hits, axis=(1,2))
                self.add_histograms_for_file(hits, htype=HIST_TYPES.VALS, 
                        dtype=DATA_TYPES.TRANS)
                self.add_histograms_for_file(hits_avg, htype=HIST_TYPES.AVG, 
                        dtype=DATA_TYPES.TRANS)
            
            file_data = [hits, prod, omtf]
            good_data = [t[good_mask] for t in file_data]
            good_data += [np.zeros(good_n).astype(np.bool)]
            good_data += [np.ones(good_n) * code]
            null_data = [t[null_mask] for t in file_data]
            null_data += [np.ones(null_n).astype(np.bool)]
            null_data += [np.ones(null_n) * code]

            for (gb, ge, nb, ne), name  in zip(partition, self.names):
                _good_data = [_data[gb:ge] for _data in good_data]
                _null_data = [_data[nb:ne] for _data in null_data]
                if dataset[name] is None:
                    dataset[name] = [np.concatenate((_gd, _nd), axis=0) 
                            for _gd, _nd in zip(_good_data, _null_data)]
                else:
                    dataset[name] = [np.concatenate((_d, _gd, _nd), axis=0) 
                            for _d, _gd, _nd in 
                                zip(dataset[name], _good_data, _null_data)]
            data.close()
        
        self.save_train_examples_ordering(dataset[DATASET_TYPES.TRAIN])
        
        for name in self.names:
            rng_state = np.random.get_state()
            self.dataset_validator(dataset[name])
            for i in range(len(dataset[name])):
                # Shuffle each array in same way
                np.random.shuffle(dataset[name][i])
                np.random.set_state(rng_state)
            assert self.dataset_validator(dataset[name], shuffled=True), \
                    "%s dataset shuffle failed! Histograms don't match!"
        
        self.save_train_examples_ordering(dataset[DATASET_TYPES.TRAIN], 
                shuffled=True)
        self.dataset = dataset
        self.signatures = signatures
        
        
    def save_dataset(self, prefix, single_file=True):
        """
        Prepare all types of datasets with approptiate structure of elements.
        Dataset file structure is as follows:
            dict( [ ( phase name: dict([(data name, data values)]) ) ] )
        If `single_file` is True, all `phase_name` entries are stored in 
        single `*.npz` file.

        # Example - loading dict structure from `*.npz` file
        ```
            ds_npz = np.load(ds_npz_path)
            ds_train = ds_npz['TRAIN'].item()
            print(ds_train['HITS']
        ```
        """
        data = dict()

        fields_labels = [
            DATASET_FIELDS.HITS,
            DATASET_FIELDS.PT_VAL,
            DATASET_FIELDS.SIGN,
            DATASET_FIELDS.IS_NULL,
            DATASET_FIELDS.PT_CODE]

        test_fields_labels = [
            DATASET_FIELDS.OMTF_PT,
            DATASET_FIELDS.OMTF_SIGN,
            DATASET_FIELDS.OMTF_QUALITY]

        for k, v in self.dataset.items():
            fields = [
                v[0],
                v[1][:,0],
                v[1][:,3],
                v[3],
                v[4]]
            if k is not DATASET_TYPES.TEST:
                data[k] = dict(zip(fields_labels, fields))
            else:
                test_fields = [
                    v[2][:,1],
                    v[2][:,0],
                    v[2][:,3]]
                data[k] = dict(zip(fields_labels + test_fields_labels, 
                    fields + test_fields))
        if single_file:
            np.savez_compressed(prefix, **data)
        else:
            for k, v in data.items():
                path = prefix + '-' + k.lowercase()
                np.savez_compressed(path, **{k: v})
    

    def save_stats(self, path):
        stats = dict()
        stats[DSET_STAT_FIELDS.TRAIN_EXAMPLES_ORDERING] = self.train_examples_order

        orig = self.histograms[DATA_TYPES.ORIG]
        trans = self.histograms[DATA_TYPES.TRANS]

        stats[DSET_STAT_FIELDS.HISTS_TOTAL_ORIG] = {
            HIST_TYPES.VALS: orig[HIST_TYPES.VALS][HIST_SCOPES.TOTAL],
            HIST_TYPES.AVG: orig[HIST_TYPES.AVG][HIST_SCOPES.TOTAL]}

        stats[DSET_STAT_FIELDS.HISTS_TOTAL_TRANS] = {
            HIST_TYPES.VALS: trans[HIST_TYPES.VALS][HIST_SCOPES.TOTAL],
            HIST_TYPES.AVG: trans[HIST_TYPES.AVG][HIST_SCOPES.TOTAL]}

        stats[DSET_STAT_FIELDS.HISTS_ORIG] = {
            HIST_TYPES.VALS: orig[HIST_TYPES.VALS][HIST_SCOPES.CODE],
            HIST_TYPES.AVG: orig[HIST_TYPES.AVG][HIST_SCOPES.CODE]}

        stats[DSET_STAT_FIELDS.HISTS_TRANS] = {
            HIST_TYPES.VALS: trans[HIST_TYPES.VALS][HIST_SCOPES.CODE],
            HIST_TYPES.AVG: trans[HIST_TYPES.AVG][HIST_SCOPES.CODE]}

        stats[DSET_STAT_FIELDS.HISTS_BINS] = self.bins
        stats[DSET_STAT_FIELDS.MUON_SIGNATURES] = self.signatures
        stats[DSET_STAT_FIELDS.TRANSFORM] = self.transform
        stats[DSET_STAT_FIELDS.TRESHOLD] = self.treshold

        data = {FILE_TYPES.DATASET_STATISTICS: stats}
        np.savez_compressed(path, **data)


    def get_train_examples_ordering(self):
        '''
        Get examples ordering arrays.
        '''
        return (self.train_examples_order[ORD_TYPE.ORIG], 
            self.train_examples_order[ORD_TYPE.SHUF])


    def show(path, n=5):
        npz = np.load(path)
        if DSET_STAT_FIELDS.GROUP_NAME in npz.files:
            OMTFDataset._show_stats(npz[DSET_STAT_FIELDS.GROUP_NAME].item())
        for name in [n for n in vars(DATASET_TYPES) if not n.startswith('_')]:
            if name in npz.files:
                OMTFDataset._show_dataset(name, npz[name].item(), n)


    def _show_dataset(name, data, n):
        print('=' * 10 + ' DATASET ' + name)
        for label, arr in data.items():
            print(label+ ":")
            print(arr[:n])


    def _show_stats(data):
        print('=' * 10 + ' DATASET INFO')
        print("Included signatures:")
        for c, s in data[DSET_STAT_FIELDS.MUON_SIGNATURES]:
            print('pt code: %d sign: %s' % (c, s))

