# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

"""

import os
import pickle
from nn4omtf.utils import load_dict_from_npz, float_feature


class OMTFDataset:
    """Dataset paths and metadata.
    It's used for mapping and file paths extracting.
    """
    
    DEFAULT_PARAMS = {
            'events_frac': 1.0,
            'train_frac': 0.7,
            'valid_frac': 0.15,
            'compress': False,
    }


    class CONST:
        OBJ_FILENAME = ".dataset"
        SET_NAMES = ['train', 'valid', 'test']


    def __init__(self, name, path, **kw):
        """Create object with defined parameters.
        Args:
            name: dataset name
            path: dataset directory (parent of dataset root dir)
            compress: use compressed TFRecords
            events_frac: float in range [0, 1], fraction of all events in *.npz files
                which will be stored in TFRecords
            train_frac: float in range [0, 1], fraction of saved events used as train dataset
            valid_frac: float in range [0, 1], fraction of saved events used as validation dataset
        
        Having #train_frac and #valid_frac, value test_frac is calculated and it's equal: '1 - valid_frac - train_frac'.
        """
        self.name = name
        self.path = os.path.join(path, name)
        self.objfile = os.path.join(self.path, OMTFDataset.CONST.OBJ_FILENAME)
        assert not os.path.exists(self.objfile), "Directory already contains dataset object!"
        self.params = OMTFDataset.DEFAULT_PARAMS
        for k, v in kw:
            if k in self.params:
                self.params[k] = v
        self._calc_events_frac()

        self.sets = dict([(n, []) for n in OMTFDataset.CONST.SET_NAMES])
        for el in OMTFDataset.CONST.SET_NAMES:
            p = os.path.join(self.path, el)
            os.makedirs(p, exist_ok=True)

        self.write_opt = None
        if self.params['compress']:
            self.write_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    

    def load(path):
        """Load dataset object from disk.
        Args:
            path: source path where dataset object is stored
        Returns:
            Loaded object
        """
        filepath = os.path.join(path, OMTFDataset.CONST.OBJ_FILENAME)
        assert os.path.exists(filepath), "Directory doesn't contain dataset object!"
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        return obj


    def save(self):
        """Pickle object and store on disk."""
        filepath = os.path.join(self.path, OMTFDataset.CONST.OBJ_FILENAME)
        os.makedirs(self.path, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(obj=self, file=f)
        
    def _calc_events_frac(self):
        assert 0 <= self.params['events_frac'] <= 1, "Events fraction must be float in range [0, 1]"
        assert 0 <= (self.params['train_frac'] + self.params['valid_frac']) <= 1, "train + valid frac must be in [0, 1] range"
        self.params['test_frac'] = 1 - self.params['train_frac'] - self.params['valid_frac']


    def get_dataset(self, name='train', ptc_min=None, ptc_max=None):
        """Get list TFRecords' paths.
        Filter files which doesn't contain events with muons' pT code
        between given limits.
        Args:
            name: dataset name
            ptc_min: minimal pt
            ptc_max: maximal pt
        Returns:
            list of files, length of list
        """
        assert name in self.sets, "Dataset {} doesn't exist!".format(name)
        l = []
        for el in self.sets[name]:
            ptc = el['code']
            if ptc >= ptc_min and ptc <= ptc_max:
                l.append(el['path'])
        return l, len(l)


    def add_npz_to_dataset(self, files):
        """Add data examples from npz files.
        Args:
            files: list of npz files to import
        """
        for f in files:
            self._add_npz_file(f)

    def _add_npz_file(self, f):
        data = load_dict_from_npz(f)



    def get_summary(self):
        summary = "name: %s\n" % self.name
        for k, v in self.params.items():
            summary += "{}: {}\n".format(k, v)
        for name, sets in self.sets.items():
            summary += "\nset: {}\n".format(name)
            for el in sets:
                summary += "> code: {code}, pt_min: {pt_min}, pt_max: {pt_max}\npath: {path}\n".format(**el)
        return summary


    def __str__(self):
        return self.get_summary()
    
    


    def _save_tfrecords(self, filename, **kw):
        """Save single TFRecords file.
        Args:
            filename: TFRecords files
            **kw: data to save
        """
        writer = tf.python_io.TFRecordWriter(filename, self.write_opt)
        for i in range(begin, end):
            feature = {k, float_feature(v[i]) for k, v in kw.items()}
            features = tf.train.Features(feature)
            event = tf.train.Example(features=features)
            writer.write(event.SerializeToString())
        writer.close()

    def generate(self):
        """Start generating dataset."""
        self.time_start = time.time()
        self._log("Set generation started.")

        # If directory exists - remove subtree
        if os.path.exists(self.out_path):
            shutil.rmtree(self.out_path)

        # Prepare output directories
        self.out_path_train = os.path.join(self.out_path, self.dir_name_train)
        self.out_path_valid = os.path.join(self.out_path, self.dir_name_valid)
        self.out_path_test = os.path.join(self.out_path, self.dir_name_test)
        os.makedirs(self.out_path_train)
        os.makedirs(self.out_path_valid)
        os.makedirs(self.out_path_test)

        self.time_last = time.time()
        for f in self.in_files:
            self._log("Converting file: %s" % f)

            self._convert_file(f)

            self.time_now = time.time()
            self._log("Done... [Tfile: %.3fs, Tall: %.3fs]" % (self.time_now - self.time_last, self.time_now - self.time_start))
            self.time_last = self.time_now

        self._log("Dataset saved in %s" % self.out_path)
        self._log("Total time: %.3fs" % (self.time_last - self.time_start))


    def _convert_file(self, path):
        """Convert single file."""
        
        in_data = np.load(path, encoding='bytes')
        name = in_data['name']
        val = in_data['val']
        sign = in_data['sign']
        prod = in_data['prod']
        omtf = in_data['omtf']
        hits = in_data['hits' if self.save_full_hits else 'hits2']

        ev_all = prod.shape[0]
        ev_save = int(ev_all * self.ev_frac)
        ev_train = int(ev_save * self.ev_train_frac)
        ev_valid = int(ev_save * self.ev_valid_frac)
        ev_test = ev_all - ev_train - ev_valid
        suffix =  '%s-%d-%s.tfrecords' % (str(name).lower(), val, sign)
        out_file_train = os.path.join(self.out_path_train, suffix)
        out_file_valid = os.path.join(self.out_path_valid, suffix)
        out_file_test = os.path.join(self.out_path_test, suffix)

        self._print_file_metric(name, val, sign, ev_all, ev_save, ev_train, ev_valid, ev_test, self.out_path, suffix)

        self._save_tfrecords(out_file_train, hits, prod, omtf, 0, ev_train)
        self._save_tfrecords(out_file_valid, hits, prod, omtf, ev_train, ev_train + ev_valid)
        self._save_tfrecords(out_file_test, hits, prod, omtf, ev_train + ev_valid, ev_save)


    def _print_file_metric(self, name, val, sign, ev_all, ev_save, ev_train, ev_valid, ev_test, out_path, suffix):
        info = "input file: %s\n" % name
        info += "output path: %s\n" % out_path
        info += "output file suffix: %s\n " % suffix
        info += "pt code: %d\n" % val
        info += "charge: %s\n" % sign
        info += "events:\n all: %d\n" % ev_all
        info += " to save: %d\n" % ev_save
        info += " in train set: %d\n" % ev_train
        info += " in valid set: %d\n" % ev_valid
        info += " in test set: %d\n" % ev_test
        self._log(info)

