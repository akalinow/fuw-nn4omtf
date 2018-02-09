# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

"""

import time
import os
import pickle
import tensorflow as tf
from nn4omtf.utils import load_dict_from_npz, float_feature
from nn4omtf.dataset.const import NPZ_FIELDS


class OMTFDataset:
    """Dataset paths and metadata.
    It's used for mapping and file paths extracting.
    """
    
    DEFAULT_PARAMS = {
            'events_frac': 1.0,
            'train_frac': 0.7,
            'valid_frac': 0.15,
            'test_frac': 0.15,
            'compress': False,
    }


    class CONST:
        TF_FILENAME = '{name}-{code}-{sign}.tfrecords'
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
        for k, v in kw.items():
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
        obj.path = path
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


    def add_npz_to_dataset(self, files, verbose=False):
        """Add data examples from npz files.
        Args:
            files: list of npz files to import
        """
        cnt = 1
        for f in files:
            if verbose:
                print("Processing file [{}/{}]: {}".format(cnt, len(files), f))
            self._add_npz_file(f, verbose)
            cnt += 1


    def get_summary(self):
        summary = "name: %s\n" % self.name
        for k, v in self.params.items():
            summary += "{}: {}\n".format(k, v)
        for name, sets in self.sets.items():
            summary += "\nset: {}\n".format(name)
            for el in sets:
                summary += "> events: {events}, code: {code}, sign: {sign}, pt_min: {pt_min}, pt_max: {pt_max}\npath: {path}\n".format(**el)
        return summary


    def __str__(self):
        return self.get_summary()
    
    
    def _save_tfrecords(self, filename, begin, end, **kw):
        """Save single TFRecords file.
        Args:
            filename: TFRecords files
            **kw: data to save
        """
        writer = tf.python_io.TFRecordWriter(filename, self.write_opt)
        for i in range(begin, end):
            feature = dict((k, float_feature(v[i].reshape(-1))) for k, v in kw.items())
            features = tf.train.Features(feature=feature)
            event = tf.train.Example(features=features)
            writer.write(event.SerializeToString())
        writer.close()


    def _add_npz_file(self, f, verbose=False):
        time_start = time.time()
        data = load_dict_from_npz(f)
        name = data[NPZ_FIELDS.NAME]
        code = data[NPZ_FIELDS.PT_CODE]
        sign = data[NPZ_FIELDS.SIGN]
        to_save_list = [NPZ_FIELDS.HITS_FULL, 
                        NPZ_FIELDS.HITS_REDUCED, 
                        NPZ_FIELDS.OMTF,
                        NPZ_FIELDS.PROD]
        data_to_save = dict([(k, data[k]) for k in to_save_list])
        ev_all = data[NPZ_FIELDS.EV_N]
        ev_save = int(self.params['events_frac'] * ev_all)
        if verbose:
            print("Events in file: %d" % ev_all)
            print("Events to save: %d (%0.2f%% of  all)" % (ev_save, ev_save * 100. / ev_all))
        filename = OMTFDataset.CONST.TF_FILENAME.format(
                name=str(name).lower(),
                code=code,
                sign=sign)
        ev_a = 0
        ev_b = 0
        for setname in self.CONST.SET_NAMES:
            filepath = os.path.join(self.path, setname, filename)
            ev = int(ev_save * self.params[setname + '_frac'])
            ev_b += ev
            if verbose:
                print("Saving %d events into '%s' set (%s)" % (ev, setname, filepath))
            self._save_tfrecords(filepath, ev_a, ev_b, **data_to_save)
            ev_a += ev
            info = {
                "path": os.path.join(setname, filename),
                "code": code,
                "sign": sign,
                "name": name,
                "events": ev,
                "pt_min": data[NPZ_FIELDS.PT_MIN],
                "pt_max": data[NPZ_FIELDS.PT_MAX]
            }
            self.sets[setname].append(info)
        time_elapsed = time.time() - time_start
        if verbose:
            print("Finished in: %d''%d'" % (time_elapsed // 60, int(time_elapsed) % 60))

