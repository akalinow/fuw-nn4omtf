# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    *.tfrecord dataset generator.
"""

import sys
import shutil
import numpy as np
import time
import tensorflow as tf
import os


class OMTFDatasetGenerator:
    """
    Class generating a bit customized dataset in multi-file TFRecords format.
    """

    logs = True
    stdlog = sys.stderr

    dir_name_train = "training"
    dir_name_valid = "validation"
    dir_name_test = "testing"


    def __init__(self, npz_files, out_path, events_frac=1., train_frac=0.8, valid_frac=0.1, compression=True, save_full_hits=False):
        """
        Args:
            npz_files: *.npz files list, files are required to be saved using utils function
            out_path: path where TFRecords dataset out of given files should  be saved
            events_frac (optional): what fraction of all events in each files should be saved in dataset
            train_frac (optional): what fraction of saved events should be put in training dataset
            valid_frac (optional): fraction of saved events put in validation set
            compression (optional): saving dataset with compression?
            save_full_hits (optional): save full 18x14 hits tensor for each event or filtered 18x2 without non-meaningful data

        Having #train_frac and #valid_frac, value test_frac is calculated and it's equal: '1 - valid_frac - train_frac'.
        """
        self._check_input_files(npz_files)
        self.in_files = npz_files
        self.out_path = out_path
        
        self.ev_frac = events_frac
        self.ev_train_frac = train_frac
        self.ev_valid_frac = valid_frac
        self._calc_event_frac()

        self.write_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP) if compression else None
        self.save_full_hits = save_full_hits


    def add_input_files(self, npz_files):
        """Add input *.npz file paths."""
        self._check_input_paths(files)
        for f in npz_files:
            self.in_files.append(f)


    def _check_input_paths(self, npz_paths):
        for f in npz_paths: # Check file existence
            if not os.path.exists(f):
                raise FileNotFoundError("File %s doesn't exist" % f)


    def set_events_frac(self, frac):
        self.ev_frac = frac
        self._calc_events_frac()


    def set_train_events_frac(self, frac):
        self.ev_train_frac = frac
        self._calc_events_frac()


    def set_valid_events_frac(self, frac):
        self.ev_valid_frac = frac
        self._calc_events_frac()


    def _calc_events_frac(self):
        assert self.ev_frac <= 1
        assert self.ev_train_frac + self.ev_valid_frac <= 1
        self.ev_test_frac = 1 - train_frac - valid_frac


    def logs_on(self):
        self.logs = True


    def logs_off(self):
        self.logs = False


    def set_log_dest(self, out=sys.stderr):
        self.stdlog = out


    def _log(self, info):
        if self.logs:
            self.stdlog.write(info + '\n')


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


    def _float_feature(self, value):
        """Creates TensorFlow feature from value."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))


    def _save_tfrecords(self, filename, hits, prod, omtf, begin, end):
        """Save single TFRecords file."""

        writer = tf.python_io.TFRecordWriter(filename, self.write_opt)
        for i in range(begin, end):
            features = tf.train.Features(feature={
                    'hits': self._float_feature(hits[i].reshape(-1)),
                    'prod': self._float_feature(prod[i]),
                    'omtf': self._float_feature(omtf[i])
                })
            event = tf.train.Example(features=features)
            writer.write(event.SerializeToString())
        writer.close()

