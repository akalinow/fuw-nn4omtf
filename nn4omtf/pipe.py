# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    Input pipe.
"""

import tensorflow as tf
import numpy as np
import multiprocessing
from nn4omtf.const_dataset import DATASET_TYPES, DATASET_FIELDS

class OMTFInputPipe:
    """
    Input pipe is an abstraction over `*.npz` datasets
    generated with `OMTFDataset`.
    Uses tf.Dataset API to wrap numpy arrays from 
    dataset file and seamlessly feeds neural network.

    # Loading big dataset from file

    In case of not so big datasets whole file is loaded into memory.
    Eventualy, tf.Dataset can use generators which may load data in
    smaller portions. It can be implemented with `mmap_mode` in `np.load`
    method.
    TODO: Implement feeder using `Dataset.from_generator`
    
    # Mapping pt value onto classes
    
    `pt_bins` list must begin with ZERO. 
    First class (pt < 0) is considered as NULL/INCORRECT/UNKNOWN class.
    NULL class has ID = 0.
    All pt >= 0 will be thrown into classes with ID > 0 accordingly to
    pt, charge sign and bin edges values.
    Let `k` be number of bin which are defined by `pt_bins` list values.

    Final pt class is:
    - `2k` if moun has positive charge sign
    - `2k-1` if muon has negative charge sign
    - `0` if muon doesn't have enough data to be classified correctly

    # TEST dataset additional data 
    
    Data labels are created by hand to fix data order in tuple.
    Only those are fetched from dataset file.
    In case of TEST dataset, all additional data will be read by
    statistics and plotter module.
    There's no need to use those data where.
    """

    def __init__(self, npz_path, dataset_type, pt_bins, batch_size=1,
            apply_is_null=True):
        """
        Create input pipe and load data from `*.npz` dataset.
        Args:
            npz_path: dataset file generated with `OMTFDataset`
            dataset_type: value from `DATASET_TYPES`
            pt_bins: muon pt classe bins' edges list
            batch_size: size of batch
            apply_is_null: set muon class to 0 if `is_null` arr element is null
        """
        types = [v for k, v in vars(DATASET_TYPES).items() if not k.startswith('_')]
        assert dataset_type in types, dataset_type + ' is not valid dataset type!'
        assert pt_bins[0] == 0, 'First pt bin edge in not ZERO!'

        self.batch_size = batch_size
        self.apply_is_null = apply_is_null
        self.pt_bins = pt_bins
        self.class_n = 2 * len(pt_bins) + 1
        self.path = npz_path
        self.data_labels = [
            DATASET_FIELDS.HITS,
            DATASET_FIELDS.PT_VAL,
            DATASET_FIELDS.SIGN,
            DATASET_FIELDS.IS_NULL]

        print('Loading `%s` data from `%s`...' % (dataset_type, npz_path))
        self.dataset = np.load(npz_path)[dataset_type].item()

        self.iterator = self.build_pipe(dataset_type, **self.dataset)
        self.initializer = self.iterator.initializer
        self.next_op = self.iterator.get_next()
        self.session = None


    def build_pipe(self, dataset_type, **kw):
        """
        Build input pipe.
        Args:
            dataset_type: value from `DATASET_TYPES`
            kw: entries from given type of dataset
        Returns:
            dataset iterator
        """
        cores_count = max(multiprocessing.cpu_count() // 4, 1)

        def get_pt_class(pv, sign, is_null):
            """
            Helper function calculating correct muon class.
            """
            c = np.digitize(pv, self.pt_bins)
            c = np.where(sign > 0, 2 * c, 2 * c - 1)
            if self.apply_is_null:
                c = np.where(is_null, 0, c)
            return c.astype(np.int32)

        pt_to_class_fn = lambda p, s, n:  tf.py_func(get_pt_class,
                [p, s, n],
                tf.int32,
                stateful=False,
                name='pt_class')

        map_fn = lambda h, p, s, n: (h, pt_to_class_fn(p, s, n))
        
        with tf.device('/cpu:0'):
            with tf.name_scope('pipe-' + dataset_type.lower()):
                datasets = [tf.data.Dataset.from_tensor_slices(kw[k]) 
                        for k in self.data_labels]
                dataset = tf.data.Dataset.zip(tuple(datasets))
                # Batch elements before map to hide function call overhead
                dataset = dataset.batch(10000)
                dataset = dataset.map(map_func=map_fn, num_parallel_calls=None)
                dataset = dataset.apply(tf.contrib.data.unbatch())
                dataset = dataset.prefetch(buffer_size=4*self.batch_size)
                dataset = dataset.batch(self.batch_size)
                iterator = dataset.make_initializable_iterator()
        return iterator


    def initialize(self, session):
        """
        Initialize input pipe.
        Args:
            session: tf session
        """
        session.run(self.iterator.initializer)
        self.session = session


    def fetch(self):
        """
        Fetch examples from input pipe.
        """
        assert self.session is not None, "Input pipe must be initialized!"
        try:
            return self.session.run(self.next_op)
        except tf.errors.OutOfRangeError:
            return None

    def get_initializer_and_op(self):
        return self.initializer, self.next_op
   

# ===== TEST

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Input pipe test")
    parser.add_argument('--examples', type=int, default=5, 
        help='Number of examples to fetch')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--dont_apply_is_null', action="store_false")
    parser.add_argument('type', choices=['TRAIN', 'VALID', 'TEST'])
    parser.add_argument('dataset_file')
    parser.add_argument('pt_bins', nargs='+', type=float)

    FLAGS = parser.parse_args()
    print(FLAGS)

    pipe = OMTFInputPipe(FLAGS.dataset_file, FLAGS.type, FLAGS.pt_bins,
            batch_size=FLAGS.batch_size, apply_is_null=FLAGS.dont_apply_is_null)
    with tf.Session() as sess:
        pipe.initialize(sess)
        for _ in range(FLAGS.examples):
            data = pipe.fetch()
            print(data)

    

