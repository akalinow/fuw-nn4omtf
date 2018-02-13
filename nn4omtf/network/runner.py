# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    Neural network based muon momentum classifier trainer and tester.
"""

import tensorflow as tf
import time
import os
import threading
import subprocess

from nn4omtf.dataset import OMTFDataset
from nn4omtf.network import OMTFNN
from nn4omtf.utils import init_uninitialized_variables

from nn4omtf.network.runner_helpers import *
from nn4omtf.network.input_pipe import OMTFInputPipe


class OMTFRunner:
    """OMTFRunner  is base class for neural nets training and testing.
    It's reading data from TFRecord OMTF dataset created upon
    OMTF simulation data. Input pipe is automaticaly established.
    Main goal is to prepare 'placeholder' and provide universal
    input and output interface for different net's architectures.

    Dataset can be created using OMTFDataset class provided in this package.
    """

    DEFAULT_PARAMS = {
        "valid_batch_size": 1000000,
        "batch_size": 1000,
        "sess_prefix": "",
        "shuffle": False,
        "acc_ival": 10,
        "run_meta_ival": 100,
        "reps": 1,
        "steps": -1,
        "logs": None,
        "verbose": False
    }

    def __init__(self, dataset, network):
        """Create runner instance.
        Args:
            dataset: OMTFDataset object, data source
            network: OMTFNN object, working object
        """
        self.dataset = dataset
        self.network = network

    def test(self, **kw):
        print("TEST", self.dataset.name, self.network.name)

    def train(self, **kw):
        """Run model training."""
        assert self.dataset is not None, "Run failed! Setup dataset first."
        assert self.network is not None, "Run failed! Setup model first."
        tf.reset_default_graph()

        def get_kw(opt):
            return OMTFRunner.DEFAULT_PARAMS[opt] if opt not in kw else kw[opt]

        verbose = get_kw('verbose')

        def vp(s):
            if verbose:
                print("OMTFRunner: " + s)

        # Preload parameters
        check_accuracy_every = get_kw('acc_ival')
        collect_run_metadata_every = get_kw('run_meta_ival')
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        sess_pref = get_kw('sess_prefix')
        sess_name = "{}{}_{}".format(
            sess_pref,
            self.network.name,
            timestamp)
        batch_size = get_kw('batch_size')
        shuffle = get_kw('shuffle')
        reps = get_kw('reps')
        steps = get_kw('steps')
        log_dir = get_kw('logs')
        in_type = self.network.in_type
        out_class_bins = self.network.pt_class
        out_len = len(out_class_bins) + 1

        vp("Preparing training session: %s" % sess_name)
        vp("Creating input pipes...")
        with tf.name_scope("input_pipes"):
            # Input pipes configuration
            train_pipe = OMTFInputPipe(
                    dataset=self.dataset,
                    name='train',
                    hits_type=in_type,
                    out_class_bins=out_class_bins,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    reps=reps)
            valid_pipe = OMTFInputPipe(
                    dataset=self.dataset,
                    name='valid',
                    hits_type=in_type,
                    batch_size=OMTFRunner.DEFAULT_PARAMS['valid_batch_size'],
                    out_class_bins=out_class_bins)

        with tf.Session() as sess:
            meta_graph, net_in, net_out = self.network.restore(
                    sess=sess, sess_name=sess_name)
            initialized = sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            vp("Loaded model: %s" % self.network.name)
            labels = tf.placeholder(tf.int8, shape=[None, out_len],
                                    name="labels")

            train_op = setup_trainer(net_out, labels)
            accuracy_op = setup_accuracy(net_out, labels)
            summary_op = tf.summary.merge_all()
            init_uninitialized_variables(sess, initialized)
            train_pipe.initialize(sess)

            if log_dir is not None:
                valid_path = os.path.join(log_dir, sess_name, 'valid')
                train_path = os.path.join(log_dir, sess_name, 'train')
                valid_summary_writer = tf.summary.FileWriter(valid_path)
                train_summary_writer = tf.summary.FileWriter(
                    train_path, sess.graph)

            last = time.time()
            start = time.time()
            i = 1
            
            vp("Training started at %s" %
               time.strftime("%Y-%m-%d %H:%M:%S"))
            while i <= steps or steps < 0:
                train_in, train_labels = train_pipe.fetch()
                if train_in is None:
                    vp("Train dataset is empty!")
                    break

                train_feed_dict = {net_in: train_in, labels: train_labels}
                args = {
                    'feed_dict': train_feed_dict,
                    'options': None,
                    'run_metadata': None
                }
                if collect_run_metadata_every is not None:
                    if i % collect_run_metadata_every == 0:
                        args['options'] = tf.RunOptions(
                            trace_level=tf.RunOptions.FULL_TRACE)
                        args['run_metadata'] = tf.RunMetadata()
                summary, _ = sess.run([summary_op, train_op], **args)
                if log_dir is not None:
                    train_summary_writer.add_summary(summary, i)
                    if args['options'] is not None:
                        train_summary_writer.add_run_metadata(
                            args['run_metadata'], "step-%d" % i)

                if i % check_accuracy_every == 0:
                    summaries, accuracy = check_accuracy(
                            session=sess,
                            pipe=valid_pipe,
                            net_in=net_in,
                            labels=labels,
                            summary_op=summary_op,
                            accuracy_op=accuracy_op)
                    if log_dir is not None:
                        for s in summaries:
                            valid_summary_writer.add_summary(s, i)
                    self.network.add_summaries(summaries, i)
                    self.network.add_log(accuracy, i, sess_name)
                    vp("Step %d, accuracy: %f" % (i, accuracy))
                now = time.time()
                elapsed = now - last
                last = now
                vp("Time elapsed: %.3f, last step: %.3f" %
                    (now - start, elapsed))
                i += 1

            vp("Training finished at %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
            # Save network state after training
            self.network.finish()
            self.network.save()
            vp("Model saved!")
