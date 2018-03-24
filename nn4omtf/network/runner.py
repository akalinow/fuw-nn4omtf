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
from nn4omtf.utils import init_uninitialized_variables, dict_to_object

from nn4omtf.network.runner_helpers import collect_statistics,\
        setup_accuracy, setup_trainer
from nn4omtf.network.input_pipe import OMTFInputPipe
from nn4omtf.network.input_pipe_const import PIPE_EXTRA_DATA_NAMES


class OMTFRunner:
    """OMTFRunner  is base class for neural nets training and testing.
    It's reading data from TFRecord OMTF dataset created upon
    OMTF simulation data. Input pipe is automaticaly established.
    Main goal is to prepare 'placeholder' and provide universal
    input and output interface for different net's architectures.

    Dataset can be created using OMTFDataset class provided in this package.
    """

    DEFAULT_PARAMS = {
        "valid_batch_size": 10000,
        "batch_size": 1000,
        "sess_prefix": "",
        "shuffle": False,
        "acc_ival": 100,
        "get_meta_ival": 500,
        "epochs": 1,
        "steps": -1,
        "logdir": None,
        "verbose": False,
        "learning_rate": 0.001
    }

    def __init__(self, dataset, network, **kw):
        """Create parametrized runner instance.
        Args:
            dataset: OMTFDataset object, data source
            network: OMTFNN object, working object
            **kw: additional keyword args,
                substitute defaults params if key matches
        """
        self.dataset = dataset
        self.network = network

        # Setup parameters
        params = OMTFRunner.DEFAULT_PARAMS
        for key, val in kw.items():
            if key in params:
                params[key] = val
        # Setup runner variables
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        pref = params['sess_prefix']
        out_class_bins = self.network.pt_class
        var = {
            'timestamp': timestamp,
            'sess_name': "{}{}_{}".format(pref, self.network.name, timestamp),
            'in_type': self.network.in_type,
            'out_class_bins': out_class_bins,
            'out_len': len(out_class_bins) + 1
        }
        for k, v in var.items():
            params[k] = v
        self.params = params


    def _get_verbose_printer(self):
        """Get verbose printer.
        Returns:
            Lambda which prints out its argument if verbose flag is set
        """
        vp = lambda s: print("OMTFRunner: " + s) if self.params['verbose'] else None
        return vp


    def _update_params(self, params_dict):
        """Update runner parameters
        Args:
            params_dict: dict with new parameters
        Note that some values (like session name) won't be updated.
        """
        for k, v in params_dict.items():
            if k in self.params:
                self.params[k] = v


    def _start_clock(self):
        """Start runner timers.
        Returns:
            dict with start timestamp
        """
        self.start = time.time()
        self.last = self.start
        self.start_datetime = time.strftime("%Y-%m-%d-%H-%M-%S")
        return {'start_datetime': self.start_datetime}
    

    def _next_tick(self):
        """Update clock.
        Returns dict with useful falues which can be directly
        passed into string.format() method.
        Returns:
            dict of:
                - datetime string
                - elapsed time (since start)
                - last time (between ticks)
        """
        datetime = time.strftime("%Y-%m-%d-%H-%M-%S")
        now = time.time()
        last = now - self.last
        elapsed = now - self.start
        self.last = now
        res = {
            'start_datetime': self.start_datetime,
            'datetime': datetime,
            'elapsed': elapsed,
            'last': last
        }
        return res


    def show_params(self):
        print("==== OMTFRunner configuration")
        for k, v in self.params.items():
            print("> {:.<20}:{}".format(k, v))
        print("=============================")


    def train(self, **kw):
        """Run model training.
        Training logs are saved on disk if `logs` flag is set.
        Short summaries are always appended to OMTFNN object.
        
        Args:
            **kw: additional args which can update previously set params
        """
        self._update_params(kw)
        opt = dict_to_object(self.params)
        opt.sess_name += "/train"
        vp = self._get_verbose_printer()

        tf.reset_default_graph()
        vp("Preparing training session: %s" % opt.sess_name)
        vp("Creating input pipes...")
        with tf.name_scope("input_pipes"):
            # Input pipes configuration
            train_pipe = OMTFInputPipe(
                    dataset=self.dataset,
                    name='train',
                    hits_type=opt.in_type,
                    out_class_bins=opt.out_class_bins,
                    batch_size=opt.batch_size,
                    shuffle=opt.shuffle,
                    reps=opt.epochs)
            valid_pipe = OMTFInputPipe(
                    dataset=self.dataset,
                    name='valid',
                    hits_type=opt.in_type,
                    batch_size=opt.valid_batch_size,
                    out_class_bins=opt.out_class_bins)
        
        self.show_params()

        with tf.Session() as sess:
            _, net_in, net_pt_out, net_sgn_out = self.network.restore(
                    sess=sess, sess_name=opt.sess_name)

            # Get collection of initialized variables
            init = sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            vp("Loaded model: %s" % self.network.name)

            # ==== NETWORK PLACEHOLDERS
            pt_labels = tf.placeholder(tf.int8, shape=[None, opt.out_len],
                                    name="pt_labels")
            sgn_labels = tf.placeholder(tf.int8, shape=[None, 2],
                                    name="sgn_labels")
            net_pholders = [
                net_in,
                pt_labels,
                sgn_labels
            ]

            # ==== SETUP TRAINER NODES
            pt_train_op, sgn_train_op = setup_trainer(
                    net_pt_out=net_pt_out,
                    net_sgn_out=net_sgn_out,
                    labels_pt=pt_labels,
                    labels_sgn=sgn_labels,
                    learning_rate=opt.learning_rate)

            pt_acc_op, sgn_acc_op, pt_class_op, sgn_class_op = setup_accuracy(
                    net_pt_out=net_pt_out,
                    net_sgn_out=net_sgn_out,
                    pt_labels=pt_labels,
                    sgn_labels=sgn_labels)

            summary_op = tf.summary.merge_all()

            # At after all, initialize new nodes
            init_uninitialized_variables(sess, initialized=init)
            train_pipe.initialize(sess)

            if opt.logdir is not None:
                valid_path = os.path.join(opt.logdir, opt.sess_name, 'valid')
                train_path = os.path.join(opt.logdir, opt.sess_name, 'train')
                valid_summary_writer = tf.summary.FileWriter(valid_path)
                train_summary_writer = tf.summary.FileWriter(
                    train_path, sess.graph)

            i = 1            
            stamp = self._start_clock()
            vp("{start_datetime} - training started".format(**stamp))
            while i <= opt.steps or opt.steps < 0:
                # Fetch next batch of input data and its labels
                # Ignore extra data during trainings 
                train_data = train_pipe.fetch()
                if train_data[0] is None:
                    vp("Train dataset is empty!")
                    break

                # Prepare training feed dict
                train_feed_dict = dict([(k, v) for k, v in zip(net_pholders, train_data[:-1])])

                # Configure extra options
                args = {
                    'feed_dict': train_feed_dict,
                    'options': None,
                    'run_metadata': None
                }
    
                if opt.get_meta_ival is not None:
                    if i % opt.get_meta_ival == 0:
                        args['options'] = tf.RunOptions(
                            trace_level=tf.RunOptions.FULL_TRACE)
                        args['run_metadata'] = tf.RunMetadata()
                
                # Do mini-batch iteration
                summary, _, _ = sess.run(
                    [summary_op, pt_train_op, sgn_train_op], 
                    **args
                )
                
                # Save training logs for TensorBoard
                if opt.logdir is not None:
                    train_summary_writer.add_summary(summary, i)
                    if args['options'] is not None:
                        train_summary_writer.add_run_metadata(
                            args['run_metadata'], "step-%d" % i)

                self._next_tick()
                if i % opt.acc_ival == 0:
                    summaries, acc_d, nn_stats = collect_statistics(
                            sess=sess,
                            sess_name=opt.sess_name,
                            pipe=valid_pipe,            # test using valid set
                            net_pholders=net_pholders,  # net placeholders
                            net_pt_out=net_pt_out,      # pt logits out
                            net_sgn_out=net_sgn_out,    # sgn logits out
                            net_pt_class=pt_class_op,   # pt class out
                            net_sgn_class=sgn_class_op, # sgn class out
                            pt_acc=pt_acc_op, 
                            sgn_acc=sgn_acc_op,
                            summary_op=summary_op)      # summary operator
                    
                    nn_stats.set_bins(opt.out_class_bins)

                    if opt.logdir is not None:
                        for s in summaries:
                            valid_summary_writer.add_summary(s, i)
                    
                    self.network.add_summaries(summaries, i)
                    self.network.add_log(acc_d, i, opt.sess_name)
                    self.network.add_statistics(nn_stats)
                    print(nn_stats)
                    
                    stamp = self._next_tick()
                    vp("Validation @ step {step}".format(step=i))
                    vp("Now: {datetime}, elapsed: {elapsed:.1f} sec.".format(**stamp))
                    vp("Validation run took: {last:.1f} sec.".format(**stamp))
                    vp("Accuracy:\n\tpt: {pt:f}\n\tsgn: {sgn:f}\n".format(
                        **acc_d))

                i += 1

            # End of main while loop, training finished
            stamp = self._next_tick()
            vp("{datetime} - training finished!".format(**stamp))
            vp("Training took: {elapsed:.1f} sec,".format(**stamp))
            vp("Steps in total: %d" % i)

            # Save network state after training
            self.network.finish()
            self.network.save()
            vp("Model saved!")


    def test(self, **kw):
        """Test loaded model on test dataset.
        Args:
            **kw: additional args which can update previously set params
        Returns:
            Short dict-summary from whole run.
        """
        self._update_params(kw)
        opt = dict_to_object(self.params)
        opt.sess_name += "/test"
        vp = self._get_verbose_printer()

        tf.reset_default_graph()
        vp("Preparing test session: %s" % opt.sess_name)
        vp("Creating input pipe...")
        with tf.name_scope("input_pipes"):
            test_pipe = OMTFInputPipe(
                    dataset=self.dataset,
                    name='test',
                    hits_type=opt.in_type,
                    batch_size=opt.valid_batch_size,
                    out_class_bins=opt.out_class_bins)

        self.show_params()

        with tf.Session() as sess:
            _, net_in, net_pt_out, net_sgn_out = self.network.restore(
                    sess=sess, sess_name=opt.sess_name)

            vp("Loaded model: %s" % self.network.name)
            pt_labels = tf.placeholder(tf.int8, shape=[None, opt.out_len],
                                    name="pt_labels")

            sgn_labels = tf.placeholder(tf.int8, shape=[None, 2],
                                    name="sgn_labels")
            
            pt_acc_op, sgn_acc_op, pt_class_op, sgn_class_op = setup_accuracy(
                    net_pt_out=net_pt_out,
                    net_sgn_out=net_sgn_out,
                    pt_labels=pt_labels,
                    sgn_labels=sgn_labels)
             
            summary_op = tf.summary.merge_all()
            
            stamp = self._start_clock()
            vp("{start_datetime} - test started!".format(**stamp))

            summaries, accuracy, nn_stats = collect_statistics(
                    sess=sess,
                    sess_name=opt.sess_name,
                    pipe=test_pipe,             # test using valid set
                    net_in=net_in,              # net input tensor
                    net_pt_out=net_pt_out,      # pt logits out
                    net_sgn_out=net_sgn_out,    # sgn logits out
                    net_pt_class=pt_class_op,   # pt class out
                    net_sgn_class=sgn_class_op, # sgn class out
                    pt_labels=pt_labels,        # labels placeholder
                    sgn_labels=sgn_labels,
                    summary_op=summary_op)      # summary operator

            nn_stats.set_bins(opt.out_class_bins)

            self.network.add_log(accuracy, 1, opt.sess_name)
            self.network.add_statistics(nn_stats)
            print(nn_stats)

            stamp = self._next_tick()

            vp("{datetime} - test finished!".format(**stamp))
            vp("Test run took: {last:.1f} sec.".format(**stamp))
            vp("Accuracy: %f" % accuracy)

        res = {
            'sess_name': opt.sess_name,
            'accuracy': accuracy,
            'model': self.network.name
        }
        return res


    def test_many_models(dataset, models_list, **kw):
        """Test all provided models on test dataset.
        Args:
            dataset: dataset to test on
            models_list: list of OMTFNN objects
            **kw: runner params
        Returns:
            list of #OMTFRunner.test results
        """
        test_results = []
        for model in models_list:
            runner = OMTFRunner(dataset, model, **kw)
            res = runner.test()
            test_results.append(res)
        return test_results

