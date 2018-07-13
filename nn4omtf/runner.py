# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    Neural network based muons' momentum classifier trainer and tester.
"""

import tensorflow as tf
import numpy as np
import time
from nn4omtf import OMTFInputPipe
from nn4omtf.utils import dict_to_object, to_sec
from nn4omtf.const_dataset import DATASET_TYPES

class OMTFRunner:

    LOG_TEMPLATE = '{:^7s}, epoch: {:4d} batch: {:4d} loss: {:.4f} acc: {:.4f}'


    def print_log(self, phase, epoch, n, loss, acc):
        print(OMTFRunner.LOG_TEMPLATE.format(phase, epoch, n, loss, acc))


    def timer_start(self, time_limit=None):
        self.time_start = time.time()
        self.time_last = self.time_start
        self.time_ival = 0
        self.time_elapsed = 0
        if time_limit is None:
            self.time_limit = None
        else:
            print("Time limit was set to " + time_limit)
            self.time_limit = self.time_start + to_sec(time_limit)


    def timer_tick(self):
        now = time.time()
        self.time_ival = now - self.time_last
        self.time_elapsed = now - self.time_start
        self.time_last = now
        print("Elapsed [s]: %f, Interval [s]: %f" % (self.time_elapsed, self.time_ival))


    def timer_should_stop(self):
        if self.time_limit is None:
            return False
        else:
            return time.time() > self.time_limit


    def train(self, model, no_checkpoints=False, time_limit=None, epochs=1, 
            train_summary_ival=None, validation_ival=None, **opts):
        """
        Run model training.
        Args:
            model: OMTFModel instance
            no_checkpoints: do not do checkpoints
            time_limit: set training time limit, string format is  `H+:MM:SS`
            epochs: epochs to process, if `None` train infinitely
            train_summary_ival: get train summary batches interval
            validation_ival: batches interval between validations, if `None` 
                validation is run only at the end of epoch
        """
        get_def = lambda x, y: y if x is None else x
        self.train_summary_ival = get_def(train_summary_ival, 1000)
        self.validation_ival = get_def(validation_ival, 10000)

        self.model = model
        self._build()

        assert self.model_config.ds_train is not None, "TRAIN dataset path cannot be None!"
        assert self.model_config.ds_valid is not None, "VALID dataset path cannot be None!"
        with tf.name_scope("input_pipes"):
            self.pipe_train = OMTFInputPipe(self.model_config.ds_train, 
                    DATASET_TYPES.TRAIN, self.pt_bins, 
                    batch_size=self.model_hparams.batch_size)
            self.pipe_valid = OMTFInputPipe(self.model_config.ds_valid, 
                    DATASET_TYPES.VALID, self.pt_bins, 
                    batch_size=self.model_hparams.batch_size)
            t_init, t_next = self.pipe_train.get_initializer_and_op()

        with tf.Session() as sess:
            if not self.model.restore(sess):
                tf.global_variables_initializer().run()

            epoch_n = 0
            batch_n = 0
            should_stop = False
            self.timer_start(time_limit=time_limit)
            try:
                while epochs is None or epoch_n < epochs:
                    epoch_n += 1
                    try:
                        t_init.run()
                        print("Epoch %d started!" % epoch_n)
                        while not should_stop:
                            batch_n += 1
                            xs, ys = sess.run(t_next)
                            feed = {self.x_ph: xs, 
                                self.y_ph: ys, 
                                self.training_ind_ph: True}
                            if batch_n % self.train_summary_ival == 0:
                                run_list = [
                                    self.ops.loss,
                                    self.ops.acc,
                                    self.ops.t_summaries,
                                    self.ops.step
                                ]
                                b_loss, b_acc, b_summ, _ = sess.run(run_list, 
                                        feed_dict=feed)
                                self.print_log('TRAIN', epoch_n, batch_n, b_loss, b_acc)
                                self.model.tb_add_summary(batch_n, b_summ)
                            else:
                                sess.run(self.ops.step, feed_dict=feed)

                            if batch_n % self.validation_ival == 0:
                                v_loss, v_acc, v_summ = self._validate(sess)
                                self.print_log('VALID', epoch_n, batch_n, v_loss, v_acc)

                            should_stop = self.timer_should_stop()

                    except tf.errors.OutOfRangeError:
                        print("Epoch %d - finished!" % epoch_n)
                        v_loss, v_acc, v_summ = self._validate(sess)
                        self.model.tb_add_summary(batch_n, v_summ)
                        self.print_log('VALID', epoch_n, batch_n, v_loss, v_acc)

                        if not no_checkpoints:
                            self.model.save_model(sess)

                    self.timer_tick()
                    print("Mean sec. per batch: %f" % (self.time_elapsed / batch_n))
                    if should_stop:
                        print("Time limit reached!")
                        break

            except KeyboardInterrupt:
                print("Training stopped by user!")

            if not no_checkpoints:
                self.model.save_model(sess)
            self.timer_tick()
            print("Mean sec. per batch: %f" % (self.time_elapsed / batch_n))


    def test(self, model, note='', **opts):
        """
        Run model test.
        Pass whole TEST dataset through network and save raw logits.
        Args:
            model: OMTFModel instance
            note: note to store along with results array
        """
        test_batch_size = 512
        self.model = model
        self._build()

        assert self.model_config.ds_test is not None, "TEST dataset path cannot be None!"

        with tf.name_scope("input_pipes"):
            self.pipe_test = OMTFInputPipe(self.model_config.ds_test, 
                    DATASET_TYPES.TEST, self.pt_bins, 
                    batch_size=test_batch_size)
            t_init, t_next = self.pipe_test.get_initializer_and_op()

        results = None

        with tf.Session() as sess:
            if not self.model.restore(sess):
                print("Test aborted! Cannot restore model!")
                exit(1)

            batch_n = 0
            self.timer_start()
            try:
                t_init.run()
                self.ops.metrics_init.run()
                print("Test started!")
                while True:
                    batch_n += 1
                    xs, ys = sess.run(t_next)
                    feed = {self.x_ph: xs, 
                        self.y_ph: ys, 
                        self.training_ind_ph: False}
                    run_list = [
                        self.out_logits,
                        self.ops.metrics_update,
                    ]
                    outs, _ = sess.run(run_list, 
                            feed_dict=feed)
                    if results is None:
                        results = outs
                    else:
                        results = np.concatenate((results, outs), axis=0)

            except KeyboardInterrupt:
                print("Test phase stopped by user!")
                exit(0)

            except tf.errors.OutOfRangeError:
                pass
            self.timer_tick()
            run_list = [
                self.ops.loss_cum,
                self.ops.acc_cum]
            loss, acc = sess.run(run_list)
            print("Test finished!")
            print("Mean sec. per batch: %f" % (self.time_elapsed / batch_n))
            self.print_log("TEST", 1, batch_n, loss, acc)
            self.model.save_test_results(results, note)


    def _validate(self, sess):
        """
        Run model validation on VALID dataset.
        Args:
            sess: open TF session
        Returns:
            Cummulative validation results over whole dataset
            tuple (loss, accuracy, TB summaries includes loss and acc)
        """
        v_init, v_next = self.pipe_valid.get_initializer_and_op()
        v_init.run()
        self.ops.metrics_init.run()
        try:
            while True:
                xs, ys = sess.run(v_next)
                feed = {self.x_ph: xs, 
                    self.y_ph: ys, 
                    self.training_ind_ph: False}
                sess.run(self.ops.metrics_update, feed_dict=feed)
        except tf.errors.OutOfRangeError:
            pass
        run_list = [
            self.ops.loss_cum,
            self.ops.acc_cum,
            self.ops.v_summaries]
        return sess.run(run_list)


    def _build(self):
        tf.reset_default_graph()

        builder_func = self.model.get_builder_func()
        self.model_config = self.model.get_config()
        self.model_hparams = self.model.get_hparams()

        # HIST array placeholder, dim(x) = 3, batch of 2D HITS arrays
        HITS_REDUCED_SHAPE = [18, 2]
        self.x_ph = tf.placeholder(tf.float32)
        # Labels placeholder - dim(y) = 1, batch of out class indexes
        self.y_ph = tf.placeholder(tf.int32, shape=[None])
        # Training phase indicator - boolean
        self.training_ind_ph = tf.placeholder(tf.bool)


        device = '/cpu:0'
        if self.model_config.gpu:
            device = '/gpu:0'

        with tf.device(device):
            self.out_logits, self.pt_bins = builder_func(self.x_ph, 
                    HITS_REDUCED_SHAPE, self.training_ind_ph)
            if self.out_logits.shape[1] != 2 * len(self.pt_bins) + 1:
                print("Network output logits returned from `create_nn` has\
wrong dimension!")
                print("out_logits.shape: ", self.out_logits.shape)
                print("Expected number of classes: ", 2 * len(self.pt_bins) + 1)
                exit(1)

        self._build_trainer(device)


    def _build_trainer(self, device):
        """
        Create trainer and metrics part.
        """
        # training summaries
        t_summaries = []
        # validation/test summaries
        v_summaries = [] 

        with tf.device(device):
            # acc/loss - per batch 
            pred = tf.argmax(self.out_logits, axis=1, output_type=tf.int32)
            acc = tf.reduce_mean(tf.cast(tf.equal(pred, self.y_ph), tf.float32))

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.y_ph, logits=self.out_logits) 
            loss = tf.reduce_mean(loss)

            # VALIDATION OPS - cumulative across dataset
            # We can take mean of means over batches beacuse 
            # all batches has same size.
            acc_cum, acc_cum_update = tf.metrics.mean(values=acc,
                    name="valid/metrics/acc")
            loss_cum, loss_cum_update = tf.metrics.mean(values=loss,
                    name="valid/metrics/loss")

        # Get nodes from UPDATE_OPS scope and update them on each train step
        # Required for batch norm working properly (see TF batch norm docs)
        _update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(_update_ops):
            train_step = tf.train.RMSPropOptimizer(
                learning_rate=self.model_hparams.lrate, 
                momentum=0.9).minimize(loss)

        # Add acc/loss summaries to setup TB training monitor
        t_summaries += [tf.summary.scalar("train/acc", acc)]
        t_summaries += [tf.summary.scalar("train/loss", loss)]
        v_summaries += [tf.summary.scalar("valid/loss", loss_cum)]
        v_summaries += [tf.summary.scalar("valid/acc", acc_cum)]

        # Collect all metrics variables
        metrics_vars = []
        for el in tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,
                 scope="valid/metrics"):
            metrics_vars.append(el)
        metrics_init = tf.variables_initializer(var_list=metrics_vars)

        update = [acc_cum_update, loss_cum_update]
        t_summaries = tf.summary.merge(t_summaries)
        v_summaries = tf.summary.merge(v_summaries)

        self.ops = dict_to_object({
            'step': train_step,
            'loss': loss,
            'loss_cum': loss_cum,
            'acc': acc,
            'acc_cum': acc_cum,
            'metrics_init': metrics_init,
            'metrics_update': update,
            't_summaries': t_summaries,
            'v_summaries': v_summaries
        })

