# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    Neural network based muons' momentum classifier trainer and tester.
"""

import tensorflow as tf
from nn4omtf import OMTFInputPipe
from nn4omtf.utils import dict_to_object
from nn4omtf.const_dataset import DATASET_TYPES

class OMTFRunner:

    def timer_start(self):
        pass


    def train(self, model, no_checkpoints=False, time_limit=None, epochs=1, **opts):
        """
        Run model training.
        Args:
            model: OMTFModel instance

        """
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
            batch_n = 0
            self.timer_start()
            try:
                for epoch_n in range(epochs):
                    try:
                        t_init.run()
                        print("Epoch %d started!" % epoch_n)
                        while True:
                            batch_n += 1
                            xs, ys = sess.run(t_next)
                            feed = {self.x_ph: xs, 
                                self.y_ph: ys, 
                                self.training_ind_ph: True}
                            print(batch_n, batch_n * xs.shape[0], xs.shape)

                            if batch_n % 1000 == 0: #tconf.train_log_ival == 0:
                                b_loss, b_acc, b_summ, _ = sess.run([op_loss, op_acc, op_summ, op_train], feed_dict=feed)
                            else:
                                sess.run(self.ops.step, feed_dict=feed)

                            if batch_n % tconf.validation_ival == 0:
                                v_loss, v_acc, v_summ = self._validate()

                    except tf.errors.OutOfRangeError:
                        print("Epoch %d - finished!" % epoch_n)
                        v_loss, v_acc, v_summ= self._validate()
                        self.model.log_add_epoch_result(epoch_n, v_loss, v_acc)
                        self.model.save()

            except KeyboardInterrupt:
                print("Training stopped by user!")

            self.model.save()

    def _validate(self):
        v_init, v_next = self.pipe_valid.get_initializer_and_op()
        op_update = self.ops['valid_update']
        op_stat = self.ops['valid_stat']
        op_summ = self.ops['valid_summ']
        v_init.run()
        try:
            while True:
                xs, ys = sess.run(v_next)
                feed = {self.x_ph: xs, self.y_ph: ys, self.ind_ph: False}
                sess.run(op_update, feed_dict=feed)

        except tf.errors.OutOfRangeError:
            pass
        stats, summ = sess.run([op_stat, op_summ])


    def test(self):
        """
        Run model test.
        Pass whole TEST dataset through network and save raw logits.
        """
        self._build()

        with tf.name_scope("input_pipes"):
            pipe_test = OMTFInputPipe(ds_path_test, DATASET_TYPE.TEST, 
                pt_bins, batch_size=mconf.test_batch_size)
            t_init, t_next = pipe_test.get_initializer_and_op()
 
        result = None

        with tf.Session() as sess:
            batch_n = 0
            self.timer_start()
            t_init.run()
            try:
                print("Test started!")
                while True:
                    xs, ys = sess.run(t_next)
                    feed = {x_ph: xs, y_ph: ys, ind_ph: False}
                    output = sess.run(out_logits, feed_dict=feed)
                    if result is None:
                        result = output
                    else:
                        result = np.concatenate((result, output), axis=0)

            except tf.errors.OutOfRangeError:
                print("Test finished!")

            except KeyboardInterrupt:
                print("Test stopped by user!")


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
            self._build_trainer()


    def _build_trainer(self):
        """
        Create trainer and metrics part.
        """
        # training summaries
        t_summaries = []
        # validation/test summaries
        v_summaries = [] 

        # acc/loss - per batch 
        pred = tf.argmax(self.out_logits, axis=1, output_type=tf.int32)
        acc = tf.reduce_mean(tf.cast(tf.equal(pred, self.y_ph), tf.float32))

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.y_ph, logits=self.out_logits) 
        loss = tf.reduce_mean(loss)
        # Add acc/loss summaries to setup TB training monitor
        t_summaries += [tf.summary.scalar("train/acc", acc)]
        t_summaries += [tf.summary.scalar("train/loss", loss)]
       
        # VALIDATION OPS - cumulative across dataset
        # We can take mean of means over batches beacuse 
        # all batches has same size.
        acc_cum, acc_cum_update = tf.metrics.mean(values=acc,
                name="valid/metrics/acc")
        loss_cum, loss_cum_update = tf.metrics.mean(values=loss,
                name="valid/metrics/loss")
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

        # Get nodes from UPDATE_OPS scope and update them on each train step
        # Required for batch norm working properly (see TF batch norm docs)
        _update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(_update_ops):
            train_step = tf.train.RMSPropOptimizer(
                learning_rate=self.model_hparams.lrate, 
                momentum=0.9).minimize(loss)

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
        print(self.ops)

