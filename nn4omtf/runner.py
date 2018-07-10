# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    Neural network based muons' momentum classifier trainer and tester.
"""


class OMTFRunner:

    def timer_start(self):
        pass


    def train(self, model, no_checkpoints=False, time_limit=None, epochs=1, **opts):
        """
        Run model training.
        Args:
            model: OMTFModel instance

        """
        self._build()

        with tf.name_scope("input_pipes"):
            pipe_train = OMTFInputPipe(self.model_config.ds_train, 
                    DATASET_TYPE.TRAIN, self.pt_bins, 
                    batch_size=model_hparams.batch_size)
            self.pipe_valid = OMTFInputPipe(model_config.ds_valid, 
                    DATASET_TYPE.VALID, pt_bins, 
                    batch_size=model_hparams.batch_size)
            t_init, t_next = pipe_train.get_initializer_and_op()
 
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
                            feed = {self.x_ph: xs, y_ph: ys, ind_ph: True}
                            
                            if batch_n % tconf.train_log_ival == 0:
                                b_loss, b_acc, b_summ, _ = sess.run([op_loss, op_acc, op_summ, op_train], feed_dict=feed)
                            else:
                                sess.run(op_train, feed_dict=feed)

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
        model_config = self.model.get_config()
        model_hparams = self.model.get_hparams()

        # HIST array placeholder, dim(x) = 3, batch of 2D HITS arrays
        HITS_REDUCED_SHAPE = [18, 2]
        self.x_ph = tf.placeholder(tf.int32)
        # Labels placeholder - dim(y) = 1, batch of out class indexes
        self.y_ph = tf.placeholder(tf.int32, shape=[None])
        # Training phase indicator - boolean
        self.training_ind_ph = tf.placeholder(tf.bool)


        device = '/cpu:0'
        if model_config.gpu:
            device = '/gpu:0'

        with tf.device(device):
            out_logits, pt_bins = builder_func(self.x_ph, 
                    HITS_REDUCED_SHAPE, self.training_ind_ph)
            self.ops = self.build_trainer(out_logits)

