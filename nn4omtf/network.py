# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    Neural network based muon's momentum classifier trainer and tester.
"""

class OMTFRunner:
    def __init__(self, netobject):
        """
        Args:
            netobject: omtf network object
        """
        pass

    def train(self, **kw):
        """
        Run model training.
        """

        tf.reset_default_graph()

        # <- load train dataset
        # <- load valid dataset 

        with tf.name_scope("input_pipes"):
            vset_next, vset_init = pipe_build(PIPE_TYPE.VALID)
            tset_next, tset_init = pipe_build(PIPE_TYPE.TRAIN)
            
        with tf.Session() as sess:
            # network.build_model
            # setup trainer

            try:
                while True:
                    sess.run(...)

                    if valid interval:
                        sess.run(metrics_init)
                        valid_pipe.initialize(sess)

                        while True:
                        # run validation in single batch
                        sess.run()

                    batch_n += 1

            except KeyboardInterrupt:
                vp("Training stopped by user!")

            # save model
