# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    Neural network based muon momentum classifier trainer and tester.
"""
import numpy as np
import tensorflow as tf
import time
import os
import threading
import subprocess

from nn4omtf.dataset import OMTFDataset
from nn4omtf.network import OMTFNN
from nn4omtf.utils import init_uninitialized_variables, dict_to_object

from nn4omtf.network.runner_helpers import setup_metrics, setup_trainer
from nn4omtf.network.input_pipe import OMTFInputPipe
from nn4omtf.const import PIPE_EXTRA_DATA_NAMES, NN_HOLDERS_NAMES,\
        PHASE_NAME, CNAMES, PT_CODE_RANGE, PLT_DATA_TYPE


class OMTFRunner:
    """OMTFRunner  is base class for neural nets training and testing.
    It's reading data from TFRecord OMTF dataset created upon
    OMTF simulation data. Input pipe is automaticaly established.
    Main goal is to prepare 'placeholder' and provide universal
    input and output interface for different net's architectures.

    Dataset can be created using OMTFDataset class provided in this package.
    """

    DEFAULT_PARAMS = {
        "detect_no_signal": False,
        "valid_batch_size": 1000,
        "batch_size": 1000,
        "sess_prefix": "",
        "shuffle": False,
        "acc_ival": 1000,
        "epochs": 1,
        "steps": -1,
        "logdir": '.',
        "verbose": False,
        "learning_rate": 0.001,
        "shiftval": 600,
        "nullval": 0,
        "limit_valid_examples": None,
        "limit_test_examples": None,
        "debug": False,
        "log": 'none'
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

    def _log_init(self):
        # Here we'll store validation data
        # Do not left your PC for months
        self.results = None           
        
        self.log_hnd = {
                PHASE_NAME.VALID: None,
                PHASE_NAME.TEST: None
        }
        self.writers = self.log_hnd.copy()

        if self.params['log'] is 'none':
            return

        fname = os.path.join(self.params['logdir'], self.params['sess_name'])
        os.makedirs(fname)
        self.lognpz = os.path.join(fname, 'valid-logs.npz')

        if self.params['log'] in ['txt', 'both']:
            if self.params['phase'] == PHASE_NAME.TRAIN:
                vname = os.path.join(fname,'valid.txt')
                vf = open(vname, 'w')
                self.log_hnd[PHASE_NAME.VALID] = vf
                vf.write(self._params_string())
                
            else:
                tname = os.path.join(fname, 'test.txt')
                self.log_hnd[PHASE_NAME.TEST] = open(tname, 'w')

        if self.params['log'] in ['tb', 'both']:
            if self.params['phase'] == PHASE_NAME.TRAIN:
                vname = os.path.join(fname, PHASE_NAME.VALID)
                self.writers[PHASE_NAME.VALID] = tf.summary.FileWriter(vname)
            else:
                tname = os.path.join(fname, PHASE_NAME.TEST)
                self.writers[PHASE_NAME.TEST] = tf.summary.FileWriter(tname)

    
    def log(self, phase, step, data):
        f = self.log_hnd[phase]
        if f is None:
            return
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        f.write(timestamp + " @ " +str(step) + ": ")
        for n, d in data:
            f.write(n + "=" + str(d) + ", ")
        f.write('\n')
        f.flush()


    def log_summary(self, phase, step, summs):
        writer = self.writers[phase]
        if writer is None:
            return
        for summ in summs:
            writer.add_summary(summ, step)

    def log_to_npz(self, names, data):
        if self.params['log'] is 'none':
            return
        accs = np.array([data])
        if self.results is None:
            self.results = accs
        else:
            self.results = np.append(self.results, accs, axis=0)
        np.savez(self.lognpz, names=names, data=self.results)

    def _log_deinit(self):
        for k, v in self.log_hnd.items():
            if v is not None:
                v.close()


    def _get_verbose_printer(self, lvl=1):
        """Get verbose printer.
        Returns:
            Lambda which prints out its argument if verbose flag is set
        """
        vp = lambda s: print("OMTFRunner: " + s) if self.params['verbose'] >= lvl else None
        return vp


    def _update_params(self, params_dict):
        """Update runner parameters
        Args:
            params_dict: dict with new parameters
        Note that some values (like session name) won't be updated.
        """
        for k, v in params_dict.items():
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
        print(self._params_string())

    def _params_string(self):
        r = "==== OMTFRunner configuration\n"
        for k, v in self.params.items():
            r += "> {:.<20}:{}\n".format(k, v)
        r += "=============================\n"
        return r


    def train(self, **kw):
        """Run model training.
        Training logs are saved on disk if `logs` flag is set.
        Short summaries are always appended to OMTFNN object.
        
        Args:
            **kw: additional args which can update previously set params
        """
        kw['phase'] = PHASE_NAME.TRAIN
        self._update_params(kw)
        self._log_init()
        opt = dict_to_object(self.params)
        vp = self._get_verbose_printer(lvl=1)
        vvp = self._get_verbose_printer(lvl=2)
        vvvp = self._get_verbose_printer(lvl=3)

        tf.reset_default_graph()
        vp("Preparing training session: %s" % opt.sess_name)
        vp("Creating input pipes...")
        with tf.name_scope("input_pipes"):
            # Input pipes configuration
            train_pipe = OMTFInputPipe(
                    dataset=self.dataset,
                    name=PHASE_NAME.TRAIN,
                    hits_type=opt.in_type,
                    out_class_bins=opt.out_class_bins,
                    batch_size=opt.batch_size,
                    shuffle=opt.shuffle,
                    reps=opt.epochs,
                    remap_data=(opt.nullval, opt.shiftval),
                    detect_no_signal=True
                    )
            valid_pipe = OMTFInputPipe(
                    dataset=self.dataset,
                    name=PHASE_NAME.VALID,
                    hits_type=opt.in_type,
                    batch_size=opt.valid_batch_size,
                    out_class_bins=opt.out_class_bins,
                    remap_data=(opt.nullval, opt.shiftval),
                    detect_no_signal=True,
                    limit_examples=opt.limit_valid_examples)
        self.show_params()

        with tf.Session() as sess:
            # Restore model and get I/O tensors
            _, tsd = self.network.restore(sess=sess, sess_name=opt.sess_name)

            # Get collection of initialized variables
            init = sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            vp("Loaded model: %s" % self.network.name)

            # ==== NETWORK PLACEHOLDERS
            pt_labels = tf.placeholder(tf.int8, shape=[None, opt.out_len],
                                    name="pt_labels")
            sgn_labels = tf.placeholder(tf.int8, shape=[None, 3],
                                    name="sgn_labels")
            # ==== SETUP TRAINER NODES
            logits_list = [
                    ("pt", tsd[OMTFNN.CONST.OUT_PT_NAME], pt_labels),
                    ("sgn", tsd[OMTFNN.CONST.OUT_SGN_NAME], sgn_labels)
            ]
            # Get update operation tensors for batch normalization
            # and make statistics updating during training
            # More info? See this:
            # https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_names, train_ops, train_summ_ops, train_vals = setup_trainer(
                        train_list=logits_list,
                        learning_rate=opt.learning_rate)

            ops, metrics_init = setup_metrics(logits_list=logits_list)
            metrics_out = [out for _, out, _, _, _ in ops if out is not None]
            metrics_ops = [op for _, _, op, _, _ in ops]
            metrics_ups = [up for _, _, _, up, _ in ops]
            metrics_summ = [s for _, _, _, _, s in ops if s is not None]
            names = [x[0] for x in ops]
            cnt_op = metrics_ops[-1]

            holders = [
                tsd[OMTFNN.CONST.IN_HITS_NAME],
                sgn_labels,
                pt_labels
            ]
            hdict = dict([(k, v) for k, v in zip(NN_HOLDERS_NAMES, holders)])
            # summary_op = tf.summary.merge_all()

            # At after all, initialize new nodes
            init_uninitialized_variables(sess, initialized=init)
            train_pipe.initialize(sess)
            
            i = 1            
            stamp = self._start_clock()
            vp("{start_datetime} - training started".format(**stamp))

            try:
                while i <= opt.steps or opt.steps < 0:
                    # ======= TRAINING SECTION
                    # Fetch next batch of input data and its labels
                    # Ignore extra data during trainings 
                    ddict, edict = train_pipe.fetch()
                    if ddict is None:
                        vp("Train dataset is empty!")
                        break
                    # Prepare training feed dict
                    train_feed_dict = dict([(hdict[k], ddict[k]) for k in NN_HOLDERS_NAMES])
                    train_feed_dict[tsd[OMTFNN.CONST.IN_PHASE_NAME]] = True
                    if opt.debug:
                        _, train_summ, train_ent, train_outs = sess.run(
                                [train_ops, train_summ_ops, train_vals, metrics_out], 
                                feed_dict=train_feed_dict)
                        edict['PT_K_OUT'] = train_outs[0]
                        edict['SGN_K_OUT'] = train_outs[1]
                        if self.show_dbg(ddict, edict):
                            vp("Exiting...")
                            break
                    else:
                        # Do mini-batch iteration
                        _, train_summ= sess.run(
                                [train_ops, train_summ_ops], 
                                feed_dict=train_feed_dict)
                    # ======= VALIDATION SECTION
                    if i % opt.acc_ival == 0:
                        vp("Validation @ step {step}".format(step=i))
                        sess.run(metrics_init)
                        valid_pipe.initialize(sess)
                        ex_cnt = 0
                        while True:
                            vvp("Examples processed: %d" % ex_cnt)
                            vdict, edict = valid_pipe.fetch()
                            if vdict is None:
                                break
                            # Prepare training feed dict
                            valid_feed_dict = dict([(hdict[k], vdict[k]) for k in NN_HOLDERS_NAMES])
                            valid_feed_dict[tsd[OMTFNN.CONST.IN_PHASE_NAME]] = False
                            sess.run(metrics_ups, feed_dict=valid_feed_dict)
                            ex_cnt = sess.run(cnt_op)
                            if opt.limit_valid_examples is not None:
                                if opt.limit_valid_examples <= ex_cnt:
                                    break
                        accs, summs = sess.run([metrics_ops, metrics_summ])
                        self.log_summary(PHASE_NAME.VALID, i, summs + train_summ)
                        self.log(PHASE_NAME.VALID, i, zip(names, accs))
                        for x, y in zip(names, accs):
                            print(x, y)
                        self.log_to_npz(names, accs)
                    self._next_tick()
                    i += 1

            except KeyboardInterrupt:
                vp("Training stopped by user!")
                
            # End of main while loop, training finished
            stamp = self._next_tick()
            vp("{datetime} - training finished!".format(**stamp))
            vp("Training took: {elapsed:.1f} sec,".format(**stamp))
            vp("Steps in total: %d" % i)

            # Save network state after training
            self.network.finish()
            self.network.save()
            vp("Model saved!")
            self._log_deinit()


    def test(self, **kw):
        """Test loaded model on test dataset.
        Args:
            **kw: additional args which can update previously set params
        """
        kw['phase'] = PHASE_NAME.TEST
        self._update_params(kw)
        opt = dict_to_object(self.params)
        vp = self._get_verbose_printer(lvl=1)
        vvp = self._get_verbose_printer(lvl=2)

        tf.reset_default_graph()
        vp("Preparing test session: %s" % opt.sess_name)
        vp("Creating input pipe...")
        with tf.name_scope("input_pipes"):
            # Input pipes configuration
            test_pipe = OMTFInputPipe(
                    dataset=self.dataset,
                    name=PHASE_NAME.TEST,
                    hits_type=opt.in_type,
                    out_class_bins=opt.out_class_bins,
                    batch_size=opt.valid_batch_size,
                    remap_data=(opt.nullval, opt.shiftval),
                    detect_no_signal=opt.detect_no_signal,
                    limit_examples=opt.limit_test_examples)
        self.show_params()

        with tf.Session() as sess:
            _, tsd = self.network.restore(sess=sess, sess_name=opt.sess_name)
            stamp = self._start_clock()
            vp("{start_datetime} - test started!".format(**stamp))

            # Get collection of initialized variables
            init = sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            vp("Loaded model: %s" % self.network.name)

            # ==== NETWORK PLACEHOLDERS
            pt_labels = tf.placeholder(tf.int8, shape=[None, opt.out_len],
                                    name="pt_labels")
            sgn_labels = tf.placeholder(tf.int8, shape=[None, 3],
                                    name="sgn_labels")
            # ==== SETUP METRICS NODES
            pt_logits = tsd[OMTFNN.CONST.OUT_PT_NAME]
            sgn_logits = tsd[OMTFNN.CONST.OUT_SGN_NAME]
            logits_list = [
                    ("pt", pt_logits, pt_labels),
                    ("sgn", sgn_logits, sgn_labels)
            ]
            ops, metrics_init = setup_metrics(logits_list=logits_list)
            metrics_out = [out for _, out, _, _, _ in ops if out is not None]
            metrics_ops = [op for _, _, op, _, _ in ops]
            metrics_ups = [up for _, _, _, up, _ in ops]
            metrics_summ = [s for _, _, _, _, s in ops if s is not None]
            names = [x[0] for x in ops]
            cnt_op = metrics_ops[-1]
            holders = [
                tsd[OMTFNN.CONST.IN_HITS_NAME],
                sgn_labels,
                pt_labels
            ]
            hdict = dict([(k, v) for k, v in zip(NN_HOLDERS_NAMES, holders)])
            
            # ==== PROBABILITY DISTRIBUTION
            prob = [
                    tf.nn.softmax(pt_logits),
                    tf.nn.softmax(sgn_logits)
            ]
            
            pt_dist = np.zeros([PT_CODE_RANGE + 1, len(opt.out_class_bins) + 1])
            sgn_dist = np.zeros([PT_CODE_RANGE + 1, 3, 3])
            i = 1            
            stamp = self._start_clock()
            vp("{start_datetime} - test started".format(**stamp))
           
            try:
                sess.run(metrics_init)
                test_pipe.initialize(sess)
                ex_cnt = 0
                while True:
                    vvp("Examples processed: %d" % ex_cnt)
                    vdict, edict = test_pipe.fetch()
                    if vdict is None:
                        break
                    # Prepare training feed dict
                    feed_dict = dict([(hdict[k], vdict[k]) for k in NN_HOLDERS_NAMES])
                    feed_dict[tsd[OMTFNN.CONST.IN_PHASE_NAME]] = False
                    
                    prob_out, vout, _ = sess.run([prob, metrics_out, metrics_ups], feed_dict=feed_dict)
                    pt_codes = edict[PIPE_EXTRA_DATA_NAMES[0]]
                    if opt.debug:
                        edict['SGN_K_OUT'] = vout[1]
                        edict['PT_K_OUT'] = vout[0]
                        if self.show_dbg(vdict, edict):
                            vp("Exiting...")
                            break
                    
                    for ptc, dist, s_dist, sgn_k in zip(pt_codes, prob_out[0], prob_out[1], edict['PT_SGN_CLASS']):
                        ptc = ptc.astype(np.int32)
                        pt_dist[ptc] += dist
                        sgn_dist[ptc][sgn_k] += s_dist

                    ex_cnt = sess.run(cnt_op)
                    if opt.limit_test_examples is not None:
                        if opt.limit_test_examples <= ex_cnt:
                            break
                accs, summs = sess.run([metrics_ops, metrics_summ])
                for x, y in zip(names, accs):
                    print(x, y)

                for k in range(PT_CODE_RANGE):
                    s = np.sum(pt_dist[k])
                    if s > 0:
                        pt_dist[k] = pt_dist[k] / s
                    for i in range(3):
                        s = np.sum(sgn_dist[k][i])
                        if s > 0:
                            sgn_dist[k][i] = sgn_dist[k][i] / s
                self.save_dist(pt_dist, sgn_dist)
            except KeyboardInterrupt:
                vp("Training stopped by user!")
            stamp = self._next_tick()
            vp("{datetime} - test finished!".format(**stamp))
            vp("Test run took: {last:.1f} sec.".format(**stamp))

    def save_dist(self, ptd, sgnd):
        if not self.params['prob_dist']:
            return
        pout = os.path.join(self.params['logdir'], 'prob_dist')
        if not os.path.exists(pout):
            os.makedirs(pout)
        npzout = os.path.join(pout, 'prob_dist.npz')
        pttxtout = os.path.join(pout, 'pt.txt')
        np.savez(npzout,
                datatype=PLT_DATA_TYPE.PROB_DIST,
                bins=self.params['out_class_bins'],
                pt_dist=ptd, 
                sgn_dist=sgnd
        )
        np.savetxt(pttxtout, ptd)
        for k in range(PT_CODE_RANGE):
            sgntxtout = os.path.join(pout, 'sgn_%#02d.txt' % k)
            np.savetxt(sgntxtout, sgnd[k])

        print("Probability distributions saved in: " + pout)


    def test_many_models(dataset, models_list, **kw):
        """Test all provided models on test dataset.
        Args:
            dataset: dataset to test on
            models_list: list of OMTFNN objects
            **kw: runner params
        """
        for model in models_list:
            runner = OMTFRunner(dataset, model, **kw)
            runner.test()

    def show_dbg(self, ddict, edict):
        """Show step-by-step debugger screen"""
        print("=================== DEBUGER")
        dlen = len(ddict['HITS'])
        for i in range(dlen):
            print("======= HITS")
            print(ddict['HITS'][i])
            print("======= LABELS")
            print("== PT")
            print(ddict['PT_LABEL'][i])
            print("== SIGN")
            print(ddict['SIGN_LABEL'][i])
            cols = list(edict.keys())
            cols.pop() # remove vectors
            cols.pop()
            hfmt = ["{%s:^15}" % e for e in cols]
            hfmt = " | ".join(hfmt) + "\n"
            dfmt = ["{%s:>15.3}" % e for e in cols]
            dfmt = " | ".join(dfmt) + "\n"
            s = hfmt.format(**dict([(k, k) for k in cols]))
            val_d = dict([(k, float(edict[k][i])) for k in cols])
            s += dfmt.format(**val_d)
            s += "\n"
            print(s)
            v = input("Next? [Y/n]")
            if v == "n" or v == "N":
                break
        v = input("Continue debugging? [Y/n]")
        if v == "n" or v == "N":
            return True
        return False
