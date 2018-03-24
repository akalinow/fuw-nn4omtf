# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    Neural network wrapper.
"""

from __future__ import print_function

import tensorflow as tf
import os
import numpy as np
import pickle
import re

import nn4omtf.utils as utils
from nn4omtf.dataset.const import HITS_TYPE



class OMTFNN:
    """OMTF neural network wrapper.
    """

    class CONST:
        OBJ_FILENAME = '.omtfnn'
        VIS_FILENAME = 'preview.html'
        NN_BUILD_FUN_FILENAME = 'builder.py'
        BUILD_FUN_NAME = "create_nn"
        GRAPH_SCOPE = 'omtfnn'
        MODEL_SIGNATURE = tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME
        IN_NAME = "IN"
        OUT_PT_NAME = "PT_OUT"
        OUT_SGN_NAME = "SGN_OUT"
        MODEL_DIR = "model"
        RUNLOG_DIR = "logs"
        STATISTICS_DIR = "statistics"

    def __init__(self, name, path, builder_fn):
        """Create network object.
        Args:
            name: model name
            path: model root directory location
            builder_fn: method which builds neural network inside
        """
        self.name = name
        self.path = os.path.join(path, name)
        obj_file = os.path.join(self.path, OMTFNN.CONST.OBJ_FILENAME)
        assert not os.path.exists(
            obj_file), "OMTFNN object already exists in this location!"
        os.makedirs(self.path)

        # Create graph
        model_dir = os.path.join(self.path, OMTFNN.CONST.MODEL_DIR)
        graph_def, signature_def_map, pt_class, in_type = OMTFNN._create_graph(
            model_dir, builder_fn=builder_fn)

        # Create preview
        subgraph = utils.get_subgraph_by_scope(
            graph_def, OMTFNN.CONST.GRAPH_SCOPE)
        vis_string = utils.get_visualizer_iframe_string(subgraph)
        utils.save_string_as(
            vis_string, os.path.join(
                self.path, OMTFNN.CONST.VIS_FILENAME))

        # Save builder code
        builder_code = utils.get_source_of_obj(builder_fn)
        builder_file = os.path.join(
            self.path, OMTFNN.CONST.NN_BUILD_FUN_FILENAME)
        with open(builder_file, 'w') as f:
            f.write(builder_code)

        self.is_session_open = False
        self.graph_def = graph_def
        self.signature_def_map = signature_def_map
        self.pt_class = pt_class
        self.in_type = in_type

        self.log = []
        # Save network object
        self.save()

    def create_with_builder_file(name, path, builder_file):
        """Create neural network object using builder python file.
        Args:
            name: model name
            path: destination directory
            builder_file: file contains model builder function
        Returns:
            etwork object
        """
        mod = utils.import_module_from_path(path=builder_file)
        fn = utils.get_from_module_by_name(
            mod=mod, name=OMTFNN.CONST.BUILD_FUN_NAME)
        network = OMTFNN(name=name, path=path, builder_fn=fn)
        return network

    def _update(self, path):
        """Update paths"""
        self.path = path


    def load(path):
        """Load saved model object.
        Args:
            path: model path
        """
        filename = os.path.join(path, OMTFNN.CONST.OBJ_FILENAME)
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
        obj._update(path)
        return obj


    def save(self):
        """Save model object."""
        self.sess = None
        self.summary_writer = None
        filename = os.path.join(self.path, OMTFNN.CONST.OBJ_FILENAME)
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


    def add_statistics(self, stats):
        """Add OMTFStatistcs object to model directory tree.
        Args:
            stats: OMTFStatistics instance
        """
        path = os.path.join(self.path, 
                OMTFNN.CONST.STATISTICS_DIR, stats.sess_name)
        os.makedirs(path, exist_ok=True)
        idx = len(os.listdir(path))
        filename = os.path.join(path, str(idx))
        with open(filename, 'wb') as f:
            pickle.dump(stats, f)


    def list_statistics(self):
        """Get list of available statistics
        Returns:
            list of tuples (sess name, # of stats)
        """
        path = os.path.join(self.path, 
                OMTFNN.CONST.STATISTICS_DIR) 
        res = []
        cnt = []
        files = os.listdir(path)
        for f in files:
            l1_dir = os.path.join(path, f)
            subdir = os.listdir(l1_dir)
            for d in subdir:
                p = os.path.join(l1_dir, d)
                res.append(os.path.join(f,d))
                cnt.append(len(os.listdir(p)))
        return zip(res, cnt)

    def match_statistics(self, regexp):
        entries = []
        base = os.path.join(self.path, 
                OMTFNN.CONST.STATISTICS_DIR)
        for stat_base in os.listdir(base):
            stat_full = os.path.join(base, stat_base)
            stat_exs = os.listdir(stat_full)
            for stat_ex in stat_exs:
                entry = os.path.join(stat_base, stat_ex)
                if re.match(regexp, entry) is not None:
                    full = os.path.join(stat_full, stat_ex)
                    cnt = len(os.listdir(full))
                    entries.append((entry, full, cnt))
        return entries


    def get_statistics_objs(self, fpath):
        """Load statistics objects from dir.
        Args:
            fpath: path
        """
        path = os.path.join(self.path, 
                OMTFNN.CONST.STATISTICS_DIR, fpath)
        objs = []
        for fname in os.listdir(path):
            fullpath = os.path.join(path,fname)
            with open(fullpath, 'rb') as f:
                objs.append((int(fname), pickle.load(f)))
        objs.sort(key=lambda x: x[0])
        return objs


    def _check_builder_results(x, y_pt, y_sgn, pt_class, intype):
        """Perform small sanity check on results of user defined builder.
        Args:
            x: input tensor (hits data) which shape agrees with 
                provided input type
            y_pt: output tensor (pt classes logits) which shape agrees 
                with provided pt class list length
            y_sgn: output tensor (charge sign logits) of shape [None, 2] 
            pt_class: pt classes bins edges
            intype: input type of OMTFDataset format
        """
        assert isinstance(x, tf.Tensor), "Input is not a tf.Tensor!"
        assert isinstance(y_pt, tf.Tensor), "pt output is not a tf.Tensor!"
        assert isinstance(y_sgn, tf.Tensor), "sgn output is not a tf.Tensor!"
        assert intype == HITS_TYPE.FULL or intype == HITS_TYPE.REDUCED,     \
                "Input type is not one of {} | {}".format(                  \
                HITS_TYPE.REDUCED, HITS_TYPE.FULL)
        xs_req = HITS_TYPE.FULL_SHAPE if intype == HITS_TYPE.FULL else HITS_TYPE.REDUCED_SHAPE
        xs_req.insert(0, None)
        in_fmt = tf.TensorShape(xs_req)
        out_pt_fmt = tf.TensorShape([None, len(pt_class) + 1])
        out_sgn_fmt = tf.TensorShape([None, 2])
        x.shape.assert_is_compatible_with(in_fmt)
        y_pt.shape.assert_is_compatible_with(out_pt_fmt)
        y_sgn.shape.assert_is_compatible_with(out_sgn_fmt)


    def _create_graph(path, builder_fn):
        """Create graph using builder function and store on disk.
        Args:
            path:
            builder_fn: builder function
        Returns:
            pt class bins, input type const
        """
        with tf.Graph().as_default() as g:
            with tf.name_scope(OMTFNN.CONST.GRAPH_SCOPE):
                x, y_pt, y_sgn, pt_class, intype = builder_fn()
                OMTFNN._check_builder_results(x, y_pt, y_sgn, pt_class, intype)
                inputs = {OMTFNN.CONST.IN_NAME: x}
                y_pt_out = tf.identity(y_pt, name=OMTFNN.CONST.OUT_PT_NAME)
                y_sgn_out = tf.identity(y_sgn, name=OMTFNN.CONST.OUT_SGN_NAME)
                outputs = {
                    OMTFNN.CONST.OUT_PT_NAME: y_pt_out,
                    OMTFNN.CONST.OUT_SGN_NAME: y_sgn_out
                }

            signature_inputs = utils.signature_from_dict(inputs)
            signature_outputs = utils.signature_from_dict(outputs)
            signature_def = tf.saved_model.signature_def_utils.build_signature_def(
                signature_inputs, signature_outputs, OMTFNN.CONST.MODEL_SIGNATURE)
            signature_def_map = {OMTFNN.CONST.MODEL_SIGNATURE: signature_def}
        # Save graph on disk
        utils.store_graph(
            graph=g,
            signature_def_map=signature_def_map,
            tags=[],
            model_dir=path)
        return g.as_graph_def(), signature_def_map, pt_class, intype


    def is_valid_object(path):
        """Static method to check whether given directory contains valid neural network object.
        Args:
            path: path to object root
        Returns:
            true if valid, false otw.
        """
        try:
            OMTFNN.load(path)
        except Exception:
            return False
        return True


    def restore(self, sess, sess_name):
        """Restore model into TF session.
        Args:
            sess: tensorflow session
            sess_name: session name
        Returns:
            tuple of:
                - graph metadata
                - input tensor
                - output tensor of pt logits
                - output tensor of charge sign logits
        """
        self.is_session_open = True
        self.sess = sess
        log_dir = os.path.join(self.path, OMTFNN.CONST.RUNLOG_DIR, sess_name)
        self.summary_writer = tf.summary.FileWriter(log_dir)
        export_path = os.path.join(self.path, OMTFNN.CONST.MODEL_DIR)
        meta = tf.saved_model.loader.load(sess=sess, tags=[], 
                export_dir=export_path)
        t_in_info = meta.signature_def[OMTFNN.CONST.MODEL_SIGNATURE].inputs[OMTFNN.CONST.IN_NAME]
        t_pt_out_info = meta.signature_def[OMTFNN.CONST.MODEL_SIGNATURE].outputs[OMTFNN.CONST.OUT_PT_NAME]
        t_sgn_out_info = meta.signature_def[OMTFNN.CONST.MODEL_SIGNATURE].outputs[OMTFNN.CONST.OUT_SGN_NAME]
        t_in = tf.saved_model.utils.get_tensor_from_tensor_info(
            t_in_info, sess.graph)
        t_pt_out = tf.saved_model.utils.get_tensor_from_tensor_info(
            t_pt_out_info, sess.graph)
        t_sgn_out = tf.saved_model.utils.get_tensor_from_tensor_info(
            t_sgn_out_info, sess.graph)
        return meta, t_in, t_pt_out, t_sgn_out


    def finish(self):
        """Finish model usage and save variables and object"""
        saver = tf.train.Saver()
        var_dir = tf.saved_model.constants.VARIABLES_DIRECTORY
        var_file = tf.saved_model.constants.VARIABLES_FILENAME
        save_path = os.path.join(
            self.path,
            OMTFNN.CONST.MODEL_DIR,
            var_dir,
            var_file)
        saver.save(self.sess, save_path, write_meta_graph=False)
        self.sess = None
        self.is_session_open = False

    def dump_trainable_variables(self):
        """Dump trainable variables from model.
        Returns:
            list of 2-tuples (tensor, value)
        """
        if self.sess is None:
            sess = tf.Session()
        else:
            sess = self.sess
        tensors = sess.graph.get_collection('trainable_variables')
        vals = sess.run(tensors)
        if self.sess is None:
            sess.close()
        return zip(tensors, vals)

    def move(self, dest):
        """Move network."""
        shutil.move(self.path, dest)
        self.path = dest

    def add_summaries(self, summaries, step):
        for s in summaries:
            self.summary_writer.add_summary(s, step)

    def add_log(self, acc, step, sess_name):
        """Add log from model run."""
        self.log.append([acc, step, sess_name])
        

    def name_matches(self, regexp):
        """Check whether network name matches given regexp.
        Args:
            regexp: regexp
        Returns:
            True if matches, False if not...
        """
        return re.match(regexp, self.name) is not None

    def get_summary(self):
        """Get short summary
        Returns:
            printable summary
        """
        r = "Model name: %s\n" % self.name
        r += "{:^40}{:^15}{:^15}{:^15}\n".format("Session name", "Step", "pt acc", "sgn acc")
        for rec in self.log:
            r += "{:<40}{:<15}{:<15.3}{:<15.3}\n".format(rec[2], rec[1], rec[0]['pt'],rec[0]['sgn'])
        r += "\nAvailable statistics list:\n"
        idx = 0
        for e, c in self.list_statistics():
            r += "%3d : %s, files: %d\n" % (idx, e, c)
            idx += 1
        return r

    def __str__(self):
        return self.get_summary()
