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


__all__ = ['OMTFNN']


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
        OUT_NAME = "OUT"
        MODEL_DIR = "model"
        RUNLOG_DIR = "logs"


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
        assert not os.path.exists(obj_file), "OMTFNN object already exists in this location!"
        os.makedirs(self.path)

        # Create graph
        model_dir = os.path.join(self.path, OMTFNN.CONST.MODEL_DIR)
        graph_def, signature_def_map, pt_class, in_type = OMTFNN._create_graph(model_dir, builder_fn=builder_fn)
        
        # Create preview
        subgraph = utils.get_subgraph_by_scope(graph_def, OMTFNN.CONST.GRAPH_SCOPE)
        vis_string = utils.get_visualizer_iframe_string(subgraph)
        utils.save_string_as(vis_string, os.path.join(self.path, OMTFNN.CONST.VIS_FILENAME))
    
        # Save builder code
        builder_code = utils.get_source_of_obj(builder_fn)
        builder_file = os.path.join(self.path, OMTFNN.CONST.NN_BUILD_FUN_FILENAME)
        with open(builder_file, 'w') as f:
            f.write(builder_code)
        
        self.is_session_open = False
        self.graph_def = graph_def
        self.signature_def_map = signature_def_map
        self.pt_class = pt_class
        self.in_type = in_type
        # Save network object
        self.save()



    def create_with_builder_file(name, path, builder_file):
        """Create neural network object using builder python file.
        Args:
            name: model name
            path: destination directory
            builder_file: file contains model builder function
        Returns:
            Network object
        """
        mod = utils.import_module_from_path(path=builder_file)
        fn = utils.get_from_module_by_name(mod=mod, name=OMTFNN.CONST.BUILD_FUN_NAME)
        network = OMTFNN(name=name, path=path, builder_fn=fn)
        return network

    def _update(self, path):
      """Update paths"""
      self.path = path


    def get_log_dir(self):
        return os.path.join(self.path, OMTFNN.CONST.RUNLOG_DIR)


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
        filename = os.path.join(self.path, OMTFNN.CONST.OBJ_FILENAME)
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


    def _check_builder_results(x, y, pt_class, intype):
        """Perform small sanity check on results of user defined builder.
        Args:
            x: input tensor which shape agrees with provided input type
            y: output tensor which shape agrees with provided pt class list length
            pt_class: pt classes bins edges
            intype: input type of OMTFDataset format
        """
        assert isinstance(x, tf.Tensor), "Input is not a tf.Tensor!"
        assert isinstance(y, tf.Tensor), "Output is not a tf.Tensor!"
        assert intype == HITS_TYPE.FULL or intype == HITS_TYPE.REDUCED, "Input type is not one of {} | {}".format(HITS_TYPE.REDUCED, HITS_TYPE.FULL)
        xs_req = HITS_TYPE.FULL_SHAPE if intype == HITS_TYPE.FULL else HITS_TYPE.REDUCED_SHAPE
        xs_req.insert(0, None)
        in_fmt = tf.TensorShape(xs_req)
        out_fmt = tf.TensorShape([None, len(pt_class) + 1])
        x.shape.assert_is_compatible_with(in_fmt)
        y.shape.assert_is_compatible_with(out_fmt)
        

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
                x, y, pt_class, intype = builder_fn()
                OMTFNN._check_builder_results(x, y, pt_class, intype)

                inputs = {OMTFNN.CONST.IN_NAME: x}
                y_out = tf.identity(y, name=OMTFNN.CONST.OUT_NAME)
                outputs = {OMTFNN.CONST.OUT_NAME: y_out}

            signature_inputs = utils.signature_from_dict(inputs)
            signature_outputs = utils.signature_from_dict(outputs)
            signature_def = tf.saved_model.signature_def_utils.build_signature_def(
                    signature_inputs, 
                    signature_outputs, 
                    OMTFNN.CONST.MODEL_SIGNATURE
            )
            signature_def_map = {OMTFNN.CONST.MODEL_SIGNATURE: signature_def}
        # Save graph on disk
        utils.store_graph(graph=g, signature_def_map=signature_def_map, tags=[], model_dir=path)
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
    

    def restore(self, sess):
        """Restore model into TF session.
        Args:
            sess: tensorflow session
        """
        self.is_session_open = True
        self.sess = sess
        path = os.path.join(self.path, OMTFNN.CONST.MODEL_DIR)
        meta = tf.saved_model.loader.load(sess=sess, tags=[], export_dir=path)
        t_in_info = meta.signature_def[OMTFNN.CONST.MODEL_SIGNATURE].inputs[OMTFNN.CONST.IN_NAME]
        t_out_info = meta.signature_def[OMTFNN.CONST.MODEL_SIGNATURE].outputs[OMTFNN.CONST.OUT_NAME]
        t_in = tf.saved_model.utils.get_tensor_from_tensor_info(t_in_info, sess.graph)
        t_out = tf.saved_model.utils.get_tensor_from_tensor_info(t_out_info, sess.graph)
        return meta, t_in, t_out
        

    def finish(self):
        """Finish model usage and save variables and object"""
        saver = tf.train.Saver()
        var_dir = tf.saved_model.constants.VARIABLES_DIRECTORY
        var_file = tf.saved_model.constants.VARIABLES_FILENAME
        save_path = os.path.join(self.path, OMTFNN.CONST.MODEL_DIR, var_dir, var_file)
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


    def append_log(self):
        """Add log from model run."""
        pass


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
        r += "Last accuracy: x\n"
        r += "Training samples: x\n"
        return r


    def __str__(self):
        return self.get_summary()


