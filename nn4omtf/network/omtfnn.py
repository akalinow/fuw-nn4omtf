# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    Neural network wrapper.
"""

from __future__ import print_function

import tensorflow as tf

import os
from io import BytesIO
import numpy as np
from functools import partial
from tensorflow.core.protobuf import saved_model_pb2

import nn4omtf.utils as utils
from nn4omtf.dataset.const import HITS_TYPE


__all__ = ['OMTFNN']


class OMTFNN:
    """OMTF neural network wrapper.
    """
    name = None
    desc = None
    model_path = None
    graph = None
    graph_def = None
    signature_def_map = None

    class CONST:
        OBJ_FILENAME = '.omtfnn.txt'
        VIS_FILENAME = 'preview.html'
        NN_BUILD_FUN_FILENAME = 'builder.py'
        BUILD_FUN_NAME = "create_nn"
        GRAPH_SCOPE = 'omtfnn'
        MODEL_SIGNATURE = tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME
        IN_NAME = "IN"
        OUT_NAME = "OUT"

    def __init__(self, name, path):
        """Create network object."""
        self.name = name
        self.path = path
        self.model_path = os.path.join(self.path, self.name)


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
        network = OMTFNN.create_with_builder_fn(name=name, path=path, create_fn=fn)
        return network


    def create_with_builder_fn(name, path, create_fn):
        """Create neural network object.
        Args:
            name: model name
            path: destination directory
            create_fn: model builder function which returns
              4-touple (in placeholder, out tensor, pt calsses, OMTFDataset input type)
        """
        network = OMTFNN(name, path)
        network._create_graph(builder_fn=create_fn)
        network._store()

    
    def load_from_dir(path):
        """@todo Load trained model."""
        pass


    def _store(self, path):
        """Save model on disk.
        Args:
            path: where object should be stored
        """
        # Check whether model directory already contains such object
        if not tf.saved_model.loader.maybe_saved_model_directory(self.model_path):
            raise Exception("Model '{name}' already exists in '{path}'.".format(name=self.name,
                                                                                path=self.path))
        utils.store_graph(graph=self.graph, signature_def_map=signature_def_map, tags=[], model_dir=self.model_path)
        subgraph = utils.get_subgraph_by_scope(self.graph_def, OMTFNN.CONST.GRAPH_SCOPE)
        vis_string = utils.get_visualizer_iframe_string(subgraph)
        utils.save_string_as(vis_string, os.path.join(self.model_path, OMTFNN.CONST.VIS_FILENAME))
        utils.save_string_as(self.builder_code, os.path.join(self.model_path, OMTFNN.CONST.NN_BUILD_FUN_FILENAME)) 


    def _create_graph(self, builder_fn):
        """Create graph using builder function.
        Args:
            builder_fn: builder function
        """
        self.builder_code = utils.get_source_of_obj(builder_fn)
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
                    OMTFNN.CONST.METHOD_NAME
            )
            signature_def_map = {OMTFNN.CONST.MODEL_SIGNATURE: signature_def}
        self.graph = g
        self.graph_def = g.graph_def
        self.signature_def_map = signature_def_map
        self.pt_class = pt_class
        self.input_type = intype


    def _check_builder_results(x, y, pt_class, intype):
        """Perform small sanity check on results of user defined builder.
        Args:
            x: input tensor which shape agrees with provided input type
            y: output tensor which shape agrees with provided pt class list length
            pt_class: pt classes bins edges
            intype: input type of OMTFDataset format
        """
        if not isinstance(x, tf.Tensor):
            raise ValueError("Input is not a tf.Tensor!")
        if not isinstance(y, tf.Tensor):
            raise ValueError("Output is not a tf.Tensor!")
        if intype != HITS_TYPE.FULL and intype != HITS_TYPE.REDUCED:
            raise ValueError("Input type is not one of {} | {}".format(HITS_TYPE.REDUCED, HITS_TYPE.FULL))

        xs_req = HITS_TYPE.REDUCED_SHAPE
        if intype == HITS_TYPE.FULL:
            xs_req = HITS_TYPE.FULL_SHAPE

        xs = x.shape
        ys = y.shape
        if xs[1] != xs_req[0] and xs[2] != xs_req[1]:
            raise ValueError("Network input shape doesn't fit to [-, %d, %d]" % xs_req)
        req_len = len(pt_class) + 1
        if ys[1] != req_len:
            raise ValueError("Network output shape doesn't fit to [-, %d]" % (req_len)) 
        

    def set_description(self, desc):
        self.desc = desc


    def save(self):
        """@todo Save model."""
        pass


    def restore(self, sess):
        pass


    
    def get_class_count(self):
        """Return number of defined momentum classes for classifier.
        
        It's used in OMTFNNTrainer.
        """
        pass


    def get_class_bins(self):
        """Return momentum class bins' edges.
        
        It's used in OMTFNNTrainer.
        """
        pass


    def is_valid_object(path):
        """Static method to check whether given directory contains valid neural network object.
        Args:
            path: path to object root
        Returns:
            true if valid, false otw.
        """
        path = os.path.join(path, OMTFNN.CONST.OBJ_FILENAME)
        try:
            OMTFNN._parse_obj_file(path)
        except Exception:
            return False
        return True


    def _parse_obj_file(path):
        """Parse network object descriptor.
        Args:
            path: path to file
        Returns:
            dict with params

        02.02.2018: super simple format
        name
        pt bins, space separated
        """
        with open(path, 'r') as f:
            name = f.readline()
            l = f.readline().split(" ")[:-1]
        return {"name": name, "pt": l}


    def _save_obj_file(self, path):
        """Save network object descriptor.
        Args:
            path: path of descriptor file

        Format described in #_parse_obj_file
        """
        with open(path, 'w') as f:
            f.write("{name}\n".format(self.name))
            for e in self.pt_class:
                f.write("{} ".format(e))
            

    def restore_graph(path, sess, tags):
        """Load saved graph"""
        print("Restoring graph from: {}".format(path))
        # Get graph metadata
        meta = tf.saved_model.loader.load(sess=sess, tags=tags, export_dir=path)
        t_in_info = meta.signature_def['train'].inputs['in']
        # print(">>>> Input tensor info:\n", t_in_info)
        t_out_info = meta.signature_def['train'].outputs['out']
        # print(">>>> Output tensor info:\n", t_out_info) 
        t_in = tf.saved_model.utils.get_tensor_from_tensor_info(t_in_info, sess.graph)
        t_out = tf.saved_model.utils.get_tensor_from_tensor_info(t_out_info, sess.graph)
        return meta, t_in, t_out
        

    def dump_trainable_variables(sess, path, name):
        with open(os.path.join(path,name), 'w') as f:
            tensors = sess.graph.get_collection('trainable_variables')
            vals = sess.run(tensors)
            for t,v in zip(tensors, vals):
                f.write(t.name)
                f.write(":")
                f.write(v.__str__())
                f.write("\n")


