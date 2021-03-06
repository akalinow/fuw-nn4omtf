# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    Neural network utilities.
"""

import tensorflow as tf
from IPython.display import clear_output, Image, display, HTML
import numpy as np


def weights(shape, name=None):
    """Generates a weight variable of a given shape.
    Args:
        - shape: variable tensor shape
        - name(optional): variable tensor name
    Returns:
        Variable tensor
    """
    cnt = np.prod(shape)
    stddev = np.square(2 / cnt)
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)


def mk_fc_layer(x, sz, name_suffix="", act_fn=None, is_training=None):
    """Create fully connected layer with batch normalization or not.
    Args:
        x: input tensor
        sz: # of neurons in layer
        name_suffix: suffix appended to layer name
        act_fn: activation function applied to this layer
        is_training: boolean placeholder for inicatining train 
            and test/valid phase
    Returns:
        Output tensor
    """
    eps = 0.0001 # Epsilon for numerical stability
    name = "fc" if name_suffix == "" else "fc_" + name_suffix
    w_shape = [int(x.shape.dims[-1]), sz]
    with tf.name_scope(name):
        w = weights(shape=w_shape, name="w")
        f = tf.matmul(x, w)
        if is_training is not None:
            f = tf.contrib.layers.batch_norm(f, 
                    center=True, 
                    scale=True, 
                    is_training=is_training)
        if act_fn is not None:
            return act_fn(f)
        return f


def add_summary(var, add_stddev=True, add_min=True, add_max=True, add_hist=True):
    """Append a lot of extra summary data to given tensor.
    Appends at least mean value of tensor. Other parameters can be switched off.
    Args:
        - var:  variable tensor
        - stddev(optional,default=True): add mean to summary
        - min(optional,default=True): add min to summary
        - max(optional,default=True): add max to summary
        - hist(optional,default=True): add hist to summary
    Returns:
        List of summary tensor operations
    """
    ss = []
    with tf.name_scope('summary'):
        mean = tf.reduce_mean(var)
        s = tf.summary.scalar('mean', mean)
        ss.append(s)
        if add_stddev:
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            s = tf.summary.scalar('stddev', stddev)
            ss.append(s)
        if add_max:
            s = tf.summary.scalar('max', tf.reduce_max(var))
            ss.append(s)
        if add_min:
            s = tf.summary.scalar('min', tf.reduce_min(var))
            ss.append(s)
        if add_hist:
            s = tf.summary.histogram('histogram', var)
            ss.append(s)
    return ss


def get_subgraph_by_scope(graph_def, name_scope):
    """Get subgraph from given graph by defined name scope.
    Args:
        graph_def: GraphDef object
        name_scope: name scope
    Returns:
        GraphDef with selected nodes.
    """
    graph = tf.GraphDef()
    for n0 in graph_def.node:
        if n0.name.startswith(name_scope):
            n = graph.node.add()
            n.MergeFrom(n0)
    return graph


def get_visualizer_iframe_string(graph_def, width=1024, height=768):
    """Visualize TensorFlow graph.
    Code samples with some edits gathered from deepdream tutorial in this notebook:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb

    Args:
        graph_def: tensorflow graph or graph_def
    Returns:
        iframe HTML string with TensorBoard visualization script
    """
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:{height}px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(graph_def)), id='graph'+str(np.random.rand()), height=height)
  
    iframe = """
        <iframe seamless style="width:{}px;height:{}px;border:0" srcdoc="{}"></iframe>
    """.format(width, height, code.replace('"', '&quot;'))
    return iframe


def jupyter_display(string):
    """Display string in jupyter notebook.
    Method created to display nice picture of graph using #get_visualizer_iframe_string method.
    Args:
        string: string to display, often html
    """
    display(HTML(string))


def save_string_as(string, path):
    """Save string as file.
    Args:
        string: string
        path: path
    """
    with open(path, 'w') as f:
        f.write(string)


def get_saved_model_from_file(path):
    """Get SavedModel from saved protocol buffer.
    Args:
        path: SavedModel path
    Returns:
        SavedModel object
    """
    saved_model = saved_model_pb2.SavedModel()
    saved_model.ParseFromString(open(model_fn, "rb").read())
    return saved_model

    graph_def = saved_model.meta_graphs[0].graph_def


def store_graph(graph, signature_def_map, tags, model_dir):
    """Store graph wrapped with SavedModel in given directory.
    Model is saved along with SignatureDefMap, MetaGraphDef and variables.
    For more info, please, refer to https://www.tensorflow.org/programmers_guide/saved_model
    Args:
        graph: tensorflow graph
        signature_def_map: model signature definitions map
        tags: saved model tags
        model_dir: model directory
    """
    builder = tf.saved_model.builder.SavedModelBuilder(model_dir)
    config = tf.ConfigProto(
            log_device_placement=False,
            allow_soft_placement=True)
    with tf.Session(graph=graph, config=config) as sess:
        tf.global_variables_initializer().run()
        builder.add_meta_graph_and_variables(
                sess=sess, 
                tags=tags,
                signature_def_map=signature_def_map)
        builder.save()


def signature_from_dict(sig_dict):
    signature = {
            key: tf.saved_model.utils.build_tensor_info(tensor) for key, tensor in sig_dict.items()
            }
    return signature


def float_feature(value):
    """Creates TensorFlow feature from value."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def init_uninitialized_variables(sess, initialized):
    """Helper method to initialize uninitialized variables
    Args:
        sess: tf session
        initialized: list of initialized tensors
    """
    all_vars = sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    for var in all_vars:
        if var not in initialized:
            sess.run(var.initializer)

