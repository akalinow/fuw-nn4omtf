# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License
    
    OMTFRunner static helper methods.
"""

import tensorflow as tf
from nn4omtf.network.statistics import OMTFStatistics
from nn4omtf.const import NN_CNAMES, PIPE_EXTRA_DATA_NAMES

def setup_trainer(train_list, learning_rate=1e-3):
    """Setup model trainer.
    @NOTE 23.03.18: added required sgn trainer
    Args:
        train_list: (name, logits, labels) list
        learning_rate: just learning rate
    Returns:
        - names
        - Train operation nodes list
        - Summaries with cross-entropy
        - values of cross entropy
    """
    names = []
    train_ops = []
    summ_ops = []
    values = []
    with tf.name_scope('trainer'):
        for name, logits, labels in train_list:
            names.append(name)
            with tf.name_scope('loss_' + name):
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=labels, logits=logits)
                cross_entropy = tf.reduce_mean(cross_entropy)
                s = tf.summary.scalar('cross_entropy', cross_entropy)
            with tf.name_scope('optimizer'):
                train_step = tf.train.AdamOptimizer(
                    learning_rate=learning_rate).minimize(cross_entropy)
            train_ops.append(train_step)
            summ_ops.append(s)
            values.append(cross_entropy)
    return names, train_ops, summ_ops, values


def setup_metrics(logits_list):
    """Setup metrics collection ops for network.
    Args:
        logits_list: (name, logits, labels) list
    Returns:
        - List of tuples each input element:
         (name, predicted class, metrics op, metrics update op, summary op)
        - variables initializer
    """
    res = []
    andop = []
    running_vars = []
    with tf.name_scope('accuracy'):
        for name, logits, labels in logits_list:
            with tf.name_scope(name):
                class_out = tf.argmax(logits, 1)
                class_true = tf.argmax(labels, 1)            
                correct = tf.equal(class_out, class_true)
                andop.append(correct)
                metric_op, metric_update_op = tf.metrics.accuracy(
                        class_out, 
                        class_true, 
                        name="metrics")
                for el in tf.get_collection(
                            tf.GraphKeys.LOCAL_VARIABLES, 
                            scope='accuracy/' + name + "/metrics"):
                    running_vars.append(el)
                summ = tf.summary.scalar('acc', metric_op)
                res.append((name, class_out, metric_op, metric_update_op, summ))

        with tf.name_scope('all'):
            comm = andop[0]
            for t in andop[1:]:
                comm = tf.logical_and(comm, t)
            comm_f = tf.cast(comm, tf.float32)
            metric_op, metric_update_op = tf.metrics.mean(comm_f, name="metrics")
            summ = tf.summary.scalar('acc', metric_op)
            res.append(('all', None, metric_op, metric_update_op, summ))
            for el in tf.get_collection(
                        tf.GraphKeys.LOCAL_VARIABLES, 
                        scope="accuracy/all/metrics"):
                    running_vars.append(el)
    with tf.name_scope('cnt'):
        cnt_op, cnt_update_op = tf.contrib.metrics.count(comm_f, name="metrics")
        res.append(('count', None, cnt_op, cnt_update_op, None))
        for el in tf.get_collection(
                    tf.GraphKeys.LOCAL_VARIABLES, 
                    scope="cnt/metrics"):
            running_vars.append(el)
    initializer = tf.variables_initializer(var_list=running_vars)
    return res, initializer

def collect_statistics(
        sess,           # Session
        sess_name,
        pipe,           # Feed input pipe
        net_pholders,   # Network placeholders
        net_pt_out,     # Network pt output (logits)
        net_sgn_out,    # Network sgn output (logits) 
        net_pt_class,   # Network pt output argmax (int)
        net_sgn_class,  # Network sgn output argmax (int)
        pt_acc,         # pt accyracy node
        sgn_acc,        # sign accuracy node
        summary_op):
    """Statistics collector
    Run network and put data into OMTFStatistics object.
    Args:
        sess: TF session
        sess_name: session name
        pipe: Feed input pipe
        net_pholders: Network placeholders
        net_pt_out: Network pt output (logits)
        net_sgn_out: Network sgn output (logits) 
        net_pt_class: Network pt output argmax (int)
        net_sgn_class: Network sgn output argmax (int)
        pt_acc: pt accyracy node
        sgn_acc: sign accuracy node
        summary_op: summary tensor

    Returns:
        tuple of:
            - list of summaries
            - accuracy dict
            - OMTFStatistics object
    """
    pipe.initialize(sess)
    summaries = []
    pt_res = .0
    sgn_res = .0
    cnt = 0
    stats = OMTFStatistics(sess_name)

    while True:
        # Get also extra data dict and put into statistics
        datalist = pipe.fetch()
        if datalist[0] is None:
            break

        # Prepare feed dict
        feed_dict = dict([(k, v) for k, v in zip(net_pholders, datalist[:-1])])

        # Get basic set of data
        input_basic = [
            summary_op,
            pt_acc,
            sgn_acc
        ]
        
        # Request extra data
        # Order is important, see `NN_CNAMES` in stats_const.py
        input_extra = [
            net_pt_class,
            net_sgn_class,
            net_pt_out,
            net_sgn_out
        ]
        run_input = input_basic + input_extra
        run_output = sess.run(run_input, feed_dict=feed_dict)

        # Accumulate overall results from all batches
        l = datalist[0].shape[0]
        cnt += l
        pt_res += l * run_output[1]
        sgn_res += l * run_output[2]
        
        summaries.append(run_output[0])

        # Prapare extra result dict, merge (k, v) pairs
        kvs = [(k, v) for k, v in zip(NN_CNAMES, run_output[3:])] \
                + list(datalist[-1].items())
        stats.append(cols_dict=dict(kvs))
        
    acc = {
        "pt": pt_res / cnt,
        "sgn": sgn_res / cnt
    }
    return summaries, acc, stats
    

