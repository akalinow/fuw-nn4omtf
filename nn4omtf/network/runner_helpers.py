# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License
    
    OMTFRunner static helper methods.
"""

import tensorflow as tf
from nn4omtf.network.statistics import OMTFStatistics
from nn4omtf.network.stats_const import NN_CNAMES
from nn4omtf.network.input_pipe_const import PIPE_EXTRA_DATA_NAMES

def setup_trainer(train_dict, learning_rate=1e-3):
    """Setup model trainer.
    @NOTE 23.03.18: added required sgn trainer
    Args:
        learning_rate: just learning rate
    Returns:
        Train operation node
    """
    with tf.name_scope('trainer'):
        with tf.name_scope('loss'):
            for logs, labs in zip(logits_list, labels_list)
            pt_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=labels_pt, logits=net_pt_out)
            pt_cross_entropy = tf.reduce_mean(pt_cross_entropy)
            
            sgn_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=labels_sgn, logits=net_sgn_out)
            sgn_cross_entropy = tf.reduce_mean(sgn_cross_entropy)
        
        tf.summary.scalar('pt_cross_entropy', pt_cross_entropy)
        tf.summary.scalar('sgn_cross_entropy', sgn_cross_entropy)

        with tf.name_scope('optimizer'):
            pt_train_step = tf.train.AdamOptimizer(
                learning_rate=learning_rate).minimize(pt_cross_entropy)
            sgn_train_step = tf.train.AdamOptimizer(
                learning_rate=learning_rate).minimize(sgn_cross_entropy)
    return pt_train_step, sgn_train_step


def setup_accuracy(net_pt_out, net_sgn_out, pt_labels, sgn_labels):
    """Setup accuracy mesuring subgraph
    @NOTE 23.03.18: added signs labels
    Args:
        net_pt_out: pt labels predicted
        net_sgn_out: charge sign predicted
        pt_labels: pt ground truth labels
        sgn_labels: charge sign class ground truth
    Returns:
        Accuracy nodes
    """
    with tf.name_scope('accuracy'):
        net_pt_class_out = tf.argmax(net_pt_out, 1)
        pt_correct_prediction = tf.equal(
                net_pt_class_out, 
                tf.argmax(pt_labels, 1))

        net_sgn_class_out = tf.argmax(net_sgn_out, 1)
        sgn_correct_prediction = tf.equal(
                net_sgn_class_out, 
                tf.argmax(sgn_labels, 1))

        pt_correct_prediction = tf.cast(pt_correct_prediction, tf.float32)
        sgn_correct_prediction = tf.cast(sgn_correct_prediction, tf.float32)

        pt_accuracy = tf.reduce_mean(pt_correct_prediction)
        sgn_accuracy = tf.reduce_mean(sgn_correct_prediction)
        tf.summary.scalar('pt_accuracy', pt_accuracy)
        tf.summary.scalar('sgn_accuracy', sgn_accuracy)
    return pt_accuracy, sgn_accuracy, net_pt_class_out, net_sgn_class_out


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
    

