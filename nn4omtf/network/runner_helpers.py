# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License
    
    OMTFRunner static helper methods.
"""

import tensorflow as tf


def setup_trainer(net_pt_out, net_sgn_out, labels_pt, 
        labels_sgn, learning_rate=1e-3):
    """Setup model trainer.
    @NOTE 23.03.18: added required sgn trainer
    Args:
        net_pt_out: pt classes (logits) model output -> [p0,..,pk,..,pN]
        net_sng_out: charge sign model output, 1-D vector -> [-, +]
        labels_pt: pt labels to compare with
        labels_sgn: charge sign labels
        learning_rate: just learning rate
    Returns:
        Train operation node
    """
    with tf.name_scope('trainer'):
        with tf.name_scope('loss'):
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

        net_sng_class_out = tf.argmax(net_sgn_out, 1)
        sgn_correct_prediction = tf.equal(
                net_sgn_class_out, 
                tf.argmax(sgn_labels, 1))

        pt_correct_prediction = tf.cast(pt_correct_prediction, tf.float32)
        sgn_correct_prediction = tf.cast(sgn_correct_prediction, tf.float32)

        pt_accuracy = tf.reduce_mean(pt_correct_prediction)
        sgn_accuracy = tf.reduce_mean(sgn_correct_prediction)
        tf.summary.scalar('pt_accuracy', pt_accuracy)
        tf.summary.scalar('sgn_accuracy', sgn_accuracy)
    return pt_accuracy, sgn_accuracy, net_pt_class_out, net_pt_class_out


def collect_statistics(
        sess,           # Session
        sess_name,
        pipe,           # Feed input pipe
        net_in,         # Network input
        net_pt_out,     # Network pt output (logits)
        net_sgn_out,    # Network sgn output (logits) 
        net_pt_class,   # Network pt output argmax (int)
        net_sgn_class,  # Network sgn output argmax (int)
        pt_acc,         # pt accyracy node
        sgn_acc,        # sign accuracy node
        pt_labels,      # labels placeholder
        sgn_labels,
        summary_op):
    """Statistics collector
    Run network and put data into OMTFStatistics object.
    Args:
        sess: TF session
        sess_name: session name
        pipe: Feed input pipe
        net_in: Network input
        net_pt_out: Network pt output (logits)
        net_sgn_out: Network sgn output (logits) 
        net_pt_class: Network pt output argmax (int)
        net_sgn_class: Network sgn output argmax (int)
        pt_acc: pt accyracy node
        sgn_acc: sign accuracy node
        pt_labels: pt labels placeholder
        sgn_labels: charge sign labels placeholder
        summary_op: summary tensor

    Returns:
        tuple of:
            - list of summaries
            - accuracy (float)
            - OMTFStatistics object
    """
    pipe.initialize(sess)
    summaries = []
    pt_res = .0
    cnt = 0
    stats = OMTFStatistics(sess_name)

    while True:
        # Get extra data and put into statistics
        data_in, data_labels, extra = pipe.fetch()
        if train_in is None:
            vp("Train dataset is empty!")
            break

        # Prepare feed dict
        feed_dict = {
                net_in: data_in,
                pt_labels: data_labels[PIPE_OUT_DATA.TRAIN_PROD_PT],
                sgn_labels: data_labels[PIPE_OUT_DATA.TRAIN_PROD_SGN]
        }
        run_in = [
                summary_op,
                pt_acc,
                sgn_acc,
                net_pt_class,
                net_sgn_class
        ]
        run_out = session.run(run_in, feed_dict=feed_dict)
        print("DEBUG:", run_out)
        summaries.append(run_out[0])
        stats.append(run_out[1:])

        l = data_in.shape[0]
        cnt += l
        pt_res += l * run_out[1]
        sgn_res += l * run_out[2]

    return summaries, pt_res / cnt, sgn_res / cnt, stats
    

