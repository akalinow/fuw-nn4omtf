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
        net_pt_out: pt classes (1-hot) model output
        net_sng_out: charge sign model output
        labels_pt: pt labels to compare with
        labels_sgn: charge sign labels
        learning_rate: just learning rate
    Returns:
        Train operation node
    """
    with tf.name_scope('trainer'):
        with tf.name_scope('loss'):
            # As is stated here [https://www.tensorflow.org/api_docs/python
            # /tf/nn/sparse_softmax_cross_entropy_with_logits]
            # when labels doesn't represent 'soft' class (only one single 
            # true class is provided) this sparse cross entropy should be used
            pt_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels_pt, logits=net_pt_out)
            pt_cross_entropy = tf.reduce_mean(pt_cross_entropy)
            
            sgn_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
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
        pt_correct_prediction = tf.equal(
                tf.argmax(net_pt_out, 1), 
                tf.argmax(pt_labels, 1))

        sgn_correct_prediction = tf.equal(
                tf.argmax(net_sgn_out, 1), 
                tf.argmax(sgn_labels, 1))

        pt_correct_prediction = tf.cast(pt_correct_prediction, tf.float32)
        sgn_correct_prediction = tf.cast(sgn_correct_prediction, tf.float32)

        pt_accuracy = tf.reduce_mean(pt_correct_prediction)
        sgn_accuracy = tf.reduce_mean(sgn_correct_prediction)
        tf.summary.scalar('pt_accuracy', pt_accuracy)
        tf.summary.scalar('sgn_accuracy', sgn_accuracy)
    return pt_accuracy, sgn_accuracy


def check_accuracy(session, pipe, net_in, labels, accuracy_op, summary_op=None):
    """Measure accuracy on whole dataset.
    Args:
        session: tf session
        pipe: OMTFInputPipe object
        net_in: model input placeholder
        labels: labels placeholder
        summary_op: summary operation node
        accuracy_op: accuracy operation node
    """
    pipe.initialize(session)
    summaries = []
    res = .0
    cnt = 0
        
    while True:
        data_in, data_labels, _ = pipe.fetch()
        if data_in is None:
            break
        l = data_in.shape[0]
        cnt += l
        feed_dict = {net_in: data_in, labels: data_labels}
        if summary_op is not None:
            summary, acc = session.run([summary_op, accuracy_op],
                feed_dict=feed_dict)
            summaries.append(summary)
        else:
            acc = session.run(accuracy_op, feed_dict=feed_dict)
        res += l * acc

    return summaries, res / cnt

