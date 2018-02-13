# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

"""

import tensorflow as tf


__all__ = ['check_accuracy', 'setup_accuracy', 'setup_trainer']


def setup_trainer(net_out, labels, learning_rate=1e-3):
    """Setup model trainer.
    Args:
        net_out: model output
        labels: labels to compare with
    Returns:
        Train operation node
    """
    with tf.name_scope('trainer'):
        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=labels, logits=net_out)
        cross_entropy = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('cross_entropy', cross_entropy)
        with tf.name_scope('adam_optimizer'):
            train_step = tf.train.AdamOptimizer(
                learning_rate=learning_rate).minimize(cross_entropy)
    return train_step


def setup_accuracy(net_out, labels):
    """Setup accuracy mesuring subgraph
    Args:
        net_out: output from model
        labels: labels to compare with
    Returns:
        Accuracy measurement node
    """
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(
            tf.argmax(
                net_out, 1), tf.argmax(
                labels, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        tf.summary.scalar('accuracy', accuracy)
    return accuracy


def check_accuracy(session, pipe, net_in, labels, summary_op, accuracy_op):
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
        data_in, data_labels = pipe.fetch()
        if data_in is None:
            break
        print(data_in.shape)
        l = data_in.shape[0]
        cnt += l
        feed_dict = {net_in: data_in, labels: data_labels}
        summary, acc = session.run([summary_op, accuracy_op],
            feed_dict=feed_dict)
        summaries.append(summary)
        res += l * acc

    return summaries, res / cnt

