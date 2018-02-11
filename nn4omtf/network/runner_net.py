# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    Neural network based muon momentum classifier trainer and tester.
"""
import tensorflow as tf

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
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=net_out)
        cross_entropy = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('cross_entropy', cross_entropy)
        with tf.name_scope('adam_optimizer'):
            train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
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
        correct_prediction = tf.equal(tf.argmax(net_out, 1), tf.argmax(labels,1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    tf.summary.scalar('accuracy', accuracy)
    return accuracy

