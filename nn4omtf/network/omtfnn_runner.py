# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    Neural network based muon momentum classifier trainer and tester.
"""

import tensorflow as tf
import time

from nn4omtf.dataset import OMTFDataset
from nn4omtf.network import OMTFNN

from nn4omtf.network.runner_pipe import setup_input_pipe
from nn4omtf.network.runner_net import setup_accuracy, setup_trainer

class OMTFRunner:
  """OMTFRunner  is base class for neural nets training and testing.
  It's reading data from TFRecord OMTF dataset created upon
  OMTF simulation data. Input pipe is automaticaly established.
  Main goal is to prepare 'placeholder' and provide universal
  input and output interface for different net's architectures.

  Dataset can be created using OMTFDataset class provided in this package.
  """

  DEFAULT_PARAMS = {
      "batch_size": 1000,
      "sess_prefix": "",
      "shuffle": False,
      "acc_ival": 10,
      "run_meta_ival": None,
      "reps": 1,
      "steps": -1,
      "log_dir": None,
      "verbose": False
      }
  
  def __init__(self, dataset, network):
    """Create runner instance.
    Args:
        dataset: OMTFDataset object, data source
        network: OMTFNN object, working object
    """
    self.dataset = dataset
    self.network = network


  def test(self, **kw):
    print("TEST", self.dataset.name, self.network.name)
  

  def train(self, **kw):
    """Run model training."""
    assert self.dataset is not None, "Run failed! Setup dataset first."
    assert self.network is not None, "Run failed! Setup model first."
    tf.reset_default_graph()


    def get_kw(opt):
      return OMTFRunner.DEFAULT_PARAMS[opt] if opt not in kw else kw[opt]
    
    verbose = get_kw('verbose')
    def vp(s):
      if verbose:
        print("OMTFRunner: " + s)

    # Preload parameters
    check_accuracy_every = get_kw('acc_ival')
    collect_run_metadata_every = get_kw('run_meta_ival')
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    sess_pref = get_kw('sess_prefix')
    sess_name = "{}{}_{}".format(
        sess_pref, 
        self.network.name, 
        timestamp)
    batch_size = get_kw('batch_size')
    shuffle = get_kw('shuffle')
    reps = get_kw('reps')
    steps = get_kw('steps')
    log_dir = get_kw('log_dir')
    in_type = self.network.in_type
    out_class_bins = self.network.pt_class
    out_len = len(out_class_bins) + 1
    
    vp("Preparing training session: %s" % sess_name)

    # Input pipes configuration
    train_filenames, train_files_n = self.dataset.get_dataset(name='train')
    train_placeholder, train_iter = setup_input_pipe(
        files_n=train_files_n,
        name='train',
        compression_type=self.dataset.write_opt,
        in_type=in_type,
        out_class_bins=out_class_bins,
        batch_size=batch_size,
        shuffle=shuffle,
        reps=reps)
      
    # Validation pipe
    valid_filenames, valid_files_n = self.dataset.get_dataset(name='valid')
    valid_placeholder, valid_iter = setup_input_pipe(
        files_n=valid_files_n,
        name='valid',
        compression_type=self.dataset.write_opt,
        in_type=in_type,
        out_class_bins=out_class_bins) 
    
    with tf.Session() as sess:
      # Restore model
      # Do I have any variable to restore ??  
      meta_graph, net_in, net_out = self.network.restore(sess=sess)
      vp("Loaded model: %s" % self.network.name)
      labels = tf.placeholder(tf.int8, shape=[None, out_len])
      
      train_op = setup_trainer(net_out, labels)
      accuracy_op = setup_accuracy(net_out, labels) 
      summary_op = tf.summary.merge_all()
      sess.run(tf.global_variables_initializer())
      
      if log_dir is not None:
        valid_path = os.path.join(log_dir, sess_name, 'valid')
        train_path = os.path.join(log_dir, sess_name, 'train')
        valid_summary_writer = tf.summary.FileWriter(valid_path)
        train_summary_writer = tf.summary.FileWriter(train_path, sess.graph)
      netlog_summary_writer = tf.summary.FileWriter(self.network.get_log_dir())
      
      sess.run(train_iter.initializer, 
        feed_dict={train_placeholder: train_filenames})

      last = time.time()
      start = time.time()
      i = 1

      try:
        vp("Training started at %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
        while i <= steps or steps < 0:  
          # Fetch training examples
          train_in, train_labels = sess.run(train_iter.get_next())
          train_feed_dict = {net_in: train_in, labels: train_labels}
          args = {
              'feed_dict': train_feed_dict,
              'options': None,
              'run_metadata': None
          }
          if collect_run_metadata_every is not None:
            if i % collect_run_metadata_every == 0:
              args['options'] = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
              args['run_metadata'] = tf.RunMetadata()

          summary, _ = sess.run([summary_op, train_op], **args)
          
          if log_dir is not None:
            train_summary_writer.add_summary(summary, i)
            if args['options'] is not None:
              train_summary_writer.add_run_metadata(run_metadata, "step-%d" % i)

          if i % check_accuracy_every == 0: 
            sess.run(
                valid_iter.initializer, 
                feed_dict={valid_placeholder: valid_filenames})
            valid_in, valid_labels = sess.run(valid_iter.get_next())
            valid_feed_dict = {net_in: valid_in, labels: valid_labels}
            summary, acc = sess.run(
                [summary_op, accuracy_op],
                feed_dict=valid_feed_dict)
            if log_dir is not None:
              valid_summary_writer.add_summary(summary, i)
            netlog_summary_writer.add_summary(summary, i)
            vp("Step %d, accuracy: %f" % (i, acc))

          now = time.time()
          elapsed = now - last
          last = now
          vp("Time elapsed: %.3f, last step: %.3f, per example: %.4f" %
              (now - start, elapsed, elapsed ))
          i += 1

      except Exception as e:
        # Should signalize end of dataset...
        print(e)
      
      vp("Training finished at %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
      # Save network state after training
      self.network.finish()
      self.network.save()
      vp("Model saved!")


