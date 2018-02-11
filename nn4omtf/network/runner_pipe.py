# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License
"""
import numpy as np
import time
import multiprocessing
import tensorflow as tf
from nn4omtf.dataset.const import NPZ_FIELDS, HITS_TYPE

__all__ = ['setup_input_pipe']


def _deserialize(x, hits_type, out_len, bucket_fn):
  """Deserialize hits and convert pt value into categories using
  with one-hot encoding. 
  Args:
    hits_type: hits type const from `HITS_TYPE`
    out_len: output one-hot tensor length
    bucket_fn: float value classifier function
  """
  prod_dim = NPZ_FIELDS.PROD_SHAPE
  hits = NPZ_FIELDS.HITS_REDUCED
  hits_dim = HITS_TYPE.REDUCED_SHAPE
  if hits_type == HITS_TYPE.FULL:
    hits = NPZ_FIELDS.HITS_FULL
    hits_dim = HITS_TYPE.FULL_SHAPE
  features = {
    hits: tf.FixedLenFeature(hits_dim, tf.float32),
    NPZ_FIELDS.PROD: tf.FixedLenFeature(prod_dim, tf.float32),
  }
  examples = tf.parse_single_example(x, features)
  prod = examples[NPZ_FIELDS.PROD]
  k = tf.py_func(bucket_fn, 
      [prod[NPZ_FIELDS.PROD_IDX_PT]], 
      tf.int64, 
      stateful=False, 
      name='to-one-hot')
  label = tf.one_hot(k, out_len)
  return examples[hits], label


def _new_tfrecord_dataset(filename, compression, parallel_calls, in_type,
    out_len, bucket_fn):
  """Creates new TFRecordsDataset as a result of map function.
  It's interleaved in #setup_input_pipe method as a base of lambda.
  Interleave takes filename tensor from filenames dataset and pass 
  it to this function. New TFRecordDataset is produced  by reading 
  `block_length` examples from given file in cycles of defined 
  earlier length. It's also the place for converting data from 
  binary form to tensors.
  Args:
    filename: TFRecord file name
    compression: type of compression used to save given file
    parallel_calls: # of calls used to map
    in_type: input tensor type
    out_len: output one-hot tensor length
    bucket_fn: float value classifier function
  """
  # TFRecords can be compressed initially. Pass `ZLIB` or `GZIP` 
  # if compressed or `None` otherwise.
  dataset = tf.data.TFRecordDataset(
      filenames=filename, 
      compression_type=compression)
  # Create deserializing map function 
  map_fn = lambda x: _deserialize(x, 
      hits_type=in_type, 
      out_len=out_len, 
      bucket_fn=bucket_fn)

  # Additional options to pass in dataset.map
  # - num_threads
  # - output_buffer_size 
  return dataset.map(map_fn, num_parallel_calls=parallel_calls)


def setup_input_pipe(files_n, name, in_type, out_class_bins, compression_type,
    batch_size=None, shuffle=False, reps=1):
  """Create new input pipeline for given dataset.
  Args:
    files_n: # of files in dataset
    name: input pipe name, used in graph scope
    in_type: input tensor type, see: HITS_TYPE 
    out_class_bins: list of bins edges to convert float into one-hot vector
    batch_size(optional,default=None): how many records should be read
      in one batch
    shuffle(optional,default=False): suffle examples in batch
  Returns:
      2-tuple (filenames placeholder, dataset iterator)
  """
  # Number of cores for parallel calls 
  cores_count = multiprocessing.cpu_count()

  with tf.name_scope("input_pipe"):
    with tf.name_scope(name):
      # File names as placeholder
      files_placeholder = tf.placeholder(tf.string, shape=[files_n])
      # Create dataset of input files
      files_dataset = tf.data.Dataset.from_tensor_slices(files_placeholder)
      # Shuffle file paths dataset
      files_dataset = files_dataset.shuffle(buffer_size=files_n)
      
      # Length of output tensor, numer of classes
      out_len=len(out_class_bins) + 1
      # Prepare customized no-arg bucketize function
      bucket_fn = lambda x: np.digitize(x=x, bins=out_class_bins)
      # Prepare customized map function: (filename) => Dataset.map(...)
      map_fn = lambda filename: _new_tfrecord_dataset(filename, 
          compression=compression_type,
          parallel_calls=cores_count,
          in_type=in_type,
          out_len=out_len,
          bucket_fn=bucket_fn)

      # Now create proper dataset with interleaved samples from each TFRecord
      # file. `interleave()` maps provided function which gets filename 
      # and reads examples. They can be transformed. Results of map function
      # are interleaved in result dataset.
      dataset = files_dataset.interleave(
          map_func=map_fn, 
          cycle_length=files_n, # Length of cycle - go through all files
          block_length=1)       # One example from each input file in one cycle
      
      # How many times whole dataset will be read.
      dataset = dataset.repeat(count=reps)
      
      if batch_size is not None:
        # How big one batch will be... 
        # @jlysiak: I observed that reading batch of size 1-1000 takes almost
        #     constant time
        dataset = dataset.batch(batch_size=batch_size)
        if shuffle:
          # Data will be read into the buffer and here we can suffle it
          dataset = dataset.shuffle(
              buffer_size=batch_size,
              seed=int(time.time()))
    
  # Create initializable iterator. String placeholder will be used in sesson 
  # and whole dataset will be created then with proper 
  # set of input files, i.e. train, eval, test set of input files
  iterator = dataset.make_initializable_iterator()

  return files_placeholder, iterator

