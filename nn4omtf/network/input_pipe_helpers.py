# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    OMTFInputPipe static helper methods
"""
import numpy as np
import time
import multiprocessing
import tensorflow as tf

from nn4omtf.const import NPZ_FIELDS, HITS_TYPE, \
        PIPE_MAPPING_TYPE, PIPE_EXTRA_DATA_NAMES, NN_HOLDERS_NAMES,\
        FILTER_DEFAULT

def _deserialize(
        x, 
        hits_type, 
        out_len, 
        out_class_bins, 
        detect_no_signal=FILTER_DEFAULT,
        remap_data=None):
    """Deserialize hits and convert pt value into categories using
    with one-hot encoding.

    NOTICE: data returned from OMTF algorithm is a bit wired.
    In case of mismatch we get charge=-2 and pt = -999.0.
    If we want to compare OMTF with NN, digitizing of OMTF data is
    performed with additional bins. Resulting value is shifted to match
    NN ranges. There are some other problem with compating bins only
    but this case is the most frequent.
    Comparison of NN and OMTF can be done only by comparing bucket due to
    NN is simple clasifier.

    Args:
        hits_type: hits type const from `HITS_TYPE`
        out_len: output one-hot tensor length
        bucket_fn: float value classifier function
        detect_no_signal: map events with no signal registered to special state
                It's used in `train` and `validation` phase.
        remap_data: tuple of (NULLVAL, SHIFT), put NULLVAL instead of 5400 and
                shift all other values by SHIFT to the right.
    Returns:
        data on single event which is 3-tuple containing:
            - selected hits array (HITS_REDUCED | HITS_FULL), transfored or not
            - dict with labels of: 
                - production pt (1-hot encoded)
                - production charge sign (0 - not known, 1 - negative, 2 - positive)
            - dict with extra data:
                - exact value of pt
                - used pt code
                - omtf recognized pt (1-hot)
                - omtf recognized sign 
    """
    prod_dim = NPZ_FIELDS.PROD_SHAPE
    omtf_dim = NPZ_FIELDS.OMTF_SHAPE

    hits = NPZ_FIELDS.HITS_REDUCED
    hits_dim = HITS_TYPE.REDUCED_SHAPE
    if hits_type == HITS_TYPE.FULL:
        hits = NPZ_FIELDS.HITS_FULL
        hits_dim = HITS_TYPE.FULL_SHAPE

    # Define features to read & deserialize from TFRecords dataset 
    features = {
        hits: tf.FixedLenFeature(hits_dim, tf.float32),
        NPZ_FIELDS.PROD: tf.FixedLenFeature(prod_dim, tf.float32),
        NPZ_FIELDS.OMTF: tf.FixedLenFeature(omtf_dim, tf.float32),
        NPZ_FIELDS.PT_CODE: tf.FixedLenFeature([1], tf.float32)
    }
    examples = tf.parse_single_example(x, features)

    hits_arr = examples[hits]

    prod_arr = examples[NPZ_FIELDS.PROD]
    omtf_arr = examples[NPZ_FIELDS.OMTF]
    pt_code = examples[NPZ_FIELDS.PT_CODE][0]

    # ==== Prepare customized no-param bucketizing function
    # Bucket-functions are same for OMTF and NN.
    # OMTF returns pt=-999 if it can't recognize any pattern in HITS array
    # NN will do the same with PT and SIGN, it will say #0 class when
    # there's no enough data in HITS array.
    pt_bucket_fn = lambda x: np.digitize(x=x, bins=out_class_bins).astype(np.int32)
    sgn_bucket_fn = lambda x: np.digitize(x=x, bins=[-1.5, 0]).astype(np.int32)

    # ======= TRAINING DATA
    mean = tf.py_func(np.mean,
                            [hits_arr],
                            tf.float32,
                            stateful=False,
                            name='HITS_mean')
    # Here we build a part of graph!
    # `no_signal` is a tensor!
    no_signal = mean > detect_no_signal
    no_signal_val = lambda: tf.constant(0, dtype=tf.int32)
    pt_k = lambda: tf.py_func(pt_bucket_fn,
        [prod_arr[NPZ_FIELDS.PROD_IDX_PT]],
        tf.int32,
        stateful=False,
        name='pt_class')
    sgn_k = lambda: tf.py_func(sgn_bucket_fn,
        [prod_arr[NPZ_FIELDS.PROD_IDX_SIGN]],
        tf.int32,
        stateful=False,
        name='sgn_class')

    prod_pt_k = tf.case([(no_signal, no_signal_val)], default=pt_k)
    prod_sgn_k = tf.case([(no_signal, no_signal_val)], default=sgn_k)
    
    prod_pt_label = tf.one_hot(prod_pt_k, out_len)
    prod_sgn_label = tf.one_hot(prod_sgn_k, 3)
    # ======= EXTRA OMTF DATA
    omtf_sgn_k = tf.py_func(sgn_bucket_fn,
                    [omtf_arr[NPZ_FIELDS.OMTF_IDX_SIGN]],
                    tf.int32,
                    stateful=False,
                    name='omtf_sgn_class')
    # Encode omtf pt guessed values 
    omtf_pt_k = tf.py_func(pt_bucket_fn,
                   [omtf_arr[NPZ_FIELDS.OMTF_IDX_PT]],
                   tf.int32,
                   stateful=False,
                   name='omtf_pt_class')

    if remap_data is not None:
        nullval, shift = remap_data
        transform_fn = lambda x: np.where(x == 5400, nullval, x + shift)
        hits_arr = tf.py_func(transform_fn,
                            [hits_arr],
                            tf.float32,
                            stateful=False,
                            name='input_data_transformation')
    
        # ======== EXTRA DATA DICT
    # See `PIPE_EXTRA_DATA_NAMES` for correct order of fields
    vals = [
        pt_code,
        prod_arr[NPZ_FIELDS.PROD_IDX_PT],
        prod_pt_k,
        prod_sgn_k,
        omtf_arr[NPZ_FIELDS.OMTF_IDX_PT],
        omtf_pt_k,
        omtf_sgn_k,
        tf.cast(no_signal, tf.int8)
    ]
    edata_dict = dict([(k, v) for k, v in zip(PIPE_EXTRA_DATA_NAMES, vals)])
    data_list = [
        hits_arr,
        prod_sgn_label,
        prod_pt_label
    ]
    data_dict = dict([(k, v) for k, v in zip(NN_HOLDERS_NAMES, data_list)])
    return data_dict, edata_dict


def _new_tfrecord_dataset(
        filename, 
        compression, 
        parallel_calls, 
        in_type,
        out_len, 
        out_class_bins,
        detect_no_signal=FILTER_DEFAULT,
        remap_data=None):
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

    def map_fn(x): return _deserialize(x,
                                       hits_type=in_type,
                                       out_len=out_len,
                                       out_class_bins=out_class_bins,
                                       remap_data=remap_data,
                                       detect_no_signal=detect_no_signal)

    # Additional options to pass in dataset.map
    # - num_threads
    # - output_buffer_size
    return dataset.map(map_fn, num_parallel_calls=parallel_calls)


def setup_input_pipe(
        files_n, 
        name, 
        in_type, 
        out_class_bins, 
        compression_type,   
        batch_size=None, 
        shuffle=False, 
        reps=1,             
        mapping_type=PIPE_MAPPING_TYPE.INTERLEAVE,
        detect_no_signal=FILTER_DEFAULT,
        remap_data=None):
    """Create new input pipeline for given dataset.

    Args:
        files_n: # of files in dataset
        name: input pipe name, used in graph scope
        in_type: input tensor type, see: HITS_TYPE
        out_class_bins: list of bins edges to convert float into one-hot vector
        batch_size(optional,default=None): how many records should be read
            in one batch
        shuffle(optional,default=False): suffle examples in batch
        mapping_type(optional, default=interleave): selects which mapping
            type should be applied when producing examples dataset from
            filenames dataset
    Returns:
        2-tuple (filenames placeholder, dataset iterator)
    """
    # Number of cores for parallel calls
    cores_count = max(multiprocessing.cpu_count() // 2, 1)

    with tf.name_scope(name):
        # File names as placeholder
        files_placeholder = tf.placeholder(tf.string, shape=[files_n])
        # Create dataset of input files
        files_dataset = tf.data.Dataset.from_tensor_slices(
            files_placeholder)
        # Shuffle file paths dataset
        files_dataset = files_dataset.shuffle(buffer_size=files_n)

        # Length of output tensor, numer of classes
        out_len = len(out_class_bins) + 1

        # Prepare customized map function: (filename) => Dataset.map(...)
        def map_fn(filename): return _new_tfrecord_dataset(
            filename,
            compression=compression_type,
            parallel_calls=cores_count,
            in_type=in_type,
            out_len=out_len,
            out_class_bins=out_class_bins,
            detect_no_signal=detect_no_signal,
            remap_data=remap_data
            )

        if mapping_type == PIPE_MAPPING_TYPE.INTERLEAVE:
            # Now create proper dataset with interleaved samples from each TFRecord
            # file. `interleave()` maps provided function which gets filename
            # and reads examples. They can be transformed. Results of map function
            # are interleaved in result dataset.
            #

            ## 15.04.2018 JLysiak - due to changes in TF 1.7
            # I recommend read the docs for more information:
            # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave
            #dataset = files_dataset.interleave(
            #    map_func=map_fn,
            #    cycle_length=files_n,  # Length of cycle - go through all files
            #    block_length=1)       # One example from each input file in one cycle
            
            # Changing to parallel call
            # For more dive into: https://www.tensorflow.org/api_docs/python/tf/contrib/data/parallel_interleave
            dataset = files_dataset.apply(
                    tf.contrib.data.parallel_interleave(
                        map_func=map_fn,
                        cycle_length=files_n,   # Length of cycle - go through all files
                        block_length=1,         # One example from each input file in one cycle
                        sloppy=True)            # Output don't have to be deterministic
                    )

        elif mapping_type == PIPE_MAPPING_TYPE.FLAT_MAP:
            dataset = files_dataset.flat_map(map_func=map_fn)

        else:
            raise ValueError("mapping_type param in not one of PIPE_MAPPING_TYPE constants.")

        # How many times whole dataset will be read.
        dataset = dataset.repeat(count=reps)

        if batch_size is not None:
            # How big one batch will be...
            dataset = dataset.batch(batch_size=batch_size)
            if shuffle:
                # Data will be read into the buffer and here we can suffle
                # it
                dataset = dataset.shuffle(
                    buffer_size=batch_size,
                    seed=int(time.time()))

        # Create initializable iterator. String placeholder will be used in sesson
        # and whole dataset will be created then with proper
        # set of input files, i.e. train, eval, test set of input files
        iterator = dataset.make_initializable_iterator()

    return files_placeholder, iterator
