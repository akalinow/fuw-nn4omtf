# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    Neural network based muon momentum classifier trainer.
"""

from .dataset import OMTFDataset
import tensorflow as tf

class OMTFNNTrainer:
    """OMTFNNTrainer is base class for neural nets training.
    It's reading data from TFRecord OMTF dataset created upon
    OMTF simulation data. Input pipe is automaticaly established.
    Main goal is to prepare 'placeholder' and provide universal
    input and output interface for different net's architectures.

    OMTF dataset should be created using OMTFDatasetGenerator class
    provided in this package.
    """

    def __init__(self):
        self.dataset = None
        self.network = None


    def set_dataset(self, dataset):
        """Set OMTFDataset for network evaluation and inference."""
        self.dataset = dataset


    def set_omtf_nn(self, omtfnet):
        """Set model."""
        self.network = omtfnet


    def _bucketize(x, classes):
        """Take value x and put it into one class."""
        return np.digitize(x, classes)
        

    def _net_tfrecord_dataset(filename, compression_type, num_parallel_call, hits_dim, class_count, bucket_fn):
        """Method passed to interleave in #setup_input_pipe method as a part of lambda.
        Interleave takes filename tensor from filenames dataset and pass it to this function.
        Now here new TFRecordDataset is produced from single file. It's also place to convert 
        data from binary to tensors.
        """
        # TFRecords can be compressed initially. Pass "ZLIB" or "GZIP" if compressed.
        # This parameter is automaticaly filled by lambda function.
        dataset = tf.data.TFRecordDataset(filenames=filename, compression_type=compression_type)
        
        # Create parametrized map function
        map_fn = lambda x: OMTFNNTrainer._convert_data(x, hits_dim, class_count, bucket_fn)
        
        # Additional options to pass in map
        # - num_threads
        # - output_buffer_size 
        return dataset.map(map_fn, num_parallel_calls=num_parallel_call)


    def _convert_data(x, hits_dim, class_count, bucket_fn):
        """Deserialize hits and convert pt value into categories with one-hot encoding.
        Final converting function can take only one argument, so it must be used in lambda construction.
        """
        features = {
                'hits': tf.FixedLenFeature(hits_dim, tf.float32),
                'prod': tf.FixedLenFeature([4], tf.float32),
                'omtf': tf.FixedLenFeature([6], tf.float32),
        }
        parsed = tf.parse_single_example(x, features)
        prod = parsed['prod']
        l = tf.py_func(bucket_fn, [prod[0]], tf.int64, stateful=False, name='to-one-hot')
        label = tf.one_hot(l, class_count)
        return parsed['hits'], label


    def setup_input_pipe(self, batch_size=100, shuffle=True, buff_size=1000):
        """Create new input pipeline with source file mixing basing on Dataset API."""
    
        files_num = self.dataset.get_files_count()
        # File names as placeholder
        filenames = tf.placeholder(tf.string, shape=[files_num])
        # Create dataset of input files
        file_dataset = tf.data.Dataset.from_tensor_slices(filenames)
        # Shuffle dataset of file paths
        file_dataset = file_dataset.shuffle(files_num)

        # Map function, takes filename, reads serialized Example record and re-produce data
        # Function cannot be just class member because members takes 'self' argument
        # Thus we have to feed this lambda with additional params here
        hits_dim = self.dataset.get_hits_dim()
        class_count = self.network.get_class_count()
        bucket_fn = lambda val: OMTFTrainer._bucketize(val, self.network.get_class_bins()
        par_call = max(1, multiprocessing.cpu_count() // 4)
        map_f = lambda fname: OMTFNNTrainer._net_tfrecord_dataset(fname, self.dataset.compression_type, par_call, hits_dim, class_count, bucket_fn)

        # Now create proper dataset with interleaved samples
        # `interleave` gets filename from `file_dataset` and produces new Dataset (new Queuerunner in the background?)
        dataset = file_dataset.interleave(
                    map_func=map_f, 
                    cycle_length=files_num,         # Length of cycle - go through all files
                    block_length=1                  # One example is taken from each input file in one cycle
                )
        # How many times whole dataset will be read.
        dataset = dataset.repeat(count=1)
        # How big one batch will be... 
        # @Jacek Łysiak: I observed that reading batch of size 1-1000 is almost constant
        dataset = dataset.batch(batch_size=batch_size)
        # Data will be read into the buffer and here we can suffle it
        if shuffle:
            dataset = dataset.shuffle(buffer_size=buff_size, seed=int(time.time()))
        # Create iterator which is initializable. It means that string placeholder will be used in sesson
        # and all dataset will be created then, with propper set of input files, i.e. train, eval, test set of input files
        iterator = dataset.make_initializable_iterator()
        
        self.files_pipe = filenames
        self.dataset_iterator = iterator

    
    def setup_env(self):
        """Prepare all to run."""
        pass


    def run(self, steps, show_acc_every=None):
        """Run model training."""
        pass
