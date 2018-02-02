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
        self.logdir = None


    def set_dataset(self, dataset):
        """Set OMTFDataset for network evaluation and inference."""
        self.dataset = dataset


    def set_omtf_nn(self, omtfnet):
        """Set model."""
        self.network = omtfnet


    def set_log_dir(self, path):
        """Set directory for logging summaries."""
        self.logdir = path


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


    def setup_input_pipe(self, dataset_name, batch_size=100, shuffle=True, buff_size=1000):
        """Create new input pipeline for given dataset.
        
        Args:
            dataset_name: 'valid' | 'test' | 'train'
            batch_size(optional,default=100): how many records should be read in one batch
            shuffle(optional,default=True): suffle examples in batch
            buff_size(optional,default=True): buffer size
        Returns:
            Filenames placeholder, dataset iterator
        """
    
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


    def _setup_trainer(self):
        """Helper method to setup model trainer."""
        with tf.name_scope('trainer'):
            with tf.name_scope('loss'):
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.model_out)
            self.cross_entropy = tf.reduce_mean(cross_entropy)
            tf.summary.scalar('cross_entropy', cross_entropy)

            with tf.name_scope('adam_optimizer'):
                self.train_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cross_entropy)

            with tf.name_scope('accuracy'):
                correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y_,1))
                correct_prediction = tf.cast(correct_prediction, tf.float32)

            accuracy = tf.reduce_mean(correct_prediction)
            tf.summary.scalar('accuracy', accuracy)


    def _setup_train_pipes(self):
        """Helper method to setup input pipes for training."""
        with tf.name_scope("input_pipe"):
            with tf.name_scope("train"):
                self.f_train, self.it_train = self.setup_input_pipe('train')
            with tf.name_scope("valid"):
                self.f_valid, self.it_valid = self.setup_input_pipe('valid', batch_size=10000, shuffle=False, buff_size=10000)


    def _feed_dict(train):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if train:
            batch = sess.run(it_train.get_next())
            k = 0.9
        else:
            sess.run(it_valid.initializer, feed_dict={f_valid: files_map['valid']})
            batch = sess.run(it_valid.get_next())
            k = 1.0
        return {x_in: batch[0].reshape(-1, 36), y_: batch[1]}


    def train(self, steps=0, evaluate_every=100):
        """Run model training.
        Args:
            steps(optional,default=0): how many steps should be run, zero means till end of dataset
            evaluate_every(optional,default=100): setup number of steps after which model is tested on validation set
        """

        if self.dataset is None:
            raise ValueError("Training run failed! Setup dataset first.")
        if self.network is None:
            raise ValueError("Training run failed! Setup model first.")

        self._setup_train_pipes()

        self.labels = tf.placeholder(tf.float32, [None, self.network.get_classes_count()]) 
    
        stamp = time.strftime("%Y-%m-%d-%H-%M-%S")

        with tf.Session() as sess:
            # Now restore model
            #meta_graph, x_in, y_out = self.network.restore(sess=sess) #restore_graph(path=path, sess=sess, tags=[])

            #saver = tf.train.Saver()
            #save_path = os.path.join(path, "variables", "variables")
            
            self.model_in, self.model_out = self.network.restore(sess=sess)
            self._setup_trainer()

            merged = tf.summary.merge_all()
            

            test_writer = tf.summary.FileWriter('./tb_test/{}/test'.format(stamp))
            train_writer = tf.summary.FileWriter('./tb_test/{}/train'.format(stamp), sess.graph)
            
            sess.run(tf.global_variables_initializer())
            sess.run(it_train.initializer, feed_dict={f_train: files_map['train']})

            last = time.time()
            start = time.time()

            for i in range(1, R):
                if i % 10 == 0:     # Check accuracy 
                    summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
                    print("Step %d, accuracy: %g" % (i, acc))
                    now = time.time()
                    elapsed = now - last
                    last = now
                    print("Time elapsed: %.3f, last 200: %.3f, per batch: %.3f" %(now - start, elapsed, elapsed / 200))
                    test_writer.add_summary(summary, i)

                else:               # Record training summaries and train
                    if i % 100 == 99:
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        summary, _ = sess.run([merged, train_step], 
                                feed_dict=feed_dict(True),
                                options=run_options,
                                run_metadata=run_metadata)
                        train_writer.add_run_metadata(run_metadata, "step-%d" % i)
                        train_writer.add_summary(summary, i)
                    else:
                        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                        train_writer.add_summary(summary, i)
            saver.save(sess, save_path, write_meta_graph=False) 
            
            # Save network state after training
            self.network.save()




base_path = '/mnt/data1/Workspace/mgr_2017/datasets/28_04_2017/tf-records/'
var_path = os.listdir(base_path)
paths = [os.path.join(base_path, p) for p in var_path]
files = [[os.path.join(path, f) for f in os.listdir(path)] for path in paths]
files_map = dict(zip(var_path, files))


def _bucketize(x):
    # Setup Pt range from 5GeV to 20GeV into 5 category
    pt_range = [5,10,15,20]
    r = np.digitize(x, pt_range)
    return r

def _convert_data(x):
    """
    Deserialize hits, prod and omtf arrays.
    """
    features = {
        'hits': tf.FixedLenFeature([18,2], tf.float32),
        'prod': tf.FixedLenFeature([4], tf.float32),
        'omtf': tf.FixedLenFeature([6], tf.float32),
    }
    
    parsed = tf.parse_single_example(x, features)
    prod = parsed['prod']
    l = tf.py_func(_bucketize, [prod[0]], tf.int64, stateful=False, name='to-one-hot')
    label = tf.one_hot(l, 5)
    return parsed['hits'], label

def _new_tfrecord_dataset(filename):
    """
    This method is passed to interleave in #new_dataset method. 
    Interleave takes filename tensor from filenames dataset and pass it to this function.
    Now here new TFRecordDataset is produced from single file. It's also place to convert data from binary to
    tensors.
    """
    # TFRecords can be compressed initially. Pass "ZLIB" or "GZIP" if compressed 
    dataset = tf.data.TFRecordDataset(filenames=filename, compression_type=None)
    # Additional options to pass in map
    # - num_threads
    # - output_buffer_size 
    return dataset.map(_convert_data, num_parallel_calls=8)
    

def new_dataset(files_num, batch_size=1, shuffle=False, buff_size=1000):
    """
    Create new input pipeline with source file mixing basing on Dataset API.
    """
    # File names as placeholder
    filenames = tf.placeholder(tf.string, shape=[files_num])
    # Create dataset of input files
    file_dataset = tf.data.Dataset.from_tensor_slices(filenames)
    file_dataset = file_dataset.shuffle(files_num)
    # Now create proper dataset with interleaved samples
    # `interleave` gets filename from `file_dataset` and produces new Dataset (new Queuerunner in the background?)
    dataset = file_dataset.interleave(map_func=_new_tfrecord_dataset, cycle_length=files_num, block_length=1)
    dataset = dataset.repeat(count=1)
    dataset = dataset.batch(batch_size=batch_size)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buff_size, seed=int(time.time()))
    # Create iterator which is initializable. It means that string placeholder will be used in sesson
    # and all dataset will be created then with propper set of input files, i.e. train, eval, test set of input files
    iterator = dataset.make_initializable_iterator()
    return filenames, iterator
    

