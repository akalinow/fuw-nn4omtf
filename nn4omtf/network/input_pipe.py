# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    Handy input pipe wrapper.
"""

import tensorflow as tf
from nn4omtf.dataset import OMTFDataset
from nn4omtf.network.input_pipe_helpers import setup_input_pipe
from nn4omtf.const import PAHSE_NAME, PIPE_MAPPING_TYPE

class OMTFInputPipe:
    """
    Utility class which helps with some operation on dataset.
    """

    def __init__(self, 
            dataset, 
            name, 
            hits_type, 
            out_class_bins,
            batch_size=1, 
            shuffle=False, 
            reps=1, 
            mapping_type=PIPE_MAPPING_TYPE.INTERLEAVE, 
            limit_examples=None,
            detect_no_signal=False,
            remap_data=None):
        """Create input pipe
        Args:
            dataset: OMTFDataset object
            name:
            hist_type:
            out_class_bins:
            batch_size:
            reps:
            mapping_type:
            limit_examples:
            detect_no_signal:
            remap_data:
        """
        self.dataset = dataset
        filenames, files_n = self.dataset.get_dataset(name=name)
        self.filenames = filenames
        is_training = True if name == PHASE_NAME.TRAIN else False
        placeholder, iterator = setup_input_pipe(
            files_n=files_n,
            name=name,
            compression_type=self.dataset.get_compression_type(),
            in_type=hits_type,
            out_class_bins=out_class_bins,
            batch_size=batch_size,
            shuffle=shuffle,
            reps=reps,
            mapping_type=mapping_type,
            is_training=is_training)

        self.placeholder = placeholder
        self.iterator = iterator
        # WATCH OUT!!
        # Doing something like sess.run(iterator.get_next()) causes blowing 
        # your computer with unsued threads! DON'T DO THAT!!
        # TensorFlow deep in sources has apropriate annotation... hidden.
        # `get_next()` creates operation node and allocates ALL needed
        # resources! Threads also!
        # 12.02.2018 - jlysiak
        self.next_op = iterator.get_next()
        self.session = None


    def initialize(self, session):
        """Initialize input pipe.
        Args:
            session: tf session
        """
        feed_dict = {self.placeholder: self.filenames}
        session.run(self.iterator.initializer, feed_dict=feed_dict)
        self.session = session


    def fetch(self):
        """Fetch examples from input pipe.
        Returns:
            2-tuple of (data_dict, extra_data_dict) or (None, None) 
            if we reached end of dataset
        """
        assert self.session is not None, "Input pipe must be initialized!"
        try:
            return self.session.run(self.next_op)
        except tf.errors.OutOfRangeError:
            return None, None
