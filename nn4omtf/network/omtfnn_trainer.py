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

    def __init__(self, dataset_path=None):
        self.dataset_path = dataset_path
        self.dataset = None
        self._load_dataset() # Load dataset
        pass


    def _load_dataset(self):
        """Load dataset structure based upon given path.
        NoFileFoundError can be raised if path is wrong.
        """
        if self.dataset_path is None:
            return
        self.dataset = OMTFDataset(self.dataset_path)

    def set_omtf_nn(self, omtfnet):
        """Set prepared model for training."""
        pass

