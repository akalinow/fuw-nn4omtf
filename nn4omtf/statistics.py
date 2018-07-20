# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    OMTF test statistics generator
"""

import numpy as np

from .const_dataset import DATASET_TYPES
from .const_model import MODEL_RESULTS  


class OMTFStatistics:


    def __init__(self, path_ds_test, path_results):
        self.file_dataset = np.load(path_ds_test)
        self.file_results = np.load(path_results)
        
        self.dataset = self.file_dataset[DATASET_TYPES.TEST].item()
        
        self.note = self.file_results[MODEL_RESULTS.NOTE]
        self.logits = self.file_results[MODEL_RESULTS.RESULTS]
        self.nn_argmax = np.argmax(self.logits, axis=1)


    def __str__(self):
        """
        Return short summary of calculated statistics
        """
        s = ">>>>> OMTF NN Statistics\n"
        s += "> Note: \n%s\n>>>\n" % self.note
        s += "pt bins: " + ",".join(pt_bins)

        print(self.note)
        print(self.results.shape)
        print(self.dataset['HITS'].shape)


    def __enter__(self):
        return self


    def __exit__(self, exec_types, exec_values, traceback):
        self.file_dataset.close()
        self.file_results.close()

