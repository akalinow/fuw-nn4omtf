# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License
    
    Statistics collector constatnts
"""

from nn4omtf.network.input_pipe_const import PIPE_EXTRA_DATA_NAMES

# Column names with NN outputs
NN_CNAMES = [
    "NN_PT_CLASS",      # NN output data
    "NN_SGN_CLASS",
    "NN_PT_LOGITS",
    "NN_SGN_LOGITS"
]

# This list of labels 
CNAMES = PIPE_EXTRA_DATA_NAMES + NN_CNAMES
