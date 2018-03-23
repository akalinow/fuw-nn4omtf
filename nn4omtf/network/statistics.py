# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License
    
    Statistics collector 
"""

class OMTFStatistcs:
    ARGS = [
        "PT_CODE",
        "PT_VAL"
    ]

    def __init__(self, sess_names):
        self.sess_name = sess_name
        self.stats = []

    def set_bins(self, out_class_bins):
        self.bins = out_class_bins

    def append(**args):
        self.stats.append(args)
        

