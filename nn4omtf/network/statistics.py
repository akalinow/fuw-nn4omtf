# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License
    
    Statistics collector 
"""

from nn4omtf.network.stats_const import CNAMES
import numpy as np


class OMTFStatistics:
    """OMTFStatistics
    This object contains almost all summary data from
    network tests and validations (no need to store training data).

    Each object contains `session name` and `bins` which encodes pt classes.
    (They're defined in user code with model builder.)
    
    Columns of data can be provided using `append`.
    
    This object provides data to create online and offline 
    advanced model statistics.
    """

    def __init__(self, sess_name):
        self.sess_name = sess_name
        self.stats = dict([(k, None) for k in CNAMES])


    def set_bins(self, out_class_bins):
        self.bins = out_class_bins


    def append(self, cols_dict):
        for k in CNAMES:
            if self.stats[k] is None:
                arr = cols_dict[k]
            else:
                arr = np.append(self.stats[k], cols_dict[k], axis=0)
            self.stats[k] = arr
        self.dlen = self.stats[CNAMES[0]].shape[0]


    def __str__(self):
        s = "======= OMTFStatistics\n"
        s += "session name: %s\n" % self.sess_name
        s += "pt bins: %s\n\n" % str(self.bins)
        cols = CNAMES.copy()
        cols.pop() # remove vectors
        cols.pop()
        hfmt = ["{%s:^15}" % e for e in cols]
        hfmt = " | ".join(hfmt) + "\n"
        dfmt = ["{%s:>15.3}" % e for e in cols]
        dfmt = " | ".join(dfmt) + "\n"
        s += hfmt.format(**dict([(k, k) for k in cols]))

        for i in range(self.dlen):
            if 30 < i and i < 34:
                val_d = dict([(k, "...") for k in cols])
            elif i > self.dlen - 10 or i <= 30:
                val_d = dict([(k, float(self.stats[k][i])) for k in cols])
            else:
                continue
            print(val_d)
            s += dfmt.format(**val_d)
        s += "\n"
        return s

