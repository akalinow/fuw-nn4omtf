# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License
    
    Statistics collector 
"""

from nn4omtf.const import CNAMES, NN_CNAMES
import numpy as np
import matplotlib.pyplot as plt

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


    def nn_acc(self):
        nn_pt = self.stats[NN_CNAMES[0]]
        nn_sgn = self.stats[NN_CNAMES[1]]
        pt = self.stats[CNAMES[2]]
        sgn = self.stats[CNAMES[3]]
        tot = pt.size
        gboth = 0
        gpt = 0
        gsgn = 0
        for i in range(tot):
            p = nn_pt[i] == pt[i]
            s = nn_sgn[i] == sgn[i]
            if p:
                gpt += 1
            if s:
                gsgn += 1
            if p and s:
                gboth += 1
        return tot, gboth / tot, gpt / tot, gsgn / tot
    
    def prepare_ptc_grids(self):
        """Create two 2D histograms for OMTF and NN
        X axis -> pt codes
        Y axis -> user defined bins 
        """
        bins = self.bins.copy()
        bins.insert(0, 0) # Add extra bin for OMTF miss
        # Get data
        nn_pt_k = self.stats[NN_CNAMES[0]] + 1      # add +1 due to extra bin 
        omtf_pt_k = self.stats[CNAMES[5]] + 1       # same here
        pt_k = self.stats[CNAMES[0]].astype(int)
        nn_points = zip(pt_k, nn_pt_k)
        omtf_points = zip(pt_k, omtf_pt_k)
        nn_grid = np.zeros((32, len(bins) + 1))
        omtf_grid = np.zeros((32, len(bins) + 1))
        for p, k in nn_points:
            nn_grid[p,k] += 1
        for p, k in omtf_points:
            omtf_grid[p,k] += 1
        return nn_grid, omtf_grid, bins

    def act_curve(self, pt_tresh, save_path=None):
        """Generate activation curve for given pt treshold
        """
        ng, og, bins = self.prepare_ptc_grids()
        tresh = np.digitize(pt_tresh, bins)
        nres = []
        ores = []
        npcs = []
        opcs = []
        for pc in range(1, 32):
            ns = 0
            os = 0
            tns = 0
            tos = 0
            for k in range(0, len(bins) + 1):
                tns += ng[pc][k]
                tos += og[pc][k]
            for k in range(tresh + 1, len(bins) + 1):
                ns += ng[pc][k]
                os += og[pc][k]
            if tns > 0:
                npcs.append(pc)
                nres.append(ns/tns)
            if tos > 0:
                opcs.append(pc)
                ores.append(os/tos)
        plt.plot(npcs, nres, "*-.", opcs, ores, "o-.")
        plt.title("pt_tresh=%.2f" % pt_tresh)
        plt.show()

    def __str__(self):
        nn_acc = self.nn_acc()

        s = "======= OMTFStatistics\n"
        s += "session name: %s\n" % self.sess_name
        s += "pt bins: %s\n\n" % str(self.bins)
        s += "model accuracy:\n"
        s += "\tevents: {}\n\tpt_acc: {}\n\tsgn_acc: {}\n\tboth acc: {}\n\n".format(
                nn_acc[0], nn_acc[2], nn_acc[3], nn_acc[1])
        cols = CNAMES.copy()
        cols.pop() # remove vectors
        cols.pop()
        hfmt = ["{%s:^15}" % e for e in cols]
        hfmt = " | ".join(hfmt) + "\n"
        dfmt = ["{%s:>15.3}" % e for e in cols]
        dfmt = " | ".join(dfmt) + "\n"
        s += hfmt.format(**dict([(k, k) for k in cols]))

        for i in range(self.dlen):
            if 5 < i and i < 7:
                val_d = dict([(k, "...") for k in cols])
            elif i > self.dlen - 3 or i <= 5:
                val_d = dict([(k, float(self.stats[k][i])) for k in cols])
            else:
                continue
            s += dfmt.format(**val_d)
        s += "\n"
        return s

