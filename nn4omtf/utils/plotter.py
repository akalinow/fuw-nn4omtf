# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    Plotting data from *.npz files
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from nn4omtf.const import PLT_DATA_TYPE, PT_CODE_RANGE


class OMTFPlotter:
    def __init__(self, arrdict):
        self.data = arrdict
        self.type = self.data['datatype']
        self.plots = []

    def from_npz(fname):
        data = np.load(fname)
        return OMTFPlotter(data)

    def plot(self, outdir, prefix=""):
        self.outdir = outdir
        self.pref = prefix
        if self.type == PLT_DATA_TYPE.PROB_DIST:
            self._plot_dist()
        if self.type == PLT_DATA_TYPE.TRAIN_LOG:
            self._plot_trainlog()

    def _plot_trainlog(self):
        names = self.data['names']
        data = self.data['data']
        for i in range(len(names)):
            n = names[i]
            vals = data[:,i]
            fig, ax = plt.subplots()
            fs = ax.plot(vals)
            ax.set_xlabel('Validation step')
            ax.set_title('Log monitor of: %s' % n) 
            self._save(
                    scope=PLT_DATA_TYPE.TRAIN_LOG,
                    name="%s.png" % n,
                    fig=fig)
            plt.close()


    def _plot_dist(self):
        figs = []
        pt = self.data['pt_dist']
        sgn = self.data['sgn_dist']
        bins = self.data['bins']
        bl = len(bins)
        cls_tcs = ['NULL']\
            + ['(%.1f-%.1f]' % (bins[i], bins[i+1]) for i in range(bl-1)]\
            + ['%.1f <' % bins[-1]]
        print(cls_tcs)
        for ptc in range(1, PT_CODE_RANGE):
            ptd = pt[ptc] # dist for each bin
            fig, ax = plt.subplots()
            fs = ax.bar(cls_tcs, ptd, edgecolor='black', linewidth=1.0)
            ax.set_xlabel('Output $p_T$ class')
            ax.set_ylabel('Probability')
            ax.set_ylim(0, 1)
            ax.set_title('$p_T$ class probability distribution for $p_T$ code %d' % ptc) 
            ax.set_facecolor("#CAEEEE")
            for face in fs:
                face.set_facecolor("#389595")
            self._save(
                    scope=PLT_DATA_TYPE.PROB_DIST,
                    name="ptd_%#02d.png" % ptc,
                    fig=fig)
            plt.close()

        cls_tcs = ['NULL', '-1', '+1'] 
        for ptc in range(1, PT_CODE_RANGE):
            for sign in range(3):
                sgnd = sgn[ptc][sign] # dist for (pt code, +/-)
                print("#: %d %d" % (ptc, sign))
                print(sgnd)
                fig, ax = plt.subplots()
                fs = ax.bar(cls_tcs, sgnd, edgecolor='black', linewidth=1.0)
                ax.set_xlabel('Output sign class')
                ax.set_ylabel('Probability')
                ax.set_ylim(0, 1)
                ax.set_title('Sign class probability distribution for $p_T$ code %d and sign %s' % (ptc, cls_tcs[sign])) 
                ax.set_facecolor("#CAEEEE")
                for face in fs:
                    face.set_facecolor("#389595")
                self._save(
                        scope=PLT_DATA_TYPE.PROB_DIST,
                        name="sgn_%#02d_%s.png" % (ptc, cls_tcs[sign]),
                        fig=fig)
                plt.close()
            

    def _save(self, scope, name, fig):
        out = os.path.join(self.outdir, scope)
        if not os.path.exists(out):
            os.makedirs(out)
        fname = os.path.join(out, name)
        fig.savefig(fname)


