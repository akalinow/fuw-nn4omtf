# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    Plot nn4omtf data.
"""

import numpy as np
import os
from nn4omtf.plotters import PLOTTERS_TABLE
from nn4omtf.const_files import FILE_TYPES
import matplotlib.pyplot as plt

class OMTFPlotter:
    
    def __init__(self, file_path, file_type, **kw):
        data_file = np.load(file_path)
        content = data_file[file_type].item()
        
        membs = [e for e in vars(FILE_TYPES) if not e.startswith('_')]    
        assert file_type in membs, "Unknown file type: `%s" % file_type
        plotter = PLOTTERS_TABLE[file_type]
        plt.ioff()
        self.plots = plotter(content, **kw)
        plt.ion()


    def save(self, outdir):
        os.makedirs(outdir)
        for name, fig in self.plots:
            path = os.path.join(outdir, name + '.png')
            fig.savefig(path)


    def get_names(self):
        return [n for n, _ in self.plots]


    def get_plot(self, name):
        for n, fig in self.plots:
            if name == n:
                return fig
