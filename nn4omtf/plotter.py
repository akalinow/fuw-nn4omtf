# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    Plot nn4omtf data.
"""

import numpy as np
import os
import warnings
from .plotters import PLOTTERS_TABLE, PLOTTER_DEFAULTS
import matplotlib.pyplot as plt


class OMTFPlotter:
    
    def __init__(self, file_path, outdir='.', **kw):
        """
        Create file plotter and override its default settings, if needed.
        Args:
            file_path: data file
            **kw: plotter configuration to override
        """
        self.file_path = file_path
        self.data_file = np.load(file_path)
        self.set_outdir(outdir)
        self.plotter_config = PLOTTER_DEFAULTS
        self.plotter_config = self.get_plotter_config(**kw)


    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


    def close_figures(self):
        try:
            for n, f in self.plots:
                f.close()
        except:
            pass
        self.plots = None


    def close(self):
        self.close_figures()
        self.data_file.close()


    def get_plottables(self):
        """
        Get list of all plottable data in loaded file.
        Returns:
            list of plottable element's names
        """
        files = self.data_file.files
        plottable = [f for f in files if f in PLOTTERS_TABLE.keys()]
        return plottable
    

    def set_outdir(self, outdir='.'):
        """
        Set root output directory.
        Args:
            outdir: output directory, current cwd is default
        """
        self.outdir = outdir


    def get_plotter_config(self, **kw):
        """
        Get current plotter configuration, eventually updated with **kw.
        Args:
            **kw: configuration to override
        Returns:
            plotter configuration dict
        """
        conf = self.plotter_config
        for k, v in kw.items():
            if k in conf:
                conf[k] = v
        return conf


    def plot(self, file_type, **kw):
        """
        Plot content of file.
        Args:
            file_type: name of plottable file type to generate plots
            **kw: configuration to override
        """
        assert_info = "File type `{}` is not plottable!\n" + \
             "Plottable files available in {}: \n" + \
            "\n".join(self.get_plottables())
        assert file_type in self.get_plottables(), assert_info.format(
            file_type, self.file_path)
        content = self.data_file[file_type].item()
        plotter = PLOTTERS_TABLE[file_type]
        config = self.get_plotter_config(**kw)
        plt.ioff()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.plots = plotter(content, config)
        plt.ion()


    def save(self, group_dir=None):
        path = self.outdir
        if group_dir is not None:
            path = os.path.join(path, group_dir)
        os.makedirs(path, exist_ok=True)
        for name, fig in self.plots:
            fig.savefig(os.path.join(path, name + '.png'))


    def get_plot_names(self):
        return [n for n, _ in self.plots]


    def get_plot(self, name):
        for n, fig in self.plots:
            if name == n:
                return fig

