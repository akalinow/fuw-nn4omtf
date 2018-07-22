# -*- coding: utf-8 -*-
"""
    Copyright(C) 2018 Jacek Łysiak
    MIT License
    
    Plotting handlers
    Each plotter handers consumes content dict
    and returns list of ('fig name', figure)
"""

from nn4omtf.const_files import FILE_TYPES

from .dataset_stats import dataset_stat_plotter
from .test_stats import test_stat_plotter


PLOTTERS_TABLE = {
    FILE_TYPES.DATASET_STATISTICS: dataset_stat_plotter,
    FILE_TYPES.TEST_STATISTICS: test_stat_plotter
}

PLOTTER_DEFAULTS = {
    'sns_style': 'whitegrid',
    'fig_size': (10, 6.2),
    'title_size': 16,
    'legend_loc': 2,
    'legend_fontsize': 'x-large',
    'xlabel_size': 14,
    'ylabel_size': 14,
    'xticks_size': 12,
    'yticks_size': 12,
    'hist_type': 'step'
}
