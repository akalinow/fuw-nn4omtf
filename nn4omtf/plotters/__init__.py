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


PLOTTERS_TABLE = {
    FILE_TYPES.DATASET_STATISTICS: dataset_stat_plotter
}


