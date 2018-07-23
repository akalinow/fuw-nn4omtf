# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    Plot data from datasets.
"""

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from nn4omtf.const_stats import TEST_STATISTICS_FIELDS
from nn4omtf.utils import dict_to_object, obj_elems
from nn4omtf.const_pt import PT_CODES_DATA


def _plot_pdist(content, opts):
    pass


def find_code(pt):
    l = [e for e in PT_CODES_DATA if e[1] <= pt and pt < e[2]]
    assert len(l) == 1
    return tuple(l[0][:-1])

def get_desc(key):
    descs = {
        'curves_not_null_': '$p_T$ class $\\neq$ NULL',
        'curves_q12_pt999_': 'OMTF quality = 12, OMTF $p_T$ > 0',
        'curves_pt999_': 'OMTF $p_T$ > 0',
        'curves_q12_': 'OMTF quality = 12',
        'curves_all_': 'wszystkie'
    }
    return descs[key]

def _plot_curves(content, opts):
    plots = []
    curves = content[TEST_STATISTICS_FIELDS.CURVES]
    pt_bins = content[TEST_STATISTICS_FIELDS.PT_BINS]
    pt_codes = content[TEST_STATISTICS_FIELDS.PT_CODES]
    cls_sz = len(pt_bins) + 1
    keys = [x[:-2] for x in curves.keys() if x.endswith('nn')]
    nn_curves = [curves[key + 'nn'] for key in keys]
    omtf_curves = [curves[key + 'omtf'] for key in keys]

    for key, nnc, omtfc in zip(keys, nn_curves, omtf_curves):
        desc = get_desc(key)
        for idx in range(2, cls_sz):
            fig, ax = plt.subplots(1,1, figsize=opts.fig_size)
            pt_val = pt_bins[idx-1]
            ptc, ptc_min, ptc_max = find_code(pt_val)
            ax.plot(pt_codes, nnc[idx], 'o-.', label='NN')
            ax.plot(pt_codes, omtfc[idx], 'o-.', label='OMTF')
            plt.axvline(x=ptc, c='#ff1111', linestyle='-.', label='kod: %d $p\in[%.1f, %.1f[$ GeV, $p_T$: %.1f GeV' % (ptc, ptc_min, ptc_max, pt_val))
            ax.set_title("Krzywa włączeniowa  $p_T$ > %.1f Gev\nZdarzenia: %s" % (pt_val, desc), 
                    size=opts.title_size)
            ax.legend(loc=opts.legend_loc,  fontsize=opts.legend_fontsize)
            ax.set_xlabel('Kod pędowy', size=opts.xlabel_size)
            plt.yticks(size=opts.yticks_size)
            ax.set_ylabel('Efektywność', size=opts.ylabel_size)
            plt.xticks(size=opts.xticks_size)
            fig.tight_layout()
            plots += [('%s%.1f' % (key, pt_val), fig)]

    return plots


def test_stat_plotter(content, config):
    opts = dict_to_object(config)
    sns.set_style(opts.sns_style)
    plots = []
    plots += _plot_curves(content, opts)
#    plots += _plot_pdist(content, opts)
    return plots
