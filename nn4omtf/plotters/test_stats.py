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

    to_plot = [
        ('ptc_nnpt_all', 'ptc_omtfpt_all', 'wszystkie'),
        ('ptc_nnpt_ptnnq12', 'ptc_omtfpt_ptnnq12', 'OMTF quality = 12, OMTF $p_T$ > 0' ),
        ('ptc_nnpt_ptnn','ptc_omtfpt_ptnn', 'OMTF $p_T$ > 0'),
        ('ptc_nnpt_q12', 'ptc_omtfpt_q12', 'OMTF quality = 12')
    ]
    for nk, ok, desc in to_plot:
        n_arr, n_xs, n_ys = curves[nk]
        o_arr, o_xs, o_ys = curves[ok]

        pt_codes = n_ys[2]
        N = n_xs[0]
        nn_ranges = n_xs[1]
        omtf_ranges = o_xs[1]
        
        for idx, (nn_pt_min, nn_pt_max) in zip(range(2, N), nn_ranges[2:]):
            fig, ax = plt.subplots(1,1, figsize=opts.fig_size)
            treshold = nn_pt_min 
            ptc, ptc_min, ptc_max = find_code(treshold)
            for i, (a, b) in enumerate(omtf_ranges[1:]):
                if a >= treshold:
                    o_idx = i + 1
                    break
            ax.plot(pt_codes, n_arr[idx], 'o-.', label='NN')
            ax.plot(pt_codes, o_arr[o_idx], 'o-.', label='OMTF')
            plt.axvline(x=ptc, c='#ff1111', linestyle='-.', label='kod: %d $p\in[%.1f, %.1f[$ GeV, $p_T$: %.1f GeV' % (ptc, ptc_min, ptc_max, treshold))
            ax.set_title("Krzywa włączeniowa  $p_T$ > %.1f Gev\nZdarzenia: %s" % (treshold, desc), 
                    size=opts.title_size)
            ax.legend(loc=opts.legend_loc,  fontsize=opts.legend_fontsize)
            ax.set_xlabel('Kod pędowy', size=opts.xlabel_size)
            plt.yticks(size=opts.yticks_size)
            ax.set_ylabel('Efektywność', size=opts.ylabel_size)
            plt.xticks(size=opts.xticks_size)
            fig.tight_layout()
            plots += [('curve-%s-%.1f' % (nk, treshold), fig)]
    return plots


def test_stat_plotter(content, config):
    opts = dict_to_object(config)
    sns.set_style(opts.sns_style)
    plots = []
    plots += _plot_curves(content, opts)
    return plots
