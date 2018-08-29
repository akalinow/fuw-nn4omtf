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


def _plot_response(content, opts):
    plots = []
    hists = content[TEST_STATISTICS_FIELDS.HISTOGRAMS]
    sns.set_style(opts.sns_style_nogrid)

    to_plot = [
        ('nn_self_all', 'warunki rzeczywiste', True),
        ('nn_self_wn_all', 'warunki treningowe', False)
    ]

    for k, desc, skip_first in to_plot:
        hs, xd, yd = hists[k]
        yn, _, yl = yd
        xn, _, xl = xd
        title = "Częstość odpowiedzi sieci\n(Norm. względem kodu rzeczywistego, %s)" % desc

        fig, ax = plt.subplots(1,1, figsize=opts.fig_size)
        data = hs / hs.sum(axis=1)
        if skip_first:
            data = data[:,1:]
            xn -= 1
            xl = xl[1:]
        cax = plt.imshow(data, vmax=1, cmap='hot')
        plt.title(title, size=opts.title_size)
        plt.ylabel('Kod klasy - odpowiedź', size=opts.ylabel_size)
        plt.xlabel('Rzeczywisty kod klasy', size=opts.xlabel_size)
        plt.xticks(range(xn), xl, size=opts.xticks_size, rotation='vertical')
        plt.yticks(range(yn), yl, size=opts.yticks_size)
        cb = plt.colorbar()
        cb.set_label('Unormowana częstość odpowiedzi', size=opts.colorbar_fontsize)
        fig.tight_layout()
        plots += [('nn-response-%s' % (k), fig)]

    return plots 


def find_code(pt):
    l = [e for e in PT_CODES_DATA if e[1] <= pt and pt < e[2]]
    assert len(l) == 1
    return tuple(l[0][:-1])


def _plot_curves(content, opts):
    plots = []
    curves = content[TEST_STATISTICS_FIELDS.CURVES]
    sns.set_style(opts.sns_style)

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
                    o_min, o_max = a, b
                    break
            ax.plot(pt_codes, n_arr[idx], 'o-.', label='NN $p \geq %.1f$ GeV' % (nn_pt_min))
            ax.plot(pt_codes, o_arr[o_idx], 'o-.', label='OMTF $p \geq %.1f$ GeV' % (o_min))
            cut_desc = 'Cięcie: kod %d, $p\in[%.1f, %.1f[$ GeV' % (ptc, ptc_min, ptc_max)
            plt.axvline(x=ptc, c='#ff1111', linestyle='-.', linewidth=1)
            ax.set_title("Krzywa włączeniowa  $p_T$ > %.1f Gev\n%s\nZdarzenia: %s" % (treshold, cut_desc,desc), 
                    size=opts.title_size)
            ax.legend(loc=opts.legend_loc,  fontsize=opts.legend_fontsize, frameon=True, fancybox=True, framealpha=0.7)
            ax.set_xlabel('Kod pędowy', size=opts.xlabel_size)
            plt.yticks(size=opts.yticks_size)
            ax.set_ylabel('Efektywność', size=opts.ylabel_size)
            plt.xticks(range(1, len(pt_codes) + 1), pt_codes, size=opts.xticks_size)
            ax.set_ylim(-0.05, 1.05)
            fig.tight_layout()
            name = 'curve-%s-%.1f' % (nk, treshold)
            name = name.replace('.', '-')
            plots += [(name, fig)]
    return plots


def test_stat_plotter(content, config):
    opts = dict_to_object(config)
    plots = []
    plots += _plot_curves(content, opts)
    plots += _plot_response(content, opts)
    return plots
