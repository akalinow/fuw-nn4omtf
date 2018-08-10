# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    Plot data from datasets.
"""

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from nn4omtf.const_dataset import DSET_STAT_FIELDS, ORD_TYPES, HIST_TYPES
from nn4omtf.utils import dict_to_object, obj_elems


def _plot_hist(orig_vals, trans_vals, bins, title, opts, treshold=None):
    fig, ax = plt.subplots(1,1, figsize=opts.fig_size)
    plt.hist(x=bins[:-1]+0.1, bins=bins, histtype=opts.hist_type,  weights=trans_vals, linewidth=1.5, label='TRANS')
    plt.hist(x=bins[:-1]+0.1, bins=bins, histtype=opts.hist_type, weights=orig_vals, linewidth=1.5, label='ORIG')
    if treshold is not None:
        idx = (np.abs(bins[:-1]-treshold)).argmin()
        below = np.sum(orig_vals[:idx])
        above = np.sum(orig_vals[idx:])
        plt.axvline(x=treshold, c='#ff1111', linestyle='-.', 
                label='treshold={}, (B: {:.2e}, A: {:.2e}, F%: {:.2f})'.format(
                treshold, below, above, (above/(above+below))*100))

    plt.yscale('log')
    plt.title(title, size=opts.title_size)
    plt.xlabel('Wartość', size=opts.xlabel_size)
    plt.ylabel('Liczba wystąpień', size=opts.ylabel_size)
    plt.legend(loc=opts.legend_loc, fontsize=opts.legend_fontsize)
    plt.xticks(size=opts.xticks_size)
    plt.yticks(size=opts.xticks_size)
    plt.tight_layout()
    return fig


def _plot_hists_total(content, opts):
    """
    Plot histograms of HITS arr values and averaged values of HITS array.
    """
    bins = content[DSET_STAT_FIELDS.HISTS_BINS]
    tresh = content[DSET_STAT_FIELDS.TRESHOLD]

    hist_total_orig = content[DSET_STAT_FIELDS.HISTS_TOTAL_ORIG]
    hist_total_trans = content[DSET_STAT_FIELDS.HISTS_TOTAL_TRANS]
    hist_total_orig_avg = hist_total_orig[HIST_TYPES.AVG]
    hist_total_trans_avg = hist_total_trans[HIST_TYPES.AVG]
    hist_total_orig_vals = hist_total_orig[HIST_TYPES.VALS]
    hist_total_trans_vals = hist_total_trans[HIST_TYPES.VALS]

    fig_vals = _plot_hist(hist_total_orig_vals, hist_total_trans_vals, bins,
        "Rozkład wartości elementów macierzy HITS - wszystkie kody pędowe", opts)
    fig_avg = _plot_hist(hist_total_orig_avg, hist_total_trans_avg, bins, 
        "Rozkład wartości średnich elementów macierzy HITS - wszystkie kody pędowe", 
        opts, treshold=tresh)
    return [('hists_total_vals', fig_vals), ('hists_total_avg', fig_avg)]


def _plot_hists_codes(content, opts):
    """
    Plot histograms of HITS arr values and averaged values of HITS array.
    """
    figs = []
    signatures = content[DSET_STAT_FIELDS.MUON_SIGNATURES]
    bins = content[DSET_STAT_FIELDS.HISTS_BINS]
    tresh = content[DSET_STAT_FIELDS.TRESHOLD]

    hist_orig = content[DSET_STAT_FIELDS.HISTS_ORIG]
    hist_trans = content[DSET_STAT_FIELDS.HISTS_TRANS]

    hist_orig_avg = hist_orig[HIST_TYPES.AVG]
    hist_trans_avg = hist_trans[HIST_TYPES.AVG]
    for (pt, sgn), ho_avg, ht_avg in zip(signatures, hist_orig_avg, hist_trans_avg):
        fig_avg = _plot_hist(ho_avg, ht_avg, bins,
            "Rozkład wartości średnich elementów macierzy HITS - kod pędowy: %d, znak: %s" % (pt, sgn), 
            opts, treshold=tresh)
        figs.append(('hist_avg_%d%s' % (pt, sgn), fig_avg))

    hist_orig_vals = hist_orig[HIST_TYPES.VALS]
    hist_trans_vals = hist_trans[HIST_TYPES.VALS]
    for (pt, sgn), ho, ht in zip(signatures, hist_orig_vals, hist_trans_vals):
        fig_avg = _plot_hist(ho, ht, bins,
            "Rozkład wartości elementów macierzy HITS - kod pędowy: %d, znak: %s" % (pt, sgn), 
            opts)
        figs.append(('hist_%d%s' % (pt, sgn), fig_avg))
    return figs


def _plot_train_ordering(content, opts):
    """
    Plot train examples order
    """
    train_examples_ordering = content[DSET_STAT_FIELDS.TRAIN_EXAMPLES_ORDERING]

    fig, ax = plt.subplots(1,1, figsize=opts.fig_size)
    es = obj_elems(ORD_TYPES)
    es.sort()
    es.reverse()
    for k in es:
        ax.plot(train_examples_ordering[k][:], label=k)
        ax.set_title("Kolejność zdarzeń w zbiorze TRAIN - średnie $p_T$ w oknie rozmiaru 32", 
                size=opts.title_size)
        ax.legend(loc=opts.legend_loc,  fontsize=opts.legend_fontsize)
        ax.set_xlabel('Numer okna', size=opts.xlabel_size)
        plt.yticks(size=opts.yticks_size)
        ax.set_ylabel('$<p_T>$ (GeV)', size=opts.ylabel_size)
        plt.xticks(size=opts.xticks_size)
        fig.tight_layout()
    return [('train_examples_ordering', fig)]


def _plot_examples_distr(content, opts):
    signatures = content[DSET_STAT_FIELDS.MUON_SIGNATURES]
    data = content[DSET_STAT_FIELDS.MUON_SIGNATURES_DISTR]
    keys = list(data.keys())
    keys.sort()
    vals = [data[k] for k in keys]
    _all = list(zip(signatures, zip(*tuple(vals))))

    _all = [(int(v), str(s), d) for (v, s), d in _all]
    _all.sort(key=lambda x: x[1])
    _all.sort(key=lambda x: x[0])
    
    labels = ['0'] + ["%s%d" % ('+' if s == 'p' else '-', v) for v, s, _ in _all]
    vals = list(zip(*[d for _, _, d in _all]))
    _vals = []
    for l in vals:
        not_null = [x for x, _ in l] 
        null = sum([x for _, x in l])
        _vals.append([null] + not_null)
    vals = _vals

    w = 0.5
    l = len(labels)
    ws = [w for _ in labels]
    xs = np.arange(l)

    fig, axs = plt.subplots(3,1, figsize=(10, 9), dpi=150, sharex=True)

    for k, vs, ax in zip(keys, vals, axs):
        ax.bar(xs, vs, ws, label=k)
        ax.set_ylabel('Liczba przykładów', size=opts.ylabel_size)
        ax.set_title(k, size=opts.title_size)
        ax.set_yscale('log')

    plt.xticks(xs)
    ax.set_xticklabels(labels, rotation='vertical')
    fig.tight_layout()
    return [('examples_distr', fig)]


def dataset_stat_plotter(content, config):
    opts = dict_to_object(config)
    sns.set_style(opts.sns_style)
    plots = []
    plots += _plot_train_ordering(content, opts)
    plots += _plot_hists_total(content, opts)
    plots += _plot_hists_codes(content, opts)
    plots += _plot_examples_distr(content, opts)
    return plots


