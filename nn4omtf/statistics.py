# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    OMTF test statistics generator
"""

import numpy as np

from .utils import dict_to_json
from .const_dataset import DATASET_TYPES, DATASET_FIELDS
from .const_model import MODEL_RESULTS  
from .const_stats import TEST_STATISTICS_FIELDS
from .const_files import FILE_TYPES

from .const_pt import PT_CODES_BINS, OMTF_BINS, PT_CODES_RANGES


class OMTFStatistics:
    """
    Generate statistic from TEST run results.
    NOTE: TEST dataset actually cannot be too big.

    """

    def __init__(self, path_ds_test, path_results):
        """
        Prepare a lot of data to create many histograms which 
        are base for all others statistics.
        """
        self.file_dataset = np.load(path_ds_test)
        self.file_results = np.load(path_results)
        dataset = self.file_dataset[DATASET_TYPES.TEST].item()

        isnull = dataset[DATASET_FIELDS.IS_NULL]
        N = isnull.shape[0]
        
        # ========= SIGN
        sign_n = 3
        sign_labels = ['N', '+', '-']

        # ========= NN DATA
        nn_logits_arr = self.file_results[MODEL_RESULTS.RESULTS]
        nn_cls_arr = np.argmax(nn_logits_arr, axis=1)
        nn_pt_arr = (nn_cls_arr + 1) // 2
        nn_sign_arr = np.where(nn_cls_arr == 0, 0, (nn_cls_arr + 1) % 2 + 1)

        nn_bins = self.file_results[MODEL_RESULTS.PT_BINS]

        nn_cls_n = 2 * len(nn_bins) + 1
        nn_pt_n = len(nn_bins) + 1
        nn_pt_labels = ['N'] + list(range(1, nn_pt_n))
        nn_cls_labels = ['N'] + [y for x in  [['+%d' % l, '-%d' % l] for l in nn_pt_labels[1:]] for y in x]
        nn_pt_ranges = [(None, 0)] + list(zip(nn_bins[:-1], nn_bins[1:])) + [(nn_bins[-1], None)]

        # ========= OMTF DATA
        omtf_bins = OMTF_BINS
        omtf_ptval_arr = dataset[DATASET_FIELDS.OMTF_PT]
        omtf_sign_arr = dataset[DATASET_FIELDS.OMTF_SIGN]
        omtf_q_arr = dataset[DATASET_FIELDS.OMTF_QUALITY]
        omtf_cls_arr = OMTFStatistics.get_cls(omtf_ptval_arr, omtf_sign_arr, omtf_bins)
        omtf_nn_cls_arr = OMTFStatistics.get_cls(omtf_ptval_arr, omtf_sign_arr, nn_bins)
        omtf_pt_arr = (omtf_cls_arr + 1) // 2
        omtf_sign_arr = np.where(omtf_cls_arr == 0, 0, (omtf_cls_arr + 1) % 2 + 1)

        omtf_cls_n = 2 * len(omtf_bins) + 1
        omtf_pt_n = len(omtf_bins) + 1
        omtf_pt_labels = ['N'] + list(range(1, omtf_pt_n))
        omtf_cls_labels = ['N'] + [y for x in  [['+%d' % l, '-%d' % l] for l in omtf_pt_labels[1:]] for y in x]
        omtf_pt_ranges = [(None, 0)] + list(zip(OMTF_BINS[:-1], OMTF_BINS[1:])) + [(OMTF_BINS[-1], None)]
        
        # ========= MUON DATA
        muon_pt_arr = dataset[DATASET_FIELDS.PT_VAL]
        muon_sign_arr = dataset[DATASET_FIELDS.SIGN]
        muon_sign_cls_arr = np.where(dataset[DATASET_FIELDS.SIGN] > 0, 1, 2)
        muon_sign_cls_wn_arr = np.where(isnull, 0, muon_sign_cls_arr)
        muon_nn_cls_arr = OMTFStatistics.get_cls(muon_pt_arr, muon_sign_arr, nn_bins)
        muon_omtf_cls_arr = OMTFStatistics.get_cls(muon_pt_arr, muon_sign_arr, omtf_bins)
        muon_nn_cls_wn_arr = OMTFStatistics.get_cls(muon_pt_arr, muon_sign_arr, nn_bins, isnull)
        muon_omtf_cls_wn_arr = OMTFStatistics.get_cls(muon_pt_arr, muon_sign_arr, omtf_bins, isnull)


        muon_ptc_arr = dataset[DATASET_FIELDS.PT_CODE] - 1
        muon_ptc_list = np.unique(muon_ptc_arr)
        muon_ptc_n = len(muon_ptc_list)
        muon_ptc_ranges = PT_CODES_RANGES
        muon_ptc_labels = list(map(int, muon_ptc_list + 1))

        # ===== SOURCE DATA FOR HISTOGRAMS  

        # key, data array normalized, classes range, ranges, classes labels 
        data = [
            # pT codes
            ('muon_ptc', muon_ptc_arr, muon_ptc_n, muon_ptc_ranges, muon_ptc_labels),
            # Ground Truths for NN and OMTF
            ('muon_nn_cls', muon_nn_cls_arr, nn_cls_n, None, nn_cls_labels),
            ('muon_nn_cls_wn', muon_nn_cls_wn_arr, nn_cls_n, None, nn_cls_labels),
            ('muon_omtf_cls', muon_omtf_cls_arr, omtf_cls_n, None, omtf_cls_labels),
            # NN and OMTF full outputs 
            ('nn_cls', nn_cls_arr, nn_cls_n, None, nn_cls_labels),
            ('omtf_cls', omtf_cls_arr, omtf_cls_n, None, omtf_cls_labels),
            # OMTF outputs applied to NN bins
            ('omtf_nn_cls', omtf_nn_cls_arr, nn_cls_n, None, nn_cls_labels),
            # NN and OMTF pt only outputs (signs merged)
            ('nn_pt', nn_pt_arr, nn_pt_n, nn_pt_ranges, nn_pt_labels),
            ('omtf_pt', omtf_pt_arr, omtf_pt_n, omtf_pt_ranges, omtf_pt_labels),
            # MUON, NN, OMTF sign only data
            ('muon_sign', muon_sign_cls_arr, sign_n, None, sign_labels),
            ('muon_sign_wn', muon_sign_cls_wn_arr, sign_n, None, sign_labels),
            ('nn_sign', nn_sign_arr, sign_n, None, sign_labels),
            ('omtf_sign', omtf_sign_arr, sign_n, None, sign_labels),
        ]
        self.data = dict([(x[0], x[1:]) for x in data])

        # Events masks
        self.masks = {
            'all': np.ones(N).astype(np.bool),
            # OMTF algorithm had good enough data quality
            'q12': omtf_q_arr == 12.,
            # OMTF tried to guess pt value
            'ptnn': omtf_pt_arr > 0,
            # Both, non-null answer and good quality
            'ptnnq12': np.logical_and(omtf_q_arr == 12., omtf_pt_arr > 0)
        }

        # ===== HISTOGRAMS TO GENERATE
        # == (hist key, xs data key, ys data key, mask key)
        _htg = [
            ('nn_self', 'muon_nn_cls', 'nn_cls'),
            ('nn_self_wn', 'muon_nn_cls_wn', 'nn_cls'),
            ('omtf_self', 'muon_omtf_cls', 'omtf_cls'),
            ('omtf_as_nn', 'muon_nn_cls', 'omtf_nn_cls'),
            ('ptc_nnpt', 'muon_ptc', 'nn_pt'),
            ('ptc_omtfpt', 'muon_ptc', 'omtf_pt'),
        ]
        hists_to_generate = []
        # Generate histograms for all masks
        for k, sx, sy in _htg:
            for mk, _ in self.masks.items():
                hists_to_generate += [(k + '_' + mk, sx, sy, mk)]

        # ===== CURVES TO GENERATE
        # Put index of entry from list above
        _ctg = ['ptc_nnpt', 'ptc_omtfpt']
        curves_to_generate = [x+'_'+y for y in self.masks.keys() for x in _ctg]

        # ===== ACCURACY TO CALCULATE
        # Put index of entry from list above
        _atc = ['nn_self', 'omtf_self', 'omtf_as_nn']
        accuracy_to_calculate = [x+'_'+y for y in self.masks.keys() for x in _atc]

        self.generate_histograms(hists_to_generate)
        self.generate_curves(curves_to_generate)
        self.calculate_accuracy(accuracy_to_calculate)


    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        self.file_dataset.close()
        self.file_results.close()


    def get_cls(pt, sign, bins, isnull=None):
        c = np.digitize(pt, bins)
        c = np.where(sign > 0, 2 * c, 2 * c - 1)
        c = np.where(c == -1, 0, c)
        if isnull is not None:
            c = np.where(isnull, 0, c)
        return c.astype(np.int32)


    def mkhist2d(xs, ys, xsz, ysz, mask=None):
        """
        Make 2D histogram from two integers arrays.
        """
        hs = np.zeros((xsz, ysz))
        for v in range(xsz):
            if mask is None:
                m = xs == v
            else:
                m = np.logical_and(mask, xs == v)
            s = np.sum(np.eye(ysz)[ys[m]], axis=0)
            hs[v] = s
        return hs
 

    def mkcurve(hist):
        s = np.cumsum(hist[:,::-1], axis=1)[:,::-1]
        for i in range(s.shape[0]):
            if s[i][0] > 0:
                s[i] = s[i] / s[i][0]
        return s.T


    def generate_histograms(self, hs_list):
        self.histograms = dict()
        for k, kx, ky, km in hs_list:
            xs, xn, xr, xl = self.data[kx]
            ys, yn, yr, yl = self.data[ky]
            m = self.masks[km]
            hs = OMTFStatistics.mkhist2d(xs, ys, xn, yn, m)
            self.histograms[k] = hs, (xn, xr, xl), (yn, yr, yl) 

    
    def generate_curves(self, hist_keys):
        self.curves = dict()
        for kh in hist_keys:
            hs, xd, yd = self.histograms[kh]
            curve = OMTFStatistics.mkcurve(hs)
            self.curves[kh] = curve, yd, xd

    def calculate_accuracy(self, hist_keys):
        self.accuracies = dict()
        for kh in hist_keys:
            hs, _, _ = self.histograms[kh]
            assert hs.shape[0] == hs.shape[1], "Not square matrix!"
            self.accuracies[kh] = hs.trace() / hs.sum()


    def save(self, path, summary=True):
        """
        Save statistics data as `.npz` file.
        """
        l = [
            TEST_STATISTICS_FIELDS.HISTOGRAMS,
            TEST_STATISTICS_FIELDS.CURVES,
            TEST_STATISTICS_FIELDS.ACCURACIES,
        ]
        f = [
            self.histograms,
            self.curves,
            self.accuracies
        ]
        data = {FILE_TYPES.TEST_STATISTICS: dict(zip(l,f))}
        np.savez_compressed(path, **data) 
        if summary:
            dict_to_json(path + '_summary.json', 
                {TEST_STATISTICS_FIELDS.ACCURACIES: self.accuracies})

