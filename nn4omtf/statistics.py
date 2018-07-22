# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    OMTF test statistics generator
"""

import numpy as np

from .const_dataset import DATASET_TYPES, DATASET_FIELDS
from .const_model import MODEL_RESULTS  
from .const_stats import TEST_STATISTICS_FIELDS
from .const_files import FILE_TYPES


class OMTFStatistics:

    def __init__(self, path_ds_test, path_results):
        self.file_dataset = np.load(path_ds_test)
        self.file_results = np.load(path_results)
        
        self.dataset = self.file_dataset[DATASET_TYPES.TEST].item()
       
        self.note = self.file_results[MODEL_RESULTS.NOTE]
        self.logits = self.file_results[MODEL_RESULTS.RESULTS]
        self.pt_bins = self.file_results[MODEL_RESULTS.PT_BINS]

        self.events_total = self.logits.shape[0]
        self.pt_codes = self.dataset[DATASET_FIELDS.PT_CODE]
        self.pt_codes_list = np.unique(self.pt_codes)
        self.is_null = self.dataset[DATASET_FIELDS.IS_NULL]
        self.sign = self.dataset[DATASET_FIELDS.SIGN]
        self.pt_vals = self.dataset[DATASET_FIELDS.PT_VAL]

        # OMTF data
        self.omtf_pt = self.dataset[DATASET_FIELDS.OMTF_PT]
        self.omtf_sign = self.dataset[DATASET_FIELDS.OMTF_SIGN]
        self.omtf_quality = self.dataset[DATASET_FIELDS.OMTF_QUALITY]

        # Normalization masks
        # Event mask - OMTF algorithm had good enough data quality
        self.mask_omtf_quality_12 = self.omtf_quality == 12.
        # Event mask - OMTF tried to guess pt value
        self.mask_omtf_pt999 = self.omtf_pt >= 0
        self.mask_omtf_q12_pt999 = np.logical_and(self.mask_omtf_pt999, 
                self.mask_omtf_quality_12)

        # Ground truths
        #  NO NULLs - real tought life
        self.pt_cls_no_null = self._get_pt_class(self.pt_vals, 
                self.sign)
        # WITH NULLs - training conditions
        self.pt_cls_with_null = self._get_pt_class(self.pt_vals, 
                self.sign, self.is_null)

        # OMTF algorithm's classified answers         
        self.omtf_pt_cls = self._get_pt_class(self.omtf_pt, 
                self.omtf_sign)

        # NN answers
        self.nn_pt_cls = np.argmax(self.logits, axis=1)
        # Softmax -> probability distribution
        self.nn_prob = self._softmax()

        # Calculate statistics
        self.statistics = dict()
        self.descriptions = dict()
        self._calc_dataset_summary()
        self._calc_acc()
        self._calc_pd()
        self._calc_hists()
        self._calc_curves()


    def save(self, path):
        """
        Save statistics data as `.npz` file.
        """
        l = [
            TEST_STATISTICS_FIELDS.PT_BINS,
            TEST_STATISTICS_FIELDS.SUMMARY,
            TEST_STATISTICS_FIELDS.PDIST,
            TEST_STATISTICS_FIELDS.HISTOGRAMS,
            TEST_STATISTICS_FIELDS.CURVES,
            TEST_STATISTICS_FIELDS.PT_CODES
        ]
        f = [
            self.pt_bins,
            self.statistics,
            self.pd,
            self.histograms,
            self.curves,
            self.pt_codes_list
        ]
        data = {FILE_TYPES.TEST_STATISTICS: dict(zip(l,f))}
        np.savez_compressed(path, **data)        


    def _calc_pd(self):
        """
        Calculate NN output probability distributions for both cases:
          - real pt class
          - training conditions with null class
        Normalization - per original pt class.
        Same things are generated for OMTF algorithm but PD is approxiamted
        by summing up OMTF single output and normalization per original
        pt class.

        Format:
         <distr array>[original pl class][nn pt class]

        """
        sz = self.nn_prob.shape[1]
        pd_no_null = np.zeros([sz, sz])
        pd_with_null = np.zeros([sz, sz])
        omtf_pd_no_null = np.zeros([sz, sz])
        omtf_pd_with_null = np.zeros([sz, sz])
        for idx in range(sz):
            mask_nn = self.pt_cls_no_null == idx
            s_nn = np.sum(mask_nn)
            mask_wn = self.pt_cls_with_null == idx
            s_wn = np.sum(mask_wn)
            if s_nn > 0:
                pd_no_null[idx] = np.sum(self.nn_prob[mask_nn], axis=0) / s_nn
                omtf_pd_no_null[idx] = np.sum(np.eye(sz)[self.omtf_pt_cls[mask_nn]], axis=0) / s_nn
            if s_wn > 0:
                pd_with_null[idx] = np.sum(self.nn_prob[mask_wn], axis=0) / s_wn
                omtf_pd_with_null[idx] = np.sum(np.eye(sz)[self.omtf_pt_cls[mask_wn]], axis=0) / s_wn
        
        names = ['no_null', 'with_null']
        names = ['nn_' + x for x in names] + ['omtf_' + x for x in names]
        pds = [pd_no_null, pd_with_null,omtf_pd_no_null, omtf_pd_with_null]
        self.pd = dict(zip(names, pds))
         

    def _calc_hists(self):
        ptc_sz = int(self.pt_codes_list.max())
        cls_sz = self.nn_prob.shape[1]

        norm_masks = [
            np.ones(self.events_total).astype(np.bool),
            self.is_null == False,
            self.mask_omtf_quality_12,
            self.mask_omtf_pt999,
            self.mask_omtf_q12_pt999
        ] * 2
        descr = [
            'All events',
            'Not NULL events',
            'OMTF quality == 12',
            'OMTF pt != -999',
            'OMTF q. == 12 & pt != -999'
        ]
        descr = ['NN ' + x for x in descr] + ['OMTF ' + x for x in descr]

        fields = [
            'hists_all',
            'hists_not_null',
            'hists_q12',
            'hists_pt999',
            'hists_q12_pt999'
        ]
        fields = [x + '_nn' for x in fields] + [x + '_omtf' for x in fields]
        srcs = [self.nn_pt_cls] * 5 + [self.omtf_pt_cls] * 5
        hists = [np.zeros((ptc_sz + 1, cls_sz)) for _ in fields]

        for mask, src, hist in zip(norm_masks, srcs, hists):
            for ptc in self.pt_codes_list:
                mask_ptc = self.pt_codes == ptc
                m = np.logical_and(mask_ptc, mask)
                hist[int(ptc)] = np.sum(np.eye(cls_sz)[src[m]], axis=0)

        hists_pt = [np.zeros((ptc_sz + 1, cls_sz // 2 + 1 )) for _ in hists]
        for h_pt, h in zip(hists_pt, hists):
            h_pt[:,0] = h[:,0]
            h_pt[:,1:] = h[:,1::2] + h[:,2::2]
        self.histograms = dict(zip(fields, hists_pt))
        self.histograms_l = dict(zip(fields, descr))


    def _calc_curves(self):
        self.curves = dict()
        for field, hist in self.histograms.items():
            s = np.cumsum(hist[:,::-1], axis=1)[:,::-1]
            for i in range(1,s.shape[0]):
                if s[i][0] > 0:
                    s[i] = s[i] / s[i][0]
            self.curves[field.replace('hists', 'curves')] = s[1:].T


    def _calc_dataset_summary(self):
        desc = [
            'Events in total',
            'Events GOOD %',
            'Events NULL %',
            'OMTF quality == 12 %',
            'OMTF pt != -999. %',
            'OMTF q == 12 & pt != -999 %'
        ]
        fields = [
            'events_total',
            'events_good',
            'events_null',
            'events_omtf_q12',
            'events_omtf_pt999',
            'events_omtf_q12_pt999',
        ]
        is_null_p = np.sum(self.is_null) / self.events_total
        vals = [
            self.events_total,
            1 - is_null_p,
            is_null_p,
            np.sum(self.mask_omtf_quality_12) / self.events_total,
            np.sum(self.mask_omtf_pt999) / self.events_total,
            np.sum(np.logical_and(self.mask_omtf_pt999, 
                self.mask_omtf_quality_12)) / self.events_total,
        ]
        vals = np.array(vals) * 100
        self._update(fields, vals, desc)


    def _calc_acc(self):
        desc = [
            'NN real accuracy',
            'NN training conditions accuracy',
            'OTMF real accuracy',
            'OMTF training conditions accuracy'
        ]
        desc = desc + [x + ' (OMTF quality == 12)' for x in desc] +\
                [x + ' (OMTF pt != -999.)' for x in desc]
        fields = [
            'nn_acc_no_null',
            'nn_acc_with_null',
            'omtf_acc_no_null',
            'omtf_acc_with_null'
        ]
        fields = fields + [x + '_q12' for x in fields] +\
                [x + '_g' for x in fields]
        l = [
            self.nn_pt_cls == self.pt_cls_no_null,
            self.nn_pt_cls == self.pt_cls_with_null,
            self.omtf_pt_cls == self.pt_cls_no_null,
            self.omtf_pt_cls == self.pt_cls_with_null
        ]
        l_norm_q12 = [x[self.mask_omtf_quality_12] for x in l]
        l_norm_g = [x[self.mask_omtf_pt999] for x in l] 
        l = l + l_norm_q12 + l_norm_g
        l = [np.sum(v) / v.shape[0] for v in l]
        self._update(fields, l, desc)


    def _stat_str(self):
        s = ''
        keys = list(self.statistics.keys())
        keys.sort()
        for k in keys:
            s += '> %s: ' % self.descriptions[k]
            s += '%.3f\n' % self.statistics[k]
        return s


    def _update(self, labels, values, desc):
        self.statistics.update(zip(labels, values))
        self.descriptions.update(zip(labels, desc))


    def _get_pt_class(self, pv, sign, is_null=None):
        """
        Helper function calculating correct muon class.
        """
        c = np.digitize(pv, self.pt_bins)
        c = np.where(sign > 0, 2 * c, 2 * c - 1)
        c = np.where(c == -1, 0, c)
        if is_null is not None:
            c = np.where(is_null, 0, c)
        return c.astype(np.int32)


    def __str__(self):
        """
        Return short summary of calculated statistics
        """
        pt_bins_str = [str(x) for x in self.pt_bins]
        s = ">>>>> OMTF NN Statistics\n"
        s += "> Note: \n%s\n" % self.note
        s += "> pt bins: " + ",".join(pt_bins_str)
        s += '\n'

        s += self._stat_str()
        return s


    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        self.file_dataset.close()
        self.file_results.close()


    def _softmax(self):
        assert len(self.logits.shape) == 2, "Wrong dimensions!"
        m = np.max(self.logits, axis=1)
        m = m[:, np.newaxis]
        e = np.exp(self.logits - m)
        s = np.sum(e, axis=1)
        s = s[:, np.newaxis]
        return e / s
