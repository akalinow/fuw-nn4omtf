# -*- coding: utf-8 -*-
"""
    Copyright(C) 2018 Jacek Łysiak
    MIT License

    Dataset constants used in nn4omtf.
"""


class NPZ_DATASET:
    HITS_REDUCED = 'hits_reduced'
    PROD = 'prod'
    OMTF = 'omtf'

    PT_CODE = "pt_code"
    SIGN = "sign"


class DATASET_TYPES:
    TRAIN = 'TRAIN'
    VALID = 'VALID'
    TEST = 'TEST'


class HIST_TYPES:
    AVG = 'AVG'
    VALS = 'VALS'


class DATA_TYPES:
    ORIG = 'ORIG'
    TRANS = 'TRANS'


class HIST_SCOPES:
    TOTAL = 'TOTAL'
    CODE = 'CODE'


class ORD_TYPES:
    ORIG = 'ORIG'
    SHUF = 'SHUF'


class DATASET_FIELDS: 
    # Train/Valid/Test datasets
    HITS = 'HITS'
    PT_VAL = 'PT_VAL'
    SIGN = 'SIGN'
    IS_NULL = 'IS_NULL'
    # pt code can be used also to characterizing train dataset
    PT_CODE = 'PT_CODE' 

    # Test dataset only
    # Fields used only by statistics and ploter module!
    # There's no need to process them along with network input data!
    OMTF_PT = 'OMTF_PT'
    OMTF_SIGN = 'OMTF_SIGN'
    OMTF_QUALITY = 'OMTF_QUALITY'
    

class DSET_STAT_FIELDS: 
    TRAIN_EXAMPLES_ORDERING = 'TRAIN_EXAMPLES_ORDERING'
    # All pt codes accumulated in single histograms 
    HISTS_TOTAL_ORIG = 'HISTS_TOTAL_ORIG'
    HISTS_TOTAL_TRANS = 'HISTS_TOTAL_TRANS'
    # Lists of separate historgrams for each pt code
    HISTS_ORIG = 'HISTS_ORIG'
    HISTS_TRANS = 'HISTS_TRANS'
    MUON_SIGNATURES = 'MUON_SIGNATURES'
    MUON_SIGNATURES_DISTR = 'MUON_SIGNATURES_DISTR'
    HISTS_BINS = 'HISTS_BINS'
    TRANSFORM = 'TRANSFORM'
    TRESHOLD = 'TRESHOLD'


class NPZ_FIELDS:
    PROD = "prod"
    PROD_SHAPE = [4]
    PROD_IDX_PT = 0
    PROD_IDX_SIGN = 3
    OMTF_IDX_PT = 1
    OMTF_IDX_SIGN = 0
    OMTF = "omtf"
    OMTF_SHAPE = [6]
    HITS_FULL = "hits_full"
    HITS_REDUCED = "hits_reduced"
    PT_MIN = "pt_min"
    PT_MAX = "pt_max"
    EV_N = "ev_n"
    NAME = "name"


