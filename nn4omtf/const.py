"""
    Copyright(C) 2018 Jacek Łysiak

    Constants used in nn4omtf.
"""

PT_CODE_RANGE = 31
PT_CODES = [
    [1, 1.00, 1.50, 1.25],
    [2, 1.50, 2.00, 1.75],
    [3, 2.00, 2.50, 2.25],
    [4, 2.50, 3.00, 2.75],
    [5, 3.00, 3.50, 3.25],
    [6, 3.50, 4.00, 3.75],
    [7, 4.00, 4.50, 4.25],
    [8, 4.50, 5.00, 4.75],
    [9, 5.00, 6.00, 5.50],
    [10, 6.00, 7.00, 6.50],
    [11, 7.00, 8.00, 7.50],
    [12, 8.00, 10.00, 9.00],
    [13, 10.00, 12.00, 11.00],
    [14, 12.00, 14.00, 13.00],
    [15, 14.00, 16.00, 15.00],
    [16, 16.00, 18.00, 17.00],
    [17, 18.00, 20.00, 19.00],
    [18, 20.00, 25.00, 22.50],
    [19, 25.00, 30.00, 27.50],
    [20, 30.00, 35.00, 32.50],
    [21, 35.00, 40.00, 37.50],
    [22, 40.00, 45.00, 42.50],
    [23, 45.00, 50.00, 47.50],
    [24, 50.00, 60.00, 55.00],
    [25, 60.00, 70.00, 65.00],
    [26, 70.00, 80.00, 75.00],
    [27, 80.00, 90.00, 85.00],
    [28, 90.00, 100.00, 95.00],
    [29, 100.00, 120.00, 110.00],
    [31, 140.00, 1000.00, 570.00],
]

class PIPE_MAPPING_TYPE:
    INTERLEAVE = 0
    FLAT_MAP = 1

class PHASE_NAME:
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"

# Defines names for extra data provided from dataset
# It's used by OMTFStatistics and OTMFInputPipe
PIPE_EXTRA_DATA_NAMES = [
    "PT_CODE",          # Production pt code
    "PT_VAL",           # Production real pt
    "PT_CLASS",         # Classified real pt using provided by user pt bins
    "PT_SGN_CLASS",     # Classified real muon charge
    "OMTF_PT_VAL",      # OMTF pt results
    "OMTF_PT_CLASS",    # Classified OMTF pt result (-1 <=> -999.0)
    "OMTF_SGN_CLASS",   # Classified OMTF charge sign
    "NO_SIGNAL"         # Empty hits array
]

class HITS_TYPE:
    FULL = "FULL"
    FULL_SHAPE = [18, 14]
    FULL_PH_SHAPE = [None, 18, 14]
    REDUCED = "REDUCED"
    REDUCED_SHAPE = [18, 2]
    REDUCED_PH_SHAPE = [None, 18, 2]

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
    PT_CODE = "pt_code"
    PT_MIN = "pt_min"
    PT_MAX = "pt_max"
    EV_N = "ev_n"
    NAME = "name"
    SIGN = "sign"

SIGN_OUT_SZ = 3

NN_HOLDERS_NAMES = [
    "HITS",
    "SIGN_LABEL",
    "PT_LABEL"
]

# Column names with NN outputs
NN_CNAMES = [
    "NN_PT_CLASS",      # NN output data
    "NN_SGN_CLASS",
    "NN_PT_LOGITS",
    "NN_SGN_LOGITS"
]

# This list of labels 
CNAMES = PIPE_EXTRA_DATA_NAMES + NN_CNAMES


class PLT_DATA_TYPE:
    PROB_DIST = 'prob_disti'
    TRAIN_LOG = 'train_log'
    HIST_CODE_BIN = 'hits_code_bin'
    ANSWERS = 'answers'
