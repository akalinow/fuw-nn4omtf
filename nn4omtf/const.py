"""
    Copyright(C) 2018 Jacek Łysiak

    Constants used in nn4omtf.
"""

PT_CODE_RANGE = 31

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
