"""
    Copyright(C) 2018 Jacek Łysiak

    Constants used in nn4omtf.
"""

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
    "OMTF_SGN_CLASS"    # Classified OMTF charge sign
]

SIGN_OUT_SZ = 3

