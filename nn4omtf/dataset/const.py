# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    Dataset package consts.
"""


__all__ = ['HITS_TYPE', 'NPZ_FIELDS']


class HITS_TYPE:
    FULL = "FULL"
    FULL_SHAPE = [18, 14]

    REDUCED = "REDUCED"
    REDUCED_SHAPE = [18, 2]


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

