# -*- coding: utf-8 -*-
"""
    Copyright(C) 2018 Jacek Łysiak
    MIT License

    Constants used in nn4omtf.
"""

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


PT_CODES_DATA = [
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
    [28, 90.00, 100.00, 95.00]
    #   Codes not used in experiments  
    #   [29, 100.00, 120.00, 110.00],
    #   [31, 140.00, 1000.00, 570.00],
]

PT_CODES_IDX = [c[0] for c in PT_CODES_DATA]
PT_CODES_RANGE_MIN = [c[1] for c in PT_CODES_DATA]
PT_CODES_RANGE_MAX = [c[2] for c in PT_CODES_DATA]
PT_CODES_RANGE_MID = [c[3] for c in PT_CODES_DATA]

