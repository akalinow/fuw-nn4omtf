# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT license

    Methods to load ROOT dicts and convert data to Numpy format.

    INFO:
    It could be tougt to install ROOT and related python packages.
    It might be easier to use Docker and download linux image with ROOT.

    This part of package is very specific. ROOT dict structure is defined by
    external program which generates data from OMTF simulation.
"""

import ROOT
from root_numpy import root2array

import os
import sys
import re
import time
import numpy as np

from nn4omtf.dataset.const import NPZ_FIELDS


def load_root_dict(path):
    """Loads ROOT's 'omtfPatternMaker/OMTFHitsTree' dictionary file.
    This method assumes that files and directory structure comes from @akalinow 
    script which generates simulation of OMTF algorithm.
    This method is to import these ROOT files from directories
    which names match pattern "SignleMu_[0-9]+_[pm]".
    ROOT file should be OMTFHitsData.root.

    Args:
        path: path to directory containing mentioned ROOT file
    Returns:
        (ROOT dict dataset, dict with dataset name, pt code, muon charge sign)
    """
    NAME = 'OMTFHitsData.root'
    fpath = os.path.join(path, NAME)
    basename = os.path.basename(path)
    r = re.compile(r"(?P<name>[A-Za-z]+)_(?P<val>[0-9]+)_(?P<sign>[pm])")
    m = r.match(basename)
    if m is None:
        raise ValueError("Provided path does not match pattern: name_[0-9]+_[pm]")
    d = m.groupdict()
    d['val'] = int(d['val'])
    return root2array(fpath, 'omtfPatternMaker/OMTFHitsTree'), d


def root_to_numpy(name, sign, ptcode, data):
    """Method to extract data from ROOT dict and pack it
    along with some info from directory name
    Extracted data:
        - hits_18x2, - reduced hits tensor,
        - hits_18x14 - full hits tensors, 
        - production data,
        - omtf data.

    ROOT data structure per simulation/event:
        - Hits tensor [18, 14]
        - Moun prodution data: {MuonPt, MuonPhi, MuonEta, MuonCharge}
        - OMTF simulation data: {omtfCharge, omtfPt, omtfEta, omtfQuality, omtfHits, omtfRefLayer}

    Note: 
        ROOT dicts are not imported as ndarray. They cannot be sliced and iterated using numpy tools.
        Single dict conversion can last a few minutes.

    Args: 
        name: extracted name from parent directory
        sign: muon charge's sign
        ptcode: pt code extracted from parent directory name
        data: ROOT dictionary dataset
    Returns:
        dict with data arrays, np.arrays of production data, OMTF data, hits 18x14, hits 18x2
    """
    omtf = []
    prod = []
    hits14 = []
    hits2 = []
    pt_max = data[0][1]
    pt_min = pt_max
    def get_hits(hits):
        r18x14 = []
        r18x2 = []
        for l in hits:
            r18x14.append(l)
            l = l[l != 5400]
            if l.size == 2:
                r18x2.append(l)
            elif l.size == 1:
                r18x2.append(np.array([l[0], 5400]))
            else:
                r18x2.append(np.ones(2) * 5400)
        return r18x14, r18x2
    
    for e in data:
        h14, h2 = get_hits(e[0])
        hits2.append(h2)
        hits14.append(h14)
        temp = [e[i] for i in range(1, 11)]
        prod.append(temp[0:4])
        pt_max = max(pt_max, temp[0])
        pt_min = min(pt_min, temp[0])
        omtf.append(temp[4:])

    ev_n = len(prod)
    res = dict()
    res[NPZ_FIELDS.HITS_FULL] = np.array(hits14)
    res[NPZ_FIELDS.HITS_REDUCED] = np.array(hits2)
    res[NPZ_FIELDS.PROD] = np.array(prod)
    res[NPZ_FIELDS.OMTF] = np.array(omtf)
    res[NPZ_FIELDS.PT_CODE] = ptcode
    res[NPZ_FIELDS.PT_MAX] = pt_max
    res[NPZ_FIELDS.PT_MIN] = pt_min
    res[NPZ_FIELDS.NAME] = name
    res[NPZ_FIELDS.SIGN] = sign
    res[NPZ_FIELDS.EV_N] = ev_n
    return res


