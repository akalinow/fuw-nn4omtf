"""
    Copyright (C) 2018 Jacek Łysiak
    MIT license

    It could be tougt to install ROOT and related python packages.
    I've used docker image to solve this problem.
"""

import ROOT
from root_numpy import root2array

import os
import sys
import re
import time
import numpy as np


def root_to_numpy(data):
    """ 
    Takes ROOT dict and extract:
        - hits_18x2, - filtered hits tensor,
        - hits_18x14 - full hits tensors, 
        - production data,
        - omtf data.

    ROOT data structure per simulation/event:
        - Hits tensor [18, 14]
        - Moun prodution data: {MuonPt, MuonPhi, MuonEta, MuonCharge}
        - OMTF simulation data: {omtfCharge, omtfPt, omtfEta, omtfQuality, omtfHits, omtfRefLayer}

    @param: data ROOT dictionary
    return: np.arrays of production data, OMTF data, hits 18x14, hits 18x2
    """

    omtf = []
    prod = []
    hits14 = []
    hits2 = []
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
        omtf.append(temp[4:])
    return np.array(prod), np.array(omtf), np.array(hits14), np.array(hits2)


def load_root_dict(path):
    """
    Loads ROOT's 'omtfPatternMaker/OMTFHitsTree' dictionary file.
    """
    return root2array(path, 'omtfPatternMaker/OMTFHitsTree')


# ---

filename = 'OMTFHitsData.root'
try:
	path = sys.argv[1]
	dest = sys.argv[2]
except Exception:
	print("Give dataset root path and dest")
	sys.exit(1)

print("Dataset root path: %s" % path)
print("Looking for data...")
r = re.compile(r"(?P<name>[A-Za-z]+)_(?P<val>[0-9]+)_(?P<sign>[pm])")

i = 1
l = len(os.listdir(path))
start_time = time.time()
for el in os.listdir(path):
	last = time.time()
	m = r.match(el)
	d = m.groupdict()
	print("[%3d / %3d]> " % (i, l), d['name'], d['val'], d['sign'])
	arr = root2array(os.path.join(path,el,filename), 'omtfPatternMaker/OMTFHitsTree')
	prod, omtf, h14, h2 = root_to_numpy(arr)
	np.savez_compressed(file=os.path.join(dest,"%s-%s-%s" % (d['name'].lower(), d['val'], d['sign'])), name=d['name'], val=int(d['val']), sign=d['sign'], prod=prod, omtf=omtf, hits14=h14, hits2=h2)
	print("Done!")
	print("Last: %.3fs Elapsed: %.3fs" % (time.time() - last, time.time() - start_time))
	i += 1

