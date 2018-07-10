# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    Just some utils...
"""

import json


def to_sec(timestr):
    """
    Converts `hh:mm:ss` to duration in seconds.
    """
    ts = [i for i in map(int, timestr.split(":"))]
    l = len(ts)
    k = 1
    sec = 0
    ts.reverse()
    for v in ts:
        sec += v * k
        k *= 60
    return sec


def dict_to_json(path, d):
    """
    Create JSON from dictionary.
    """
    s = json.dumps(d, sort_keys=True, indent=4)
    with open(path, 'w') as f:
        f.write(s)


def dict_to_json_string(d):
    return json.dumps(d, sort_keys=True, indent=4)


def json_to_dict(path):
    """
    Create dictionary from JSON.
    """
    with open(path, 'r') as f:
        r = f.read()
    return json.loads(r)
