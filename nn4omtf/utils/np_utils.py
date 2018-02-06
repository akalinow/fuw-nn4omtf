"""
    Copyright (C) 2018 Jacek Łysiak
    MIT license

    Loading and saving data in *.npz format.
    It looks like this module is not very useful but
    after np.load() returned object is not pure dict with np.arrays.
"""

import numpy as np


__all__ = ['save_dict_as_npz', 'load_dict_from_npz']


def save_dict_as_npz(path, **kw):
    """Save dict passed as kwargs into *.npz file.
    This dict will be reflected in archive.
    Same labels will appear in dict afted loading file into memory.
    Args:
        path: path to file
        kw: key words with data to save
    """
    np.savez_compressed(file=path, **kw)


def load_dict_from_npz(path):
    """Loads dictionary saved in *.npz file.
    Args:
        filename: path to *.npz file
    Returns:
        Dictionary loaded from *.npz file.
    """
    data = np.load(path)
    data = dict(data)
    return { k: v if v.size != 1 else v.item() for k, v in data.items() }
    
