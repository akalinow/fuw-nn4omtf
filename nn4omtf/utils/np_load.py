"""
    Copyright (C) 2018 Jacek Łysiak
    MIT license

    Loading and saving data in *.npz format.

    In *.npz files data is stored as dictionary with (k, v), where keys are as follows:
    - pt_code: Muon momentum code
    - sign: Muon charge
    - prod: tensor of event's production data
    - omtf: tensor of OMTF simulated reaction on each generated Muon (event)
    - hits: full registered hits data for reach generated Muon
    - hits2: filtered hits array, without 5400 'nothing' values

"""

import numpy as np

__all__ = ['save_npz', 'load_npz']


def save_npz(filename, pt_code, sign, data_arrays):
    """
    Save data stored in np.arrays along with momentum code and mouon sign in *.npz file.
    """
    np.savez_compressed(
            file=filename, 
            pt_code=pt_code, 
            sign=sign, 
            prod=data_arrays[0], 
            omtf=data_arrays[1], 
            hits=data_arrays[2], 
            hits2=data_arrays[3])


def load_npz(filename):
    """Load numpy compressed file."""
    pass


