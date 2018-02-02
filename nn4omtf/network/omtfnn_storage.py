# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    Neural network storage helper class.
    Let store all related models in one place...
"""

import os

__all__ = ['OMTFNNStorage']

class OMTFNNStorage:

    class CONST:
        DESC_FILE_NAME = ".omtfnn_storage"


    def __init__(self, path=None):
        if path is not None:
            self.load(path)
        else:
            self.path = path
            self.nnlist = [] # Empty list of networks


    def load(self, path): 
        """Read list of networks in this set."""
        self.path = path
        path = os.path.join(path, self.DESC_FILE_NAME)
        if not os.path.exists(path):
            raise FileNotFoundError("Not found: %s" % path)
        with open(path) as f:
            for line in f:
                self.nnlist.append(f.readline())


    def save(self, path):
        """Save neural networks."""
        self.path = path
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, self.DESC_FILE_NAME)
        with open(path, 'w') as f:
            for el in self.nnlist:
                f.write(el + '\n')

    def is_valid_object(path):
        """Check whether path points to valid object.
        """
        path = os.path.join(path, OMTFNNStorage.CONST.DESC_FILE_NAME)
        try:
            OMTFNNStorage._parse_obj_file(path)
        except Exception:
            return False
        return True


    def _parse_obj_file(path):
        """Parse stroage object file.
        """
        with open(path, 'r') as f:
            paths = f.read().split("\n")
        return {"paths": paths}


