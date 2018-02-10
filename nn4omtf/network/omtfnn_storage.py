# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    Neural network storage helper class.
    Store all related models in one place...
"""

import os
import pickle
import re
import shutil


__all__ = ['OMTFNNStorage']


class OMTFNNStorage:
    class CONST:
        DESC_FILE_NAME = ".omtfnn_storage"
        NETS_CONTAINER = "nets"


    def __init__(self, name, path):
        """Create storage for NN objects.
        Args:
            name: container name
            path: container location
        """
        self.name = name
        self.path = os.path.join(path, name)
        obj_file = os.path.join(self.path, OMTFNNStorage.CONST.DESC_FILE_NAME)
        assert not os.path.exists(obj_file), "Storage already exists!"
        cont = os.path.join(self.path, OMTFNNStorage.CONST.NETS_CONTAINER)
        shutil.rmtree(cont)
        os.makedirs(cont)
        self.nets = []


    def load(path): 
        """Load storage object from disk.
        Args:
            path: object location
        Returns:
            Loaded OMTFNNStorage
        """
        obj_file = os.path.join(path, OMTFNNStorage.CONST.DESC_FILE_NAME)
        with open(obj_file, 'rb') as f:
            obj = pickle.load(f)
        obj.path = path
        return obj


    def save(self):
        """Save storage."""
        obj_file = os.path.join(self.path, OMTFNNStorage.CONST.DESC_FILE_NAME)
        with open(obj_file, 'wb') as f:
            pickle.dump(self, f)


    def is_valid_object(path):
        """Check whether path points to valid object."""
        try:
            OMTFNNStorage.load(path)
        except Exception:
            return False
        return True


    def add_network(self, net):
        """Add network to storage.
        Args:
            net: OMTFNN object
        """
        self.nets.append(net.name)
        dest = os.path.join(self.path, OMTFNNStorage.CONST.NETS_CONTAINER)
        net.move(dest)
   

    def get_by_name(self, reg):
        """Get matching networks.
        Args:
            reg: regexp matched with name
        """
        matched = self._match_names(reg)
        return self._load_by_names(matched)


    def remove_by_name(self, reg):
        """Delete nets by matching name.
        Args:
            reg: regexp matched with name
        """
        matched = self._match_names(reg)
        for m in matched:
            self.nets.remove(m)
            path = os.path.join(self.path, OMTFNNStorage.CONST.NETS_CONTAINER, m)
            shutil.rmtree(path)


    def _match_names(self, reg):
        """Get nets names by matching to regexp.
        Args:
            reg: regexp matched with name
        """
        matcher = re.compile(reg)
        matched = []
        for net in self.nets:
            if matcher.match(net):
                matched.append(net)
        return matched


    def _get_net_path(self, name):
        return os.path.join(self.path, OMTFNNStorage.CONST.NETS_CONTAINER, name)


    def _load_by_name(self, name):
        path = self._get_net_path(name)
        return OMTFNN.load(path)
    

    def _load_by_names(self, names):
        l = []
        for name in names:
            l.append(self._load_by_name(name))
        return l


    def get_summary(self):
        """Returns summary string."""
        r = "Storage name: " + self.name
        r += "\nList of networks:\n"
        nets = self._load_by_names(self.nets)
        for net in nets:
            r += '\n'
            r += net.get_summary()
        return r


    def __str__(self):
        return self.get_summary()


