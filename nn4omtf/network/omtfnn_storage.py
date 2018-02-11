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
from nn4omtf.network import OMTFNN

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
        shutil.rmtree(cont, ignore_errors=True)
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


    def add_new_network_with_builder_file(self, name, builder_file):
        """Add new network into storage. Create model using builder file.
        Args:
            name: model name
            builder_file: builder file path
        """
        assert name not in self.nets, "Network '{}' already in storage!".format(name)
        dest = os.path.join(self.path, OMTFNNStorage.CONST.NETS_CONTAINER)
        net = OMTFNN.create_with_builder_file(name, dest, builder_file)
        self.nets.append(name)


    def add_network(self, name, builder_fn):
        """Create new network 
        Args:
            name: model name
            builder_fn: builder function
        """
        assert name not in self.nets, "Network '{}' already in storage!".format(name)
        dest = os.path.join(self.path, OMTFNNStorage.CONST.NETS_CONTAINER)
        OMTFNN.create_with_builder_file(name, dest, builder_file)
        self.nets.append(name)


    def add_existing_network(self, net):
        """Add network to storage.
        Args:
            net: OMTFNN object
        """
        assert net.name not in self.nets, "Network '{}' already in storage!".format(net.name)
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
        r += "\nNetworks summaries:\n"
        nets = self._load_by_names(self.nets)
        for net in nets:
            r += '\n'
            r += net.get_summary()
        return r


    def __str__(self):
        return self.get_summary()


