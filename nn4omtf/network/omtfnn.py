# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    Neural network wrapper.
"""

from .dataset import OMTFDataset
import tensorflow as tf

class OMTFNN:
    """OMTF neural network wrapper.
    Fields:
        - x: TensorFlow input tensor
        - y: TensorFlow output tensor
        - pt_class: pt class bins edges
        - in_type: type of input shape for hits tensor
        - name: netowrk name
        - desc: network description
    """
    name = ""
    desc = ""


    def __init__(self, pt_class, x, y, in_type=OMTFDataset.HITS_TYPE_REDUCED):
        """Create neural network package.
        Args:
            - pt_class: list of pt classes bounds
            - x: input tensor placeholder
            - y: output tensor placeholder
            - in_type(optional, default: reduced): type of input hits tensor
        """
        self.pt_class = pt_class
        self.in_type=in_type
        self.set_network(x=x, y=y)


    def set_name(self, name):
        self.name = name


    def set_description(self, desc):
        self.desc = desc


    def _set_network(self, x, y):
        """Setup user defined neural network.
        User can create neural network and pass here input and output placeholders.
        Dimensions of input tensor must fit to:
            - [-, 18, 14] for FULL hits array
            - [-, 18, 2] for REDUCED hits array
        Dimensions of output must fit to [-, N] where N-1 is length 
        of categories bounds list.

        Args:
            - x: input tensor placeholder
            - y: output tensor placeholder
        """
        xs_req = (18, 2)
        if self.in_type == OMTFDataset.HITS_TYPE_FULL:
            xs_req = (18, 14)
        xs = x.shape
        ys = y.shape
        if xs[1] != xs_req[0] or xs[2] != xs_req[1]:
            raise ValueError("Network input shape doesn't fit to [-, %d, %d]" % xs_req)
        if ys[1] != self.pt_class + 1:
            raise ValueError("Network output shape doesn't fit to [-, %d]" % (self.pt_class + 1)) 
        self.x = x
        self.y = y


    def save(self, path):
        """@todo Save model."""
        pass


    def load(self, path):
        """@todo Load trained model."""
        pass


