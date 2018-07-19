# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    Base CLI tool class.
"""

from .utils import create_parser

class OMTFTool:


    def __init__(self, parser_config, desc, actions):
        """
        OMTF CLI tools base class.
        Implements very simple CLI flow control.
        Args:
            parser_config: parser configuration dictionary 
            desc: tool description
            actions: list of paris (action, handler),
                handler has signature (parsed args namespace) -> void
        """
        self.parser = create_parser(parser_config, desc=desc)
        self.actions = dict(actions)


    def run(self):
        FLAGS = self.parser.parse_args()
        action = FLAGS.action
        try:
            handler = self.actions[action]
        except:
            self.parser.print_help()
            exit(0)
        handler(FLAGS)

