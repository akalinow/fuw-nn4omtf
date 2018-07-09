# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    OMTF trainer tool
"""


from nn4omtf import OMTFModel
from nn4omtf.cli.runner_tool_config import ACTION, parser_config
from nn4omtf.cli.utils import create_parser


class OMTFRunnerTool:

    def __init__(self):
        self.parser = create_parser(parser_config, desc="OMTF NN trainer")

    def run(self):
        FLAGS = self.parser.parse_args()
        action = FLAGS.action

        if action == ACTION.SHOW:
            self._show(FLAGS)
        elif action == ACTION.MODEL:
            self._create(FLAGS)
        elif action == ACTION.TRAIN:
            self._train(FLAGS)
        elif action == ACTION.TEST:
            self._test(FLAGS)
        else:
            self.parser.print_help()
    
    def _show(self, opts):
        pass

    
    def _create(self, opts):
        pass


    def _train(self, opts):
        pass


    def _test(self, opts):
        pass





