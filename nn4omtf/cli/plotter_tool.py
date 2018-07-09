# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    Unified OMTF data plotter.
"""

from nn4omtf.cli.plotter_tool_config import ACTION, parser_config
from nn4omtf.cli.utils import create_parser


class OMTFPlotterTool:

    def __init__(self):
        self.parser = create_parser(parser_config, desc="OMTF data plotter")

    def run(self):
        pass


