# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    Unified OMTF data plotter.
"""

import os
from .tool import OMTFTool
from .plotter_tool_config import ACTION, parser_config
from .utils import create_parser


class OMTFPlotterTool(OMTFTool):

    def __init__(self):
        handlers = [
            (ACTION.SHOW, OMTFPlotterTool._show),
            (ACTION.DEFAULT_CONF, OMTFPlotterTool._default_conf),
            (ACTION.PLOT, OMTFPlotterTool._plot),
        ]
        super().__init__(parser_config, "OMTF data plotter", handlers)


    def _show(opts):
        pass


    def _default_conf(opts):
        pass


    def _plot(opts):
        pass

