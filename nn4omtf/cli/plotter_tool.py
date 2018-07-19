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
from nn4omtf import OMTFPlotter
from nn4omtf.utils import dict_to_json, json_to_dict
from nn4omtf.plotters import PLOTTER_DEFAULTS


class OMTFPlotterTool(OMTFTool):

    def __init__(self):
        handlers = [
            (ACTION.SHOW, OMTFPlotterTool._show),
            (ACTION.DEFAULT_CONF, OMTFPlotterTool._default_conf),
            (ACTION.PLOT, OMTFPlotterTool._plot),
        ]
        super().__init__(parser_config, "OMTF data plotter", handlers)


    def _show(opts):
        with OMTFPlotter(opts.path) as p:
            ls = p.get_plottables()
        if not ls:
            print("No plottable elements!")
        else:
            print("Plottable elements of `%s`:" % opts.path)
            for s in ls:
                print('- ' + s)


    def _default_conf(opts):
        fn = 'plotter_config.json'
        if opts.out is not None:
            fn = opts.out
        dict_to_json(fn, PLOTTER_DEFAULTS) 


    def _plot(opts):
        conf = dict()
        if opts.config is not None:
            conf = json_to_dict(opts.config)
        with OMTFPlotter(opts.path, outdir=opts.outdir, **conf) as p:
            for name in p.get_plottables():
                p.plot(name)
                p.save(name)
                p.close_figures()


