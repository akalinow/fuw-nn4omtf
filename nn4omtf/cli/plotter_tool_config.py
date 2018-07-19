# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    OMTF plotter command line tool configuration

"""


class ACTION:
    SHOW = 'show'
    PLOT = 'plot'
    DEFAULT_CONF = 'config'


parser_config = {
    ACTION.SHOW : {
        'help': "Preview result files or list available results",
        'opts': [],
        'pos': [
            ('path', {'help': "Result npz file path or model directory"}),
        ]
    },
    ACTION.PLOT: {
        'help': "Plot data from `*.npz` file",
        'opts': [
            ('config', {'help': "Override default plotter configuration " + 
                "using data from custom JSON file", 'metavar': 'PATH'}),
            ('outdir', {'help': "Save results in given directory", 'metavar': 'PATH'}),
        ],
        'pos': [
            ('path', {'help': "Result npz file path or model directory"}),
        ]
    },
    ACTION.DEFAULT_CONF : {
        'help': "Get plotter config file",
        'opts': [
            ('out', {'help': "Save config file as.", 'metavar': 'PATH'}),
        ],
        'pos': [
        ]
    },
}
