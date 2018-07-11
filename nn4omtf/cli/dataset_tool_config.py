# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    OMTF dataset command line tool configuration
"""


class ACTION:
    SHOW = 'show'
    CONVERT = 'root2np'
    CREATE = 'create'


parser_config = {

    'root2np': {
        'help': "Convert ROOT datasets to Numpy arrays. ROOT environment requred!",
        'opts': [
            ("verbose", {'action': "store_true"})
        ],
        'pos': [
            ("dest", {'help': "Destination directory"}),
            ("data", {'nargs': "+", 'help': "Source directories with ROOT datasets."})
        ]
    },

    'create': {
        'help': "Create dataset from `*.npz` files",
        'opts': [
            ('outdir', {'help': "Output directory"}),
            ('train', {'type': int, 'metavar': 'N', 'default': 10000}),
            ('valid', {'type': int, 'metavar': 'N', 'default': 5000}),
            ('test', {'type': int, 'metavar': 'N', 'default': 5000}),
            ('treshold', {'type': float, 'metavar': 'T', 'default': 5400}),
            ('transform', {'type': int, 'metavar': ('NULL VALUE', 'SHIFT'), 'nargs': 2})
        ],
        'pos': [
            ('file_pref', {'help': "Output file prefix"}),
            ('files', {'help': "Source files", 'nargs': '*'})
        ]
    },

    'show' : {
        'help': "Preview dataset file.",
        'opts': [],
        'pos': [
            ('file', {'help': "Dataset file"}),
        ]
    }
}
