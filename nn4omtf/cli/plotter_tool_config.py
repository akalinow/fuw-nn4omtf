# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    OMTF dataset command line tool configuration

    TODO: implement data plotting, i have to think what types of plots will be
    needed ....
"""


class ACTION:
    SHOW = 'show'


parser_config = {
    'show' : {
        'help': "Preview result files or list available results",
        'opts': [],
        'pos': [
            ('path', {'help': "Result npz file path or model directory"}),
        ]
    }
}
