# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    Utilities
"""

import argparse
 
def create_parser(parser_config, desc):
    parser = argparse.ArgumentParser(description=desc)
    subparsers = parser.add_subparsers(title="Actions", description="Actions to perform", dest="action")
    for action, opts in parser_config.items():
        p = subparsers.add_parser(action, help=opts['help'])
        for op, kw in opts['opts']:
            p.add_argument('--' + op, **kw)
        for op, kw in opts['pos']:
            p.add_argument(op, **kw)
    return parser
