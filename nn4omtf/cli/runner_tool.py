# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    Main OMTFModel tool.
    Creates, trains and tests models.
"""

import os
from nn4omtf import OMTFModel, OMTFRunner
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
        model = OMTFModel(opts.model_dir)
        print(str(model))

    
    def _create(self, opts):
        path = '.' if opts.outdir is None else opts.outdir
        path = os.path.join(path, opts.name)
        model = OMTFModel.create_new_model_with_builder_file(
                model_directory=path,
                **vars(opts))
        print("\nCreated!\n")
        print(str(model))
 

    def _train(self, opts):
        model = OMTFModel(opts.model_dir, **vars(opts))
        if opts.update_config:
            model.update_config()
        print("\nLoaded!\n")
        print(str(model))
        runner = OMTFRunner()
        print(opts)
        runner.train(model, **vars(opts))


    def _test(self, opts):
        model = OMTFModel(opts.model_dir, **vars(opts))
        if opts.update_config:
            model.update_config()
        print("\nLoaded!\n")
        print(str(model))
        runner = OMTFRunner()
        runner.test(model, **vars(opts))

