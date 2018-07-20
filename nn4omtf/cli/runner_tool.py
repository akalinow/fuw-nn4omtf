# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    Main OMTFModel tool.
    Creates, trains and tests models.
"""

import os
from nn4omtf import OMTFModel, OMTFRunner
from .runner_tool_config import ACTION, parser_config
from .tool import OMTFTool


class OMTFRunnerTool(OMTFTool):


    def __init__(self):
        handlers = [
            (ACTION.SHOW, OMTFRunnerTool._show),
            (ACTION.MODEL, OMTFRunnerTool._create),
            (ACTION.TRAIN, OMTFRunnerTool._train),
            (ACTION.TEST, OMTFRunnerTool._test)
        ]
        super().__init__(parser_config, "OMTF NN trainer", handlers)
    

    def _show(opts):
        model = OMTFModel(opts.model_dir)
        print(str(model))

    
    def _create(opts):
        path = '.' if opts.outdir is None else opts.outdir
        path = os.path.join(path, opts.name)
        model = OMTFModel.create_new_model_with_builder_file(
                model_directory=path,
                **vars(opts))
        print(str(model))
 

    def _train(opts):
        model = OMTFModel(opts.model_dir, **vars(opts))
        if opts.update_config:
            model.update_config()
        print(str(model))
        runner = OMTFRunner()
        runner.train(model, **vars(opts))


    def _test(opts):
        model = OMTFModel(opts.model_dir, **vars(opts))
        if opts.update_config:
            model.update_config()
        print(str(model))
        runner = OMTFRunner()
        print(vars(opts))
        runner.test(model, **vars(opts))

