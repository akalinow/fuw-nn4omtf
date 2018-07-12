# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    OMTF dataset command line tool configuration
"""

from nn4omtf.model_config import model_hparams_keys, model_config_keys

class ACTION:
    MODEL = 'model'
    TRAIN = 'train'
    TEST = 'test'
    SHOW = 'show'


model_hparams_opts_args = [
    {'help': 'Set learning rate', 'type': float},
    {'help': 'Set batch size', 'type': int}
]


model_config_opts_args = [ 
    {'help': "TRAIN dataset path", 'metavar': 'PATH'},
    {'help': "VALID dataset path", 'metavar': 'PATH'},
    {'help': "TEST dataset path", 'metavar': 'PATH'},
    {'help': "Use GPU", 'action': 'store_true'}
]


model_hparams_opts = [(x, y) for x, y in zip(model_hparams_keys, model_hparams_opts_args)]
model_config_opts = [(x, y) for x, y in zip(model_config_keys, model_config_opts_args)]


parser_config = {

    ACTION.MODEL: {
        'help': "Create network model",
        'opts': model_hparams_opts + model_config_opts + [
            ('outdir', {'help': "Output directory"}),
        ],
        'pos': [
            ("builder_file", {'help': "Model builder file"}),
            ("name", {'help': "Model name"})
        ]
    },

    ACTION.TRAIN: {
        'help': "Train model",
        'opts': model_config_opts + model_hparams_opts + [
            ('epochs', {'type': int, 'metavar': 'N', 'default': 1, 'help': 'Set training duration'}),
            ('no_checkpoints', {'action': 'store_true', 'help': 'Don`t make checkpoints'}),
            ('time_limit', {'metavar': 'H+:MM:SS', 'help': 'Training time limit'}),
            ('update_config', {'action': 'store_true', 'help': 'Update model config with provided options'}),
            ('validation_ival', {'type': int, 'help': 'Number of batches processed between validation'}),
            ('train_summary_ival', {'type': int, 'help': 'Number of batches processed between summary collection'})
        ],
        'pos': [
            ('model_dir', {'help': "Directory of model to be trained"}),
        ]
    },

    ACTION.TEST: {
        'help': "Run test and save results",
        'opts': model_config_opts + model_hparams_opts + [
            ('update_config', {'action': 'store_true', 'help': 'Update model config with provided options'})
        ],
        'pos': [
            ('model_dir', {'help': "Directory of model to be tested"}),
        ]
    },

    ACTION.SHOW: {
        'help': "Preview model data",
        'opts': [],
        'pos': [
            ('model_dir', {'help': "Directory of model to be shown"}),
        ]
    }
}
