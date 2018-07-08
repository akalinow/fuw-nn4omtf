# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    Model configuration structure.
"""

# Default values
_datasets = dict([(v.lower(), None) for k,v in vars(DATASET_TYPES) if not k.startwith('_')])
_hparams = {
    'lrate' : 0.001,
    'batch_size': 32
}
_config = {
    'pt_bins' : None
}

MODEL_DATA_MAIN = {
    'hparams' : _hparams,
    'config' : _config,
    'datasets': _datasets
}

_opts = [
    'pt_bins',
    

    'epochs_trained',
    'train_dataset',
    'valid_dataset',
    
]

_cli_opts = [
    'train_dataset',
    'valid_dataset',

]
