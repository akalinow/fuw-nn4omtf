# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    Model configuration structure.
"""

# Keys used also in CLI

model_hparams_keys = ['lrate', 'batch_size']
model_config_keys_datasets = ['ds_train', 'ds_valid', 'ds_test'] 
model_config_keys = model_config_keys_datasets + ['gpu']


# Model default values

_default_hparams_values = [
    0.001,
    32,
]

_default_config_values = [None] * 3 + [
    False
]


# Model data structure

_default_hparams = dict(zip(model_hparams_keys, _default_hparams_values))
_default_config = dict(zip(model_config_keys, _default_config_values))

model_data_default = {
    'hparams' : _default_hparams,
    'config' : _default_config,
}

