# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    Network model manager
"""

import json


class OMTFNetwork:
    """
    Network model manager.

    # Model directory structure

    `model name/` - root directory
      |- `checkpoints/` - model checkpoints directory
      |- `logs/` - log files
      |- `tb-logs/` - tensorboard logs
      |- `test-outputs/` - outputs from tests
      |- `model.json` - model data file
      |- `builder.py` - builder code
     
    # Model data
    
    Model data is loaded into `model_data` variable.
    Dict can be modified and then stored in updated file.

    """
    DIR_CHECKPOINTS = 'checkpoints'
    DIR_LOGS = 'logs'
    DIR_TBLOGS = 'tb-logs'
    DIR_TEST_OUTS = 'test-outputs'
    FILE_MODEL = 'model.json'
    FILE_BUILDER = 'builder.py'

    def __init__(self, model_directory):
        """
        Loads configuration and network builder from directory.
        """
        self.root_dir = model_directory
        self.model_data_path = os.path.join(self.root_dir, 'model.json')
        self.model_data = json.load(self.model_data_path)

    def save_model_data(self):
        json.dumps(self.model_data, self.model_data_path)


    def get_dataset_paths(self):
        conf = self.config['configs']
        paths = [conf[k] for k in ['TRAIN_DS_PATH', 'VALID_DS_PATH', 'TEST_DS_PATH']]
        return tuple(paths)

    def get_config(self):
        return self.model_data['config']

    
    def create_new_model_using_builder_file(model_directory, builder_file):
        pass

    def create_new_model_using_builder_fn(model_directory, builder_fn):
        pass
    
    def _create_default_structure(root_dir):
        """
        Create model directory tree.
        Args:
            root_dir: root model directory
        """
        os.makedirs(root_dir)
        dirs = [self.DIR_CHECKPOINTS,
                self.DIR_LOGS, 
                self.DIR_TBLOGS, 
                self.DIR_TEST_OUTS]
        for d in dirs:
            path = os.path.join(root_dir, d)
            os.makedirs(path)


