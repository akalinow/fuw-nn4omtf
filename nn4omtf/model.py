# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    Network model manager
"""

import json


class OMTFModel:
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

    MODEL_BUILDER_FUNC = 'create_nn'

    def __init__(self, model_directory, **opts):
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


    def __str__(self):
        return 'model data stub'
    

    def create_new_model_with_builder_file(model_directory, builder_file, **opts):
        mod = utils.import_module_from_path(path=builder_file)
        fn = utils.get_from_module_by_name(
                mod=mod, 
                name=OMTFModel.MODEL_BUILDER_FUNC)
        model = OMTFModel(
                model_directory=model_directory,
                builder_fn=fn,
                **opts)


    def create_new_model_with_builder_fn(model_directory, builder_fn, **opts):

        OMTFModel._create_default_structure(model_directory)

        # Save builder code
        builder_code = utils.get_source_of_obj(builder_fn)
        builder_file = os.path.join(
            self.path, OMTFNN.CONST.NN_BUILD_FUN_FILENAME)
        with open(builder_file, 'w') as f:
            f.write(builder_code)
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


