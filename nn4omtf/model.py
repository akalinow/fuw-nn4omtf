# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    Network model manager
"""


import json
from nn4omtf.utils import dict_to_object, get_source_of_object, json_to_dict,\
        dict_to_json, import_module_from_path 


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

    FUNC_BUILDER = 'create_nn'

    _model_paths_fields = [
        'dir_root',
        'dir_checkpoints',
        'dir_logs',
        'dir_tblogs',
        'dir_testouts',

        'file_model', 
        'file_builder',
    ]

    _model_paths_values = [
        None, 
        'checkpoints',
        'logs',
        'tb-logs',
        'test-outputs',
        
        'model.json',
        'builder.py'
    ]


    def __init__(self, model_directory, **opts):
        """
        Loads configuration and network builder from model directory.
        Args:
            model_directory: root model directory
            **opts: model data options 
        """
        self.root_dir = model_directory
        self.paths = OMTFModel._get_paths(model_directory)

        # Load actual model data
        self.model_data = json_do_dict(self.paths.model_file)
        self._update_model_data_with_opts(**opts)


    def update_config(self):
        """
        Save actual model data in model config file.
        """
        dict_to_json(self.paths.model_file, self.model_data)


    def get_dataset_paths(self):
        """
        Returns:
            Namespace with dataset paths.
        """
        paths = [(k, self.model_data['config'][k]) for k in model_config_keys_datasets]
        return dict_to_object(dict(paths))


    def create_new_model_with_builder_file(model_directory, builder_file, **opts):
        """
        Create model and its directory using builder file.
        Args:
            model_directory: path to root model directory
            builder_file: `*.py` file with `create_nn` function implemented
            **opts: model data options which should be saved

        Returns:
            OMTFModel instance

        Method just compile python source file and takes function code.
        """
        mod = utils.import_module_from_path(path=builder_file)
        fn = utils.get_from_module_by_name(
                mod=mod, 
                name=OMTFModel.MODEL_BUILDER_FUNC)
        model = OMTFModel.create_new_model_with_builder_fn(model_directory, fn, **opts)
        return model


    def create_new_model_with_builder_fn(model_directory, builder_fn, **opts):
        """
        Create model and its directory using builder function.
        Args:
            model_directory: path to root model directory
            builder_fn: function TODO: add signature!
            **opts: model data options which should be saved
        """
        # Crete directory structure first
        OMTFModel._create_structure(model_directory)
        paths =  OMTFModel._get_paths(model_directory)
        # Save builder code
        builder_code = get_source_of_obj(builder_fn)
        builder_file = paths.builder_file
        with open(builder_file, 'w') as f:
            f.write(builder_code)
        # Save default model data
        dict_to_json(paths.file_model, model_data_default)

        # Load default object with some options ...
        model = OMTFModel(
                model_directory=model_directory,
                builder_fn=fn,
                **opts)
        # ... and then save
        model.update_config()
        return model


    def get_builder_func(self):
        """
        Loads builder function from source file.
        """
        mod = import_module_from_path(path=self.paths.file_builder)
        return utils.get_from_module_by_name(mod=mod, 
                name=OMTFModel.MODEL_BUILDER_FUNC)


    def _update_model_data_with_opts(**opts):
        """
        Update model data using given opts.
        NOTE: dataset paths are converted to absolute path form
        Args:
            **opts: options
        """
        for k, v in opts.items():
            if k in model_hparams_keys:
                self.model_data['hparams'][k] = v
            if k in model_config_keys:
                if k in model_config_keys_datasets and not os.path.isabs(v):
                    v = os.path.abspath(v)
                self.model_data['config'][k] = v


    def _get_paths(root_path):
        """
        Get namespace with all important paths.
        Args:
            root_path: model root directory

        Returns:
            Namespace with all important model paths and directories.
        """
        _model_paths_values[0] = root_path
        _paths = dict(zip(_model_paths_fields, _model_paths_values))
        return dict_to_object(paths) 


    def __str__(self):
        return dict_to_json_string(self.model_data)
    

    def _create_structure(root_dir):
        """
        Create model directory tree.
        Args:
            root_dir: root model directory
        """
        os.makedirs(root_dir)
        dirs = [v for (n, v) in 
                zip(OMTFModel._model_paths_fields, OMTFModel._model_paths_values) 
                if n.startswith('dir')]
        for d in dirs:
            path = os.path.join(root_dir, d)
            os.makedirs(path)


