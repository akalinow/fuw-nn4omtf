# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    Network model manager
"""

import tensorflow as tf
import os
from nn4omtf.utils import dict_to_object, get_source_of_obj, json_to_dict,\
        dict_to_json, import_module_from_path, get_from_module_by_name,\
        dict_to_json_string
from nn4omtf.model_config import model_data_default, model_hparams_keys,\
        model_config_keys, model_config_keys_datasets 

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
        'checkpoint_prefix'
    ]

    _model_paths_values = [
        None, 
        'checkpoints',
        'logs',
        'tb-logs',
        'test-outputs',
        
        'model.json',
        'builder.py',
        'checkpoints/ckpt'
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
        self.model_data = json_to_dict(self.paths.file_model)
        self._update_model_data_with_opts(**opts)


    def update_config(self):
        """
        Save actual model data in model config file.
        """
        dict_to_json(self.paths.file_model, self.model_data)


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
        mod = import_module_from_path(path=builder_file)
        fn = get_from_module_by_name(mod=mod, name=OMTFModel.FUNC_BUILDER)
        model = OMTFModel.create_new_model_with_builder_fn(model_directory, fn, **opts)
        return model


    def create_new_model_with_builder_fn(model_directory, builder_fn, **opts):
        """
        Create model and its directory using builder function.
        Args:
            model_directory: path to root model directory
            builder_fn: function TODO: add signature!
            **opts: model data options which should be saved

        Note: 
            It's very important to return unscaled logits as a result 
            of `create_nn` builder funciton. Softmax op will be applied
            to it in trainer internals.

        """
        # Crete directory structure first
        OMTFModel._create_structure(model_directory)
        paths =  OMTFModel._get_paths(model_directory)
        # Save builder code
        builder_code = get_source_of_obj(builder_fn)
        builder_file = paths.file_builder
        with open(builder_file, 'w') as f:
            f.write(builder_code)
        # Save default model data
        dict_to_json(paths.file_model, model_data_default)

        # Load default object with some options ...
        model = OMTFModel(
                model_directory=model_directory,
                builder_fn=builder_fn,
                **opts)
        # ... and then save
        model.update_config()
        return model


    def get_builder_func(self):
        """
        Loads builder function from source file.
        """
        mod = import_module_from_path(path=self.paths.file_builder)
        return get_from_module_by_name(mod=mod, name=OMTFModel.FUNC_BUILDER)


    def get_hparams(self):
        """
        Get model hyperparameters
        """
        return dict_to_object(self.model_data['hparams'])


    def get_config(self):
        """
        Get model data as namespace.
        """
        return dict_to_object(self.model_data['config'])


    def restore(self, sess):
        """
        Restore model within TensorFlow session.
        Returns:
            True, if checkpoint was restored.
            False, if checkpoint doesn't exist.
            Raises exception if restoring failes.
        """
        self.saver = tf.train.Saver() 
        if tf.train.checkpoint_exists(self.paths.checkpoint_prefix):
            self.saver.restore(sess, self.paths.checkpoint_prefix) 
            print("Checkpoint restored!")
            return True

        return False


    def tb_add_summary(self, n, summ):
        pass


    def save_model(self, sess):
        self.saver.save(sess, self.paths.checkpoint_prefix)
        print("Model saved!")


    def _update_model_data_with_opts(self, **opts):
        """
        Update model data using given opts.
        NOTE: dataset paths are converted to absolute path form
        Args:
            **opts: options
        """
        for k, v in opts.items():
            if v is None:
                continue
            if k in model_hparams_keys:
                self.model_data['hparams'][k] = v
            if k in model_config_keys:
                if v is not None and k in model_config_keys_datasets and not os.path.isabs(v):
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
        paths = [os.path.join(root_path, v) for v in OMTFModel._model_paths_values[1:]]
        paths = [root_path] + paths
        paths = dict(zip(OMTFModel._model_paths_fields, paths))
        return dict_to_object(paths) 


    def __str__(self):
        return dict_to_json_string(self.model_data)
    

    def _create_structure(root_dir):
        """
        Create model directory tree.
        Args:
            root_dir: root model directory
        """
        paths = OMTFModel._get_paths(root_dir)
        os.makedirs(root_dir)
        dirs = [v for (k, v) in vars(paths).items() if k.startswith('dir')]
        for d in dirs:
            os.makedirs(d, exist_ok=True)


