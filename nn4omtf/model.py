# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    Network model manager
"""

import datetime
import time
import numpy as np
import tensorflow as tf
import os
from nn4omtf.utils import dict_to_object, get_source_of_obj, json_to_dict,\
        dict_to_json, import_module_from_path, get_from_module_by_name,\
        dict_to_json_string
from nn4omtf.model_config import model_data_default, model_hparams_keys,\
        model_config_keys, model_config_keys_datasets 
from nn4omtf.const_model import MODEL_RESULTS

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
        'checkpoint_prefix',
    ]

    _model_paths_values = [
        None, 
        'checkpoints',
        'logs',
        'tb-logs',
        'test-outputs',
        
        'model.json',
        'builder.py',
        'checkpoints/ckpt',
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

        self._load_model_data()
        self._update_model_data_with_opts(**opts)
        self.pt_bins = OMTFModel.test_graph_builder(self.get_builder_func())


    def open_train_logs(self):
        stamp = '{:%Y-%m-%d-%H-%M-%S}'.format(datetime.datetime.now()) 
        tb_logs_path = os.path.join(self.paths.dir_tblogs, stamp)
        self.tb_writer = tf.summary.FileWriter(tb_logs_path)
        self.train_logs = {
            'hparams': self.model_data['hparams'],
            'stamp': stamp,
            'time_start': time.time(),
            'train': [],
            'valid': []
        }


    def save_train_logs(self):
        stamp = self.train_logs['stamp']
        path = os.path.join(self.paths.dir_logs, stamp)
        dict_to_json(path, self.train_logs)
        

    def add_train_log(self, epoch, batch, loss, acc, valid=False):
        name = 'valid' if valid else 'train'
        self.train_logs[name].append([time.time(), epoch, batch, float(loss), float(acc)])


    def update_config(self):
        """
        Save actual model data in model config file.
        """
        # Convert absolute paths to relative
        paths = [(k, self.model_data['config'][k]) for k in model_config_keys_datasets]
        for k, abs_path in paths:
            rel_path = os.path.relpath(abs_path, start=self.paths.dir_root)
            self.model_data['config'][k] = rel_path
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


    def test_graph_builder(builder_func):
        """
        Test model network builder by calling it within 
        new temporary graph.
        Rises exception when:
          - returned logits shape does not match with size of pt bins list,
          - first bin list's element is not zero

        Args:
            builder_func: builder function

        Returns:
            pt bins list
        """
        with tf.Graph().as_default() as g:
            x_ph = tf.placeholder(tf.float32)
            ind_ph = tf.placeholder(tf.bool)
            HITS_REDUCED_SHAPE = [18, 2] # TODO create constant
            logits, pt_bins = builder_func(x_ph, HITS_REDUCED_SHAPE, ind_ph)

            s = "Network output logits returned from builder function"
            s +="has wrong dimension!\n"
            s += "out_logits.shape: " + str(logits.shape) + "\n"
            s += "Expected number of classes: " + str( 2 * len(pt_bins) + 1)
            assert logits.shape[1] == 2 * len(pt_bins) + 1, s 
            assert pt_bins[0] == 0, 'First pt bins element is not ZERO!'
        return pt_bins


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
        # Before creation of model direcory sturcture, test correctness of
        # model builder
        OMTFModel.test_graph_builder(builder_fn)
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


    def tb_add_summary(self, n, summ, valid=False):
        self.tb_writer.add_summary(summ, n)


    def save_model(self, sess):
        self.saver.save(sess=sess, save_path=self.paths.checkpoint_prefix,
            write_meta_graph=False)
        print("Model saved!")


    def save_test_results(self, results, suffix=None, note=''):
        path = self.paths.dir_testouts
        name = '{:%Y-%m-%d-%H-%M-%S}'.format(datetime.datetime.now()) 
        if suffix is not None:
            name = name + '-' + suffix
        path = os.path.join(path, name)
        data = {
            MODEL_RESULTS.NOTE: note,
            MODEL_RESULTS.RESULTS: results,
            MODEL_RESULTS.PT_BINS: self.pt_bins,
        }
        np.savez_compressed(path, **data)


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
        return ">>> MODEL DATA:\n" + dict_to_json_string(self.model_data) + \
                ">>>>>>>>>>>>>>"
    

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
            # Create epmty file to push & pull `empty` directories with git
            with open(os.path.join(d, '.empty'), 'w') as f:
                f.write('empty file')


    def _load_model_data(self):
        # Convert relative paths to absolute
        self.model_data = json_to_dict(self.paths.file_model)
        paths = [(k, self.model_data['config'][k]) for k in model_config_keys_datasets]
        print(paths)
        for k, rel_path in paths:
            if rel_path is not None:
                abs_path = os.path.normpath(os.path.join(self.paths.dir_root, rel_path))
                self.model_data['config'][k] = abs_path

