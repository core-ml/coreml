"""Defines the Config object used throughout the project"""
import os
from os.path import join, dirname, splitext
from typing import Dict, Any
from coreml.utils.io import read_yml, save_yml

# default output directory to use
OUT_DIR = "/output/"

# default directory where the data versions reside
DATA_ROOT = "/data"


class Config:
    """Class that loads parameters from a yml file.

    :param version: path of the .yml file which contains the parameters
    :type version: str
    """

    def __init__(self, version: str):
        assert version.endswith('.yml')
        self.version = version
        self.update_from_path(version)

        self.paths = {'OUT_DIR': self.__dict__.get('output_dir', OUT_DIR)}
        config_subpath = version.replace('.yml', '')
        self.config_save_path = os.path.join(
            self.paths['OUT_DIR'], config_subpath, 'config.yml')
        os.makedirs(dirname(self.config_save_path), exist_ok=True)

        self.output_dir = os.path.join(self.paths['OUT_DIR'], config_subpath)
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        self.log_dir = os.path.join(self.output_dir, "logs")

        # set data directory
        self.__dict__['data']['root'] = self.__dict__['data'].get(
            'root', DATA_ROOT)

        # save the config
        self.save()

        # create missing directories
        for path in [self.checkpoint_dir, self.log_dir]:
            os.makedirs(path, exist_ok=True)

    def __repr__(self):
        return "Config(version={})".format(self.version)

    def save(self):
        """Saves parameters"""
        save_yml(self.config_save_path, self.__dict__)

    def update_from_path(self, path: str):
        """Loads parameters from yml file"""
        params = read_yml(path)
        self.update_from_params(params)

    @staticmethod
    def _set_defaults(params: Dict):
        """Validates parameter values"""
        # set default train and val modes
        params['modes'] = params.get('modes', {
            'train_mode': 'train',
            'val_mode': 'val',
            'test_mode': 'test'
        })

        # logger
        params['logger'] = params.get('logger', {})

        # sampler
        # params['data']['sampler'] = params['data'].get('sampler', {})
        # params['data']['dataset']['params'] = params['data']['dataset'].get(
        #     'params', {})

        # evaluation
        params['module']['config']['eval'] = params['module']['config'].get(
            'eval', {'maximize_metric': 'specificity'})

        return params

    @staticmethod
    def _check_params(params: Dict):
        """Validates parameter values"""
        assert 'description' in params
        assert 'data' in params
        assert 'module' in params
        assert 'loss' in params['module']['config']

        # check scheduler params
        if 'optimizer' in params['module']['config']:
            optimizer_config = params['module']['config']['optimizer']
            if 'scheduler' in optimizer_config:
                scheduler_config = optimizer_config['scheduler']
                scheduler_opt_params = scheduler_config['opt_params']
                assert 'interval' in scheduler_opt_params

                if scheduler_config['name'] == 'ReduceLRInPlateau':
                    assert 'monitor' in scheduler_opt_params
                elif scheduler_config['name'] in ['CyclicLR', '1cycle']:
                    assert scheduler_opt_params['interval'] == 'batch'
                    assert 'monitor' not in scheduler_config['opt_params']

    def update_from_params(self, params: Dict):
        """Updates parameters from dict"""
        params = self._set_defaults(params)
        self._check_params(params)
        self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `config.dict['lr']`"""
        return self.__dict__
