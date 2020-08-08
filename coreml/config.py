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
        # metrics
        params['metrics_to_track'] = [
            'auc-roc', 'precision', 'specificity', 'recall']
        params['allow_val_change'] = params.get('allow_val_change', False)

        # sampler
        params['data']['sampler'] = params['data'].get('sampler', {})
        params['data']['dataset']['params'] = params['data']['dataset'].get(
            'params', {})

        # defaults for subset tracker
        params['model']['subset_tracker'] = params['model'].get(
            'subset_tracker', {})
        params['model']['subset_tracker']['train'] = params['model'][
            'subset_tracker'].get('train', {})
        params['model']['subset_tracker']['val'] = params['model'][
            'subset_tracker'].get('val', {})

        # defaults for loading checkpoints
        if 'load' in params['model']:
            load_config = params['model']['load']
            load_config['resume_optimizer'] = load_config.get(
                'resume_optimizer', False)
            load_config['resume_epoch'] = load_config.get(
                'resume_epoch', load_config['resume_optimizer'])
            params['model']['load'] = load_config

        # evaluation
        params['model']['eval'] = params['model'].get('eval', {})
        params['model']['eval']['maximize_metric'] = params['model']['eval'].get(
            'maximize_metric', 'specificity')

        return params

    @staticmethod
    def _check_params(params: Dict):
        """Validates parameter values"""
        assert 'description' in params
        assert 'data' in params
        assert 'model' in params
        assert 'loss' in params['model']

        if 'optimizer' in params['model'] and 'scheduler' in params['model']['optimizer']:
            scheduler_config = params['model']['optimizer']['scheduler']

            if scheduler_config['name'] == 'StepLR':
                assert scheduler_config['update'] == 'epoch'
                assert 'value' not in scheduler_config
            if scheduler_config['name'] == 'MultiStepLR':
                assert scheduler_config['update'] == 'epoch'
                assert 'value' not in scheduler_config
            elif scheduler_config['name'] == 'ReduceLRInPlateau':
                assert scheduler_config['update'] == 'epoch'
                assert 'value' in scheduler_config
            elif scheduler_config['name'] == 'CyclicLR':
                assert scheduler_config['update'] == 'batch'
                assert 'value' not in scheduler_config
            elif scheduler_config['name'] == '1cycle':
                assert scheduler_config['update'] == 'batch'
                assert 'value' not in scheduler_config

    def update_from_params(self, params: Dict):
        """Updates parameters from dict"""
        params = self._set_defaults(params)
        self._check_params(params)
        self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `config.dict['lr']`"""
        return self.__dict__
