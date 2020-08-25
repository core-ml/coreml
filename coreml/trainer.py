"""Defines the base classes to be extended by specific types of models."""
import warnings
import logging
from os.path import join
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import wandb

import pytorch_lightning as pl
from coreml.data.data_module import DataModule
from coreml.modules import lm_factory
from coreml.utils.logger import color


class Trainer(pl.Trainer):
    def __init__(self, config, **kwargs):
        self.config = config
        callbacks, checkpoint_callback = self._get_callbacks(kwargs)
        kwargs['callbacks'] = callbacks
        kwargs['checkpoint_callback'] = checkpoint_callback
        kwargs['default_root_dir'] = config.output_dir
        super(Trainer, self).__init__(**kwargs)

        # define data module
        self.data_module = DataModule(
            config.data, config.trainer['batch_size'],
            config.trainer['num_workers'], **config.modes)

        # define lightning module
        config.module['config']['log_dir'] = config.log_dir
        module_params = {
            'config': config.module['config']
        }
        module_params.update(config.modes)
        self.model = lm_factory.create(
            config.module['name'], **module_params)

    def _get_callbacks(self, kwargs: Dict):
        callbacks = []

        # use learning rate logger only if there is a logger
        if kwargs['logger'] is not None:
            callbacks.append(pl.callbacks.lr_logger.LearningRateLogger(
                logging_interval='step'))

        # setup model checkpoint callback
        checkpoint_args = self.config.module['config']['checkpoint']

        # set filepath
        checkpoint_args['filepath'] = join(
            self.config.checkpoint_dir, '{epoch}')
        checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
            **checkpoint_args)

        return callbacks, checkpoint_callback

    def fit(self, model=None):
        if model is None:
            model = self.model

        # train the model
        super(Trainer, self).fit(model, datamodule=self.data_module)

    def evaluate(self, mode: str, ckpt_path: str = 'best'):
        """evaluate the model

        :param mode: which dataloader to use (train/val/test)
        :type mode: str
        :param ckpt_path: Either best or path to the checkpoint you wish to
            test. If None, use the weights from the last epoch to test.
            Default to None
        :type ckpt_path: str
        """
        # define dataloader
        eval_dataloader = getattr(self.data_module, f'{mode}_dataloader')()

        # reset test_model
        self.model.test_mode = mode

        super(Trainer, self).test(
            self.model, eval_dataloader, verbose=True)
