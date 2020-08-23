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


class Trainer(pl.Trainer):
    def __init__(self, config, **kwargs):
        self.config = config
        callbacks, checkpoint_callback = self.get_callbacks(kwargs)
        kwargs['callbacks'] = callbacks
        kwargs['checkpoint_callback'] = checkpoint_callback
        kwargs['default_root_dir'] = config.output_dir

        super(Trainer, self).__init__(**kwargs)

    def get_callbacks(self, kwargs):
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

    def fit(self):
        # define data module
        data_module = DataModule(
            self.config.data, self.config.trainer['batch_size'],
            self.config.trainer['num_workers'], **self.config.modes)

        # define lightning module
        module_params = {
            'config': self.config.module['config']
        }
        module_params.update(self.config.modes)
        lightning_module = lm_factory.create(
            self.config.module['name'], **module_params)

        # train the model
        super(Trainer, self).fit(lightning_module, datamodule=data_module)

    def evaluate(self):
        pass
