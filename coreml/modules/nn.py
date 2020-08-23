"""Defines the class for feed-forward LightningModule."""
from typing import Dict, Tuple, Any, Union
from collections import OrderedDict, defaultdict
import logging
from abc import abstractmethod

import torch
import torch.nn as nn
import wandb
import pytorch_lightning as pl

from coreml.modules.backbones.utils import _correct_state_dict
from coreml.modules.layers import layer_factory
from coreml.modules.init import init_factory
from coreml.utils.typing import LayerConfigDict
from coreml.utils.logger import color
from coreml.optimization import optimizer_factory, scheduler_factory


# TODO:
# wandb-logger
# model checkpoint - load models
# tests
# check loading of checkpoints with matching keys but different shape


class NeuralNetworkModule(pl.LightningModule):
    """Extends the LightningModule for any feed-forward model

    :param config: config for defining the module
    :type config: Dict
    :param train_mode: key for the train data split, defaults to 'train'
    :type train_mode: str, optional
    :param val_mode: key for the validation data split, defaults to 'val'
    :type val_mode: str, optional
    :param test_mode: key for the test data split, defaults to 'test'
    :type test_mode: str, optional
    """
    def __init__(
            self, config: Dict, train_mode: str = 'train',
            val_mode: str = 'val', test_mode: str = 'test'):
        super(NeuralNetworkModule, self).__init__()
        self.config = config
        self.train_mode = train_mode
        self.val_mode = val_mode
        self.test_mode = test_mode

        # build and initialize the network
        self._build_network()
        self._init_network()

    def _build_network(self):
        """Defines method to build the network"""
        self.blocks = nn.Sequential()
        for index, layer_config in enumerate(self.config['network']['config']):
            layer = layer_factory.create(
                layer_config['name'], **layer_config['params'])
            self.blocks.add_module(
                name='{}_{}'.format(index, layer_config['name'].lower()),
                module=layer)

    def _init_network(self):
        """Initializes the parameters of the network"""
        if 'init' not in self.config['network']:
            return

        logging.info(color('Initializing the parameters'))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                self._init_param(m.weight, 'weight')
                if m.bias is not None:
                    self._init_param(m.bias, 'bias')

            elif isinstance(m, nn.BatchNorm2d):
                self._init_param(m.weight, 'bn_weight')
                self._init_param(m.bias, 'bn_bias')

    def _init_param(self, tensor, key):
        tensor_init_config = self.config['network']['init'].get(key)
        if 'name' in tensor_init_config:
            tensor_init_params = tensor_init_config.get('params', {})
            tensor_init_params['tensor'] = tensor
            init_factory.create(
                tensor_init_config['name'], **tensor_init_params)

    def freeze(self):
        # freeze layers based on config
        for name, param in self.blocks.named_parameters():
            layer_index = int(name.split('_')[0])
            if not self.config[layer_index].get('requires_grad', True):
                logging.info('Freezing layer: {}'.format(name))
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the network

        :param x: input to the network
        :type x: torch.Tensor

        :return: output on forward pass of input x
        """
        return self.blocks(x)

    def configure_optimizers(self):
        """Setup optimizers and schedulers to be used while training"""
        # logging.info(color("Setting up the optimizer ..."))
        kwargs = self.config['optimizer']['args']
        kwargs.update({'params': self.blocks.parameters()})
        optimizer = optimizer_factory.create(
            self.config['optimizer']['name'],
            **kwargs)

        if 'scheduler' not in self.config['optimizer']:
            return optimizer

        scheduler_config = self.config['optimizer']['scheduler']
        scheduler_config['params']['optimizer'] = optimizer

        scheduler = {
            'scheduler': scheduler_factory.create(
                scheduler_config['name'],
                **scheduler_config['params']),
            'monitor': scheduler_config['monitor'],
            'interval': scheduler_config['interval'],
            'frequency': scheduler_config.get('frequency', 1),
        }

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }

    def watch(self):
        """Defines how to track gradients and weights in wandb"""
        if self.logger is not None:
            self.logger.watch(self.blocks, log='all')

    def process_batch(self, batch: Any) -> Dict:
        """Returns the predictions, targets and loss for each batch

        :param batch: one batch of data containing inputs and targets
        :type batch: Any
        :return: dict containing predictions and targets
        """
        # process one batch to compute and return the inputs,
        # predictions, ground truth and item in the batch
        inputs = batch['signals']
        labels = batch['labels']

        predictions = self(inputs)

        return {
            'predictions': predictions.squeeze(),
            'targets': labels,
            'items': batch['items']
        }

    @abstractmethod
    def calculate_instance_loss(
            self, predictions, targets, mode: str,
            as_numpy: bool = False) -> dict:
        """Calculate loss per instance in a batch

        :param predictions: Predictions (Predicted)
        :type predictions: Any
        :param targets: Targets (Ground Truth)
        :type targets: Any
        :param mode: train/val/test mode
        :type mode: str
        :param as_numpy: flag to decide whether to return losses as np.ndarray
        :type as_numpy: bool

        :return: dict of losses with list of loss values per instance
        """
        pass

    def calculate_batch_loss(self, instance_losses) -> dict:
        """Calculate mean of each loss for the batch

        :param instance_losses: losses per instance in the batch
        :type instance_losses: dict

        :return: dict containing various loss values over the batch
        """
        losses = dict()
        for key in instance_losses:
            losses[key] = torch.mean(instance_losses[key])

        return losses

    def _step(
            self, batch: Any, mode: str, log: bool = False
            ) -> Union[pl.TrainResult, pl.EvalResult]:
        """Perform one step of train/val/test

        :param batch: one batch of data containing inputs and targets
        :type batch: Any
        :param mode: either of train/val/test mode
        :type mode: str
        :param log: whether to log values, defaults to False
        :type log: bool, optional
        """
        batch_data = self.process_batch(batch)

        # calculate loss per instance in the batch
        instance_losses = self.calculate_instance_loss(
            predictions=batch_data['predictions'],
            targets=batch_data['targets'],
            mode=mode)

        # calculate loss for the batch
        batch_losses = self.calculate_batch_loss(instance_losses)
        loss = batch_losses['loss']

        if log and self.logger is not None:
            self.logger.experiment.log({
                f'{mode}/step_loss': loss,
            }, step=self.logger.experiment.step)

        return OrderedDict({'loss': loss})

    def training_step(self, batch, batch_idx):
        return self._step(batch, self.train_mode, log=True)

    def validation_step(self, batch, batch_idx):
        return self._step(batch, self.val_mode)

    def test_step(self, batch, batch_idx):
        return self._step(batch, self.test_mode)

    def _epoch_end(self, outputs, mode):
        epoch_outputs = defaultdict(list)
        for output in outputs:
            for key, value in output.items():
                epoch_outputs[key].append(value)

        logs = {}
        for key, value in epoch_outputs.items():
            logs[f'{mode}/{key}'] = torch.stack(value).mean()
            epoch_outputs[key] = torch.stack(value).mean()

        if self.logger is not None:
            # ideally this should be self.global_step but lightning
            # calls self.logger without step (within trainer/evaluation_loop.py
            # - __log_evaluation_epoch_metrics()) implicitly and that
            # increments the experiment step by 1.
            self.logger.experiment.log(logs, step=self.logger.experiment.step)
        return OrderedDict(epoch_outputs)

    def training_epoch_end(self, outputs):
        return self._epoch_end(outputs, self.train_mode)

    def validation_epoch_end(self, outputs):
        return self._epoch_end(outputs, self.val_mode)

    def test_epoch_end(self, outputs):
        return self._epoch_end(outputs, self.test_mode)

    def on_fit_start(self):
        # log gradients and model parameters
        self.watch()

    # def on_fit_end(self):
    #     import ipdb; ipdb.set_trace()

    # def on_save_checkpoint(self, checkpoint):
    #     import ipdb; ipdb.set_trace()
