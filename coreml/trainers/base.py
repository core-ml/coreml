"""Defines the base classes to be extended by specific types of models."""
import warnings
import logging
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
# from coreml.callbacks import ModelCheckpoint

#     def calculate_epoch_loss(self, loss_dict: dict) -> dict:
#         """Calculate mean of each loss for the epoch

#         :param loss_dict: dictionary containing arrays of losses in the epoch
#         :type loss_dict: dict

#         :return: dict containing aggregated loss values over the epoch
#         """
#         epoch_losses = dict()

#         for key in loss_dict.keys():
#             epoch_losses[key] = np.mean(loss_dict[key])

#         return epoch_losses

#     def _gather_losses(self, loss_dict: defaultdict) -> dict:
#         """Gather all values per loss in one tensor

#         :param loss_dict: dictionary containing lists of various losses
#         :type loss_dict: defaultdict

#         :return: dict containing a running list of various losses per batch
#         """
#         for loss_name, loss_value in loss_dict.items():
#             loss_dict[loss_name] = torch.cat(
#                 loss_dict[loss_name]).detach().cpu().numpy()

#         return loss_dict

#     def _accumulate_losses(self, loss_dict: defaultdict, losses: dict) -> dict:
#         """Update the accumulated dict of epoch losses with the current losses

#         :param loss_dict: dictionary containing lists of various losses
#         :type loss_dict: defaultdict
#         :param losses: losses to be added
#         :type losses: dict

#         :return: dict containing a running list of various losses per batch
#         """
#         for loss_name, loss_value in losses.items():
#             loss_dict[loss_name].append(loss_value.reshape(-1).detach().cpu())

#         return loss_dict

#     @abstractmethod
#     def _gather_data(self, epoch_data: dict) -> Tuple:
#         """Gather preds, targets & other epoch data in one tensor

#         :param epoch_data: dictionary containing lists of various epoch values
#         :type epoch_data: dict

#         :return: dictionary with different values as one tensor
#         """
#         pass

#     @abstractmethod
#     def log_epoch_summary(self, mode: str, epoch_losses: dict, metrics: dict,
#                           epoch_data: dict, learning_rates: List[Any],
#                           batch_losses: defaultdict,
#                           instance_losses: defaultdict, use_wandb: bool):
#         """Logs the summary of the epoch (losses, metrics and visualizations)

#         :param mode: train/val or test mode
#         :type mode: str
#         :param epoch_losses: aggregate losses aggregated for the epoch
#         :type epoch_losses: dict
#         :param metrics: metrics for the epoch
#         :type metrics: dict
#         :param epoch_data: dictionary of various values in the epoch
#         :type epoch_data: dict
#         :param learning_rates: Dynamically accumulated learning rates per batch
#             over all epochs
#         :type learning_rates: List[Any]
#         :param batch_losses: Dynamically accumulated losses per batch
#         :type batch_losses: defaultdict
#         :param instance_losses: losses per instance in the batch
#         :type instance_losses: defaultdict
#         :param use_wandb: flag to decide whether to log visualizations to wandb
#         :type use_wandb: bool
#         """
#         pass

#     @abstractmethod
#     def get_eval_params(self, epoch_data: dict) -> Tuple:
#         """Get evaluation params by optimizing on the given data

#         :param epoch_data: dictionary of various values in the epoch
#         :type epoch_data: dict

#         :return: dict containing evaluation parameters
#         """
#         pass

#     @abstractmethod
#     def compute_epoch_metrics(self, predictions: Any, targets: Any) -> dict:
#         """Computes metrics for the epoch

#         :param targets: Targets (Ground Truth)
#         :type targets: Any
#         :param predictions: Predictions (Predicted)
#         :type predictions: Any

#         :return: dictionary of metrics as provided in the config file
#         """
#         pass

#     @abstractmethod
#     def save(self, epoch_metric_values: Dict, use_wandb: bool):
#         """Saves the model and optimizer states

#         :param epoch_metric_values: validation metrics computed for current epoch
#         :type epoch_metric_values: Dict
#         :param use_wandb: flag to decide whether to log visualizations to wandb
#         :type use_wandb: bool
#         """
#         pass

#     @abstractmethod
#     def load(self, load_config: Dict):
#         """Loads the network and optimizer states (optionally) from a config.

#         :param load_config: config defining parameters related to
#             loading the model and optimizer
#         :type load_config: Dict
#         """
#         pass

#     @abstractmethod
#     def _accumulate_lr(self, learning_rates: List[Any]) -> dict:
#         """Accumulate learning rate values

#         :param learning_rates: Dynamically accumulated learning rates per batch
#             over all epochs
#         :type learning_rates: List[Any]
#         :return: dict containing a running list of learning rates
#         """
#         pass

#     @abstractmethod
#     def process_batch(self, batch: Any, mode: str = None):
#         """Returns the predictions and targets for each batch

#         :param batch: one batch of data
#         :type batch: Any
#         :param mode: train/val/test mode
#         :type mode: str

#         :return: dict containing predictions and targets
#         """
#         pass

#     def process_epoch(
#             self, data_loader: DataLoader, mode: str = None,
#             training: bool = False, use_wandb: bool = True,
#             log_summary: bool = True, overfit_batch: bool = False):
#         """Basic epoch function (Used for train/val/test epochs)
#         Args:
#         :param dataloader: torch DataLoader for the epoch
#         :type dataloader: DataLoader
#         :param mode: train/val/test mode
#         :type mode: str, defaults to None
#         :param training: specifies where the model should be in training mode;
#             if True, network is set to .train(). Else, it is set to .eval()
#         :type training: str, defaults to False
#         :param use_wandb: whether to log visualizations to wandb
#         :type use_wandb: bool, defaults to True
#         :param log_summary: whether to log epoch summary
#         :type log_summary: bool, defaults to True
#         :param overfit_batch: whether this run is for overfitting on a batch
#         :type overfit_batch: bool
#         """
#         instance_losses = defaultdict(list)
#         batch_losses = defaultdict(list)

#         epoch_data = defaultdict(list)
#         learning_rates = []

#         if training:
#             training_mode = color('train', 'magenta')
#             self.network.train()
#         else:
#             training_mode = color('eval', 'magenta')
#             self.network.eval()

#         logging.info('{}: {}'.format(
#             color('Setting network training mode:', 'blue'),
#             color(training_mode)
#         ))

#         iterator = tqdm(data_loader, dynamic_ncols=True)

#         for batchID, batch in enumerate(iterator):
#             # process one batch to compute and return the inputs, predictions,
#             # ground truth and item in the batch
#             batch_data = self.process_batch(batch)

#             # calculate loss per instance in the batch
#             _instance_losses = self.calculate_instance_loss(
#                 predictions=batch_data['predictions'],
#                 targets=batch_data['targets'],
#                 mode=mode)

#             # calculate loss for the batch
#             _batch_losses = self.calculate_batch_loss(_instance_losses)

#             if mode is not None:
#                 # log batch summary
#                 self.log_batch_summary(iterator, mode, _batch_losses)

#                 # update network weights in training mode
#                 if training:
#                     self.update_network_params(_batch_losses)

#             # append batch loss to the list of losses for the epoch
#             instance_losses = self._accumulate_losses(
#                 instance_losses, _instance_losses)

#             # append batch loss to the list of losses for the epoch
#             batch_losses = self._accumulate_losses(batch_losses, _batch_losses)

#             # accumulate learning rate before scheduler step
#             self._accumulate_lr(learning_rates)

#             # update optimizer parameters using schedulers that operate
#             # per batch like CyclicalLearningRate
#             if hasattr(self, 'update_freq') and 'batch' in self.update_freq and training:
#                 self.update_optimizer_params(_batch_losses, 'batch')

#             # accumulate predictions, targets and items over the epoch
#             for key in batch_data:
#                 if isinstance(batch_data[key], torch.Tensor):
#                     batch_data[key] = batch_data[key].detach().cpu()

#                 # ignore storing inputs
#                 if key == 'inputs':
#                     continue

#                 epoch_data[key].append(batch_data[key])

#             # ignore other batches after the first batch if we are
#             # overfitting a batch
#             if overfit_batch:
#                 break

#             # break

#         logging.info('Gathering data')
#         epoch_data = self._gather_data(epoch_data)

#         logging.info('Gathering losses')

#         # gather all instance losses
#         instance_losses = self._gather_losses(instance_losses)

#         # gather all batch losses
#         batch_losses = self._gather_losses(batch_losses)

#         # accumulate list of batch losses to epoch loss
#         epoch_losses = self.calculate_epoch_loss(batch_losses)

#         logging.info('Computing metrics')

#         # get parameters for evaluation like the optimal
#         # threshold for classification
#         eval_params = self.get_eval_params(epoch_data)

#         # calculate metrics for the epoch
#         logging.info('Computing metrics')
#         metrics = self.compute_epoch_metrics(
#             epoch_data['predictions'], epoch_data['targets'],
#             **eval_params)

#         if log_summary:
#             logging.info('Logging epoch summary')

#             # log losses, metrics and visualizations
#             self.log_epoch_summary(
#                 mode, epoch_losses, metrics, epoch_data,
#                 learning_rates, batch_losses, instance_losses,
#                 use_wandb)

#         results = dict()
#         results.update(epoch_losses)
#         results.update(metrics)
#         results['batch_losses'] = batch_losses
#         results['instance_losses'] = instance_losses

#         for key in epoch_data:
#             results[key] = epoch_data[key]
#         return results

#     @abstractmethod
#     def evaluate(
#             self, data_loader: DataLoader, mode: str, use_wandb: bool = True,
#             ignore_cache: bool = True):
#         """Evaluate the model on given data

#         :param data_loader: data_loader made from the evaluation dataset
#         :type data_loader: DataLoader
#         :param mode: split of the data represented by the dataloader
#             (train/test/val)
#         :type mode: str
#         :param use_wandb: flag to decide whether to log visualizations to wandb
#         :type use_wandb: bool, defaults to True
#         :param ignore_cache: whether to ignore cached values
#         :type ignore_cache: bool, defaults to True
#         """
#         pass

#     def _update_wandb(self, mode: str, epoch_losses: dict, metrics: dict):
#         """Logs values to wandb

#         :param mode: train/val or test mode
#         :type mode: str
#         :param epoch_losses: aggregate losses aggregated for the epoch
#         :type epoch_losses: dict
#         :param metrics: metrics for the epoch
#         :type metrics: dict
#         """
#         logging.info('Logging to W&B')
#         self.wandb_logs = {}

#         for loss, value in epoch_losses.items():
#             self.wandb_logs['{}/{}'.format(mode, loss)] = value

#         for metric, value in metrics.items():
#             # only log metrics with scalar values here
#             if isinstance(value, (int, float)):
#                 self.wandb_logs['{}/{}'.format(mode, metric)] = value
