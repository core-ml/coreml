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


#     @abstractmethod
#     def load(self, load_config: Dict):
#         """Loads the network and optimizer states (optionally) from a config.

#         :param load_config: config defining parameters related to
#             loading the model and optimizer
#         :type load_config: Dict
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

