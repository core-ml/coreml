"""Defines the binary classification model"""
import sys
from os import makedirs
from os.path import join, exists, dirname, basename, splitext
import logging
from collections import defaultdict
from typing import Any, Dict, Tuple, List
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import wandb
from sklearn.metrics import precision_recall_curve, accuracy_score,\
    recall_score, precision_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

from coreml.data.dataloader import get_dataloader
from coreml.models.base import Model
from coreml.utils.logger import color
from coreml.utils.io import save_pkl, load_pkl
from coreml.utils.metrics import ConfusionMatrix
from coreml.utils.metrics import factory as metric_factory
from coreml.utils.viz import fig2im, plot_lr_vs_loss, \
    plot_classification_metric_curve
from coreml.utils.wandb import get_audios, get_images, get_indices, \
    get_confusion_matrix
from coreml.utils.loss import loss_factory
from coreml.networks import factory as network_factory
from coreml.optimizer import optimizer_factory, scheduler_factory
np.set_printoptions(suppress=True)


class BinaryClassificationModel(Model):
    """Classification model class

    :param config: Config object
    :type config: Config
    """
    def log_epoch_summary(self, mode: str, epoch_losses: dict, metrics: dict,
                          epoch_data: dict, learning_rates: List[Any],
                          batch_losses: defaultdict,
                          instance_losses: defaultdict,
                          use_wandb: bool):
        """Logs the summary of the epoch (losses, metrics and visualizations)

        :param mode: train/val or test mode
        :type mode: str
        :param epoch_losses: aggregate losses aggregated for the epoch
        :type epoch_losses: dict
        :param metrics: metrics for the epoch
        :type metrics: dict
        :param epoch_data: dictionary of various values in the epoch
        :type epoch_data: dict
        :param learning_rates: Dynamically accumulated learning rates per batch
            over all epochs
        :type learning_rates: List[Any]
        :param batch_losses: Dynamically accumulated losses per batch
        :type batch_losses: defaultdict
        :param instance_losses: losses per instance in the batch
        :type instance_losses: dict
        :param use_wandb: flag to decide whether to log visualizations to wandb
        :type use_wandb: bool
        """
        logging.info(
            color("V: {} | Epoch: {} | {} | Avg. Loss {:.4f}".format(
                self.config.version, self.epoch_counter, mode.capitalize(),
                epoch_losses['loss']
            ), 'green')
        )

        metric_log = "V: {} | Epoch: {} | {}".format(
                    self.config.version, self.epoch_counter, mode.capitalize())

        for metric in self.config.metrics_to_track:
            metric_log += ' | {}: {:.4f}'.format(metric, metrics[metric])

        logging.info(color(metric_log, 'green'))

        # update wandb
        if use_wandb:
            self._update_wandb(
                mode, epoch_losses, metrics, epoch_data, learning_rates,
                batch_losses)

        if batch_losses is not None:
            # reshape batch losses to the shape of instance losses
            instance_batch_losses = dict()
            for loss_name, loss_value in batch_losses.items():
                loss_value = loss_value.reshape(-1, 1)
                loss_value = np.repeat(
                    loss_value, self.model_config['batch_size'],
                    axis=-1).reshape(-1)

                # correct for incomplete last batch
                instance_batch_losses[loss_name] = loss_value[:len(
                    epoch_data['items'])]

        # log instance-level epochwise values
        instance_values = {
            'paths': [item.path for item in epoch_data['items']],
            'predictions': epoch_data['predictions'],
            'targets': epoch_data['targets'],
        }

        for key, value in metrics.items():
            instance_values[key] = value

        for loss_name in instance_losses:
            instance_values['instance_loss'] = instance_losses[loss_name]
            if batch_losses is not None:
                instance_values['batch_loss'] = instance_batch_losses[loss_name]

        save_path = join(self.config.log_dir, 'epochwise', '{}/{}.pt'.format(
            mode, self.epoch_counter))
        makedirs(dirname(save_path), exist_ok=True)
        torch.save(instance_values, save_path)

