"""Defines the multi-class classification model"""
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
from coreml.models.binary_classification import BinaryClassificationModel
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


class MultiClassClassificationModel(BinaryClassificationModel):
    """Multi-class classification model class

    :param config: Config object
    :type config: Config
    """
    def calculate_instance_loss(
            self, predictions: torch.FloatTensor, targets: torch.LongTensor,
            mode: str, as_numpy: bool = False) -> dict:
        """Calculate loss per instance in a batch

        :param predictions: Predictions (Predicted)
        :type predictions: torch.FloatTensor
        :param targets: Targets (Ground Truth)
        :type targets: torch.LongTensor
        :param mode: train/val/test
        :type mode: str
        :param as_numpy: flag to decide whether to return losses as np.ndarray
        :type as_numpy: bool

        :return: dict of losses with list of loss values per instance
        """
        loss_config = self.model_config.get('loss')[mode]
        criterion = loss_factory.create(
            loss_config['name'], **loss_config['params'])
        loss = criterion(predictions, targets)

        if as_numpy:
            loss = loss.cpu().numpy()

        return {'loss': loss}

    def get_eval_params(self, epoch_data: dict) -> Tuple:
        """Get evaluation params by optimizing on the given data

        :param epoch_data: dictionary of various values in the epoch
        :type epoch_data: dict

        :return: dict containing evaluation parameters
        """
        return {}

    def compute_epoch_metrics(
            self, predictions: Any, targets: Any,
            classes: List[str] = None) -> dict:
        """Computes metrics for the epoch

        :param targets: ground truth
        :type targets: Any
        :param predictions: model predictions
        :type predictions: Any
        :param classes: list of classes in the target
        :type classes: List[str], defaults to None

        :return: dictionary of metrics as provided in the config file
        """
        targets = targets.cpu()
        predicted_labels = torch.argmax(predictions, dim=1).detach().cpu()

        if classes is None:
            classes = self.model_config['classes']

        confusion_matrix = ConfusionMatrix(classes)
        confusion_matrix(targets, predicted_labels)

        metrics = {
            'accuracy': accuracy_score(targets, predicted_labels),
            'confusion_matrix': confusion_matrix.cm,
        }

        return metrics

    def _update_wandb(self, mode: str, epoch_losses: dict, metrics: dict,
                      epoch_data: dict, learning_rates: List[Any] = None,
                      batch_losses: defaultdict = None):
        """Logs values to wandb

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
        :type learning_rates: List[Any], defaults to None
        :param batch_losses: Dynamically accumulated losses per batch
        :type batch_losses: defaultdict, defaults to None
        """
        super(BinaryClassificationModel, self)._update_wandb(
            mode, epoch_losses, metrics)

        # log learning rates vs losses
        if learning_rates is not None and batch_losses is not None:
            lr_vs_loss = plot_lr_vs_loss(
                learning_rates, batch_losses['loss'], as_figure=True)
            self.wandb_logs['{}/lr-vs-loss'.format(mode)] = lr_vs_loss
            plt.close()

        self.wandb_logs['{}/confusion_matrix'.format(mode)] = wandb.Image(
            get_confusion_matrix(
                metrics['confusion_matrix'], self.model_config['classes']))

        # log to wandb
        wandb.log(self.wandb_logs, step=self.epoch_counter)
