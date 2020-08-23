"""Defines the class for feed-forward LightningModule."""
from typing import Dict, Tuple, Any
import torch

from coreml.modules.nn import NeuralNetworkModule
from coreml.utils.loss import loss_factory

# TODO:
# wandb-logger
# model checkpoint
# LR logger
# writing everything in train.py
# tests
# handle binary with 1/2 outputs
# multi should only be more than 2
# sanity check binary with 1 = BCELoss


class BinaryClassificationModule(NeuralNetworkModule):
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
    def calculate_instance_loss(
            self, predictions: torch.Tensor, targets: torch.Tensor,
            mode: str, as_numpy: bool = False) -> dict:
        """Calculate loss per instance in a batch

        :param predictions: Predictions (Predicted)
        :type predictions: torch.Tensor
        :param targets: Targets (Ground Truth)
        :type targets: torch.Tensor
        :param mode: train/val/test mode
        :type mode: str
        :param as_numpy: flag to decide whether to return losses as np.ndarray
        :type as_numpy: bool

        :return: dict of losses with list of loss values per instance
        """
        loss_config = self.config['loss'][mode]
        criterion = loss_factory.create(
            loss_config['name'], **loss_config['params'])

        # correct data type to handle mismatch between
        # CrossEntropyLoss and BCEWithLogitsLoss
        if loss_config['name'] == 'cross-entropy':
            targets = targets.long()

        loss = criterion(predictions, targets)

        if as_numpy:
            loss = loss.cpu().numpy()

        return {'loss': loss}
