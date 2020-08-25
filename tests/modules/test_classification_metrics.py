"""Tests coreml.models.binary_classification compute_metrics"""
import os
from os.path import dirname, join, exists
from copy import deepcopy
import torch
import wandb
import unittest
from tqdm import tqdm
import numpy as np
from torch import optim
from coreml.config import Config
from coreml.utils.logger import set_logger, color
from coreml.modules.classification import BinaryClassificationModule


class ClassificationMetricsTestCase(unittest.TestCase):
    """Class to check the metric computation for classification"""
    @classmethod
    def setUpClass(cls):
        version = 'configs/defaults/binary-cifar.yml'
        cls.cfg = Config(version)

    def test_compute_metrics_threshold_none(self):
        """Tests no threshold specified"""
        classifier = BinaryClassificationModule(self.cfg.module['config'])

        predictions = torch.Tensor([1, -1, 0.5])
        targets = torch.Tensor([1, 0, 1])
        metrics = classifier.compute_epoch_metrics(predictions, targets)
        self.assertEqual(metrics['recall'], 1)
        self.assertEqual(metrics['precision'], 1)

    def test_compute_metrics_threshold_given(self):
        """Tests threshold specified"""
        classifier = BinaryClassificationModule(self.cfg.module['config'])

        predictions = torch.Tensor([1, -1, 0.5])
        targets = torch.Tensor([1, 0, 1])
        metrics = classifier.compute_epoch_metrics(
            predictions, targets, threshold=0.6)
        self.assertEqual(metrics['threshold'], 0.6)
        self.assertEqual(metrics['recall'], 1)
        self.assertEqual(metrics['precision'], 1)

    def test_compute_metrics_recall_none(self):
        """Tests minimum recall not specified"""
        classifier = BinaryClassificationModule(self.cfg.module['config'])

        predictions = torch.Tensor(
            [0.6, 0.4, 0.3, 0.1, 0.8, 0.9])
        targets = torch.Tensor([1, 0, 1, 0, 1, 1])
        metrics = classifier.compute_epoch_metrics(
            predictions, targets, as_logits=False)
        self.assertEqual(metrics['recall'], 1)

    def test_compute_metrics_recall_given(self):
        """Tests minimum recall specified"""
        classifier = BinaryClassificationModule(self.cfg.module['config'])

        predictions = torch.Tensor(
            [0.6, 0.4, 0.3, 0.1, 0.8, 0.9])
        targets = torch.Tensor([1, 0, 1, 0, 1, 1])
        metrics = classifier.compute_epoch_metrics(
            predictions, targets, recall=0.7)
        self.assertEqual(metrics['recall'],  0.75)


if __name__ == "__main__":
    unittest.main()
