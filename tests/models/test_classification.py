"""Tests coreml.models.binary_classification"""
import os
from os.path import dirname, join, exists
from copy import deepcopy
import torch
import unittest
import numpy as np
from torch import optim
from coreml.config import Config
from coreml.utils.logger import set_logger, color
from coreml.models.binary_classification import BinaryClassificationModel


class BinaryClassificationModelTestCase(unittest.TestCase):
    """Class to check the creation of BinaryClassificationModel"""
    @classmethod
    def setUpClass(cls):
        version = 'configs/defaults/binary-cifar-classification.yml'
        cls.cfg = Config(version)
        cls.cfg.data['dataset']['params'] = {
            'val': {
                'fraction': 0.1
            }
        }
        cls.cfg.num_workers = 10

    def test_1_model_fitting(self):
        """Test model.fit()"""
        set_logger(join(self.cfg.log_dir, 'train.log'))

        tester_cfg = deepcopy(self.cfg)
        tester_cfg.model['epochs'] = 1
        classifier = BinaryClassificationModel(tester_cfg)
        classifier.fit(debug=True, use_wandb=False)

    def test_optimizer(self):
        """Test model.fit()"""
        set_logger(join(self.cfg.log_dir, 'train.log'))

        tester_cfg = deepcopy(self.cfg)
        tester_cfg.model['epochs'] = 1
        classifier = BinaryClassificationModel(tester_cfg)
        self.assertIsInstance(classifier.optimizer, optim.SGD)
        self.assertIsInstance(
            classifier.scheduler, optim.lr_scheduler.ReduceLROnPlateau)


if __name__ == "__main__":
    unittest.main()
