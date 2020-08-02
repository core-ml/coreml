"""Tests cac.models.classification.ClassificationModel evaluation"""
import os
from os.path import dirname, join, exists
from copy import deepcopy
import torch
import wandb
import unittest
from tqdm import tqdm
import numpy as np
from coreml.config import Config
from coreml.utils.logger import set_logger, color
from coreml.data.dataloader import get_dataloader
from coreml.models.binary_classification import BinaryClassificationModel


class BinaryClassificationModelEvaluationTestCase(unittest.TestCase):
    """Class to check the evaluation of BinaryClassificationModel"""
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

    def test_2_evaluate(self):
        """Test model.evaluate()"""
        set_logger(join(self.cfg.log_dir, 'train.log'))

        tester_cfg = deepcopy(self.cfg)
        tester_cfg.model['load']['version'] = 'default'
        tester_cfg.model['load']['load_best'] = True
        model = BinaryClassificationModel(tester_cfg)
        dataloader, _ = get_dataloader(
            tester_cfg.data, 'val',
            tester_cfg.model['batch_size'],
            num_workers=4,
            shuffle=False,
            drop_last=False)
        model.evaluate(dataloader, 'val', False)


if __name__ == "__main__":
    unittest.main()
