"""Tests coreml.data.vision.base"""
import unittest
import subprocess
from os.path import join, exists
import torch
import cv2
import numpy as np
from coreml.config import DATA_ROOT
from coreml.data.vision.base import BaseImageDataset


class BaseImageDatasetTestCase(unittest.TestCase):
    """Class to run tests on BaseImageDataset"""
    @classmethod
    def setUpClass(cls):
        if not exists(join(DATA_ROOT, 'CIFAR10')):
            subprocess.call(
                'python /workspace/coreml/coreml/data/process/CIFAR10.py',
                shell=True)

    def test_image_classification_dataset(self):
        """Test default functionality of Image.read()"""
        dataset_config = [
            {
                'name': 'CIFAR10',
                'version': 'default',
                'mode': 'test'
            }
        ]
        dataset = BaseImageDataset(DATA_ROOT, dataset_config)
        self.assertEqual(len(dataset.items), 10000)


if __name__ == "__main__":
    unittest.main()
