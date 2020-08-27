"""Tests coreml.data.classification.ClassificationDataset"""
import unittest
from os.path import join, exists
import subprocess
import torch
import numpy as np
from coreml.config import DATA_ROOT
from coreml.data.classification import get_classification_dataset
from coreml.data.vision.base import BaseImageDataset
from coreml.data.transforms import DataProcessor, \
    ClassificationAnnotationTransform


class ClassificationDatasetTestCase(unittest.TestCase):
    """Class to run tests on ClassificationDataset"""
    @classmethod
    def setUpClass(cls):
        if not exists(join(DATA_ROOT, 'CIFAR10')):
            subprocess.call(
                'python /workspace/coreml/coreml/data/process/CIFAR10.py',
                shell=True)

        classes = np.arange(10).tolist()
        cls.target_transform = ClassificationAnnotationTransform(classes)

        cls.dataset_config = [
            {
                'name': 'CIFAR10',
                'version': 'default',
                'mode': 'test'
            }
        ]

        transform_config = [
            {
                'name': 'Permute',
                'params': {
                    'order': [2, 0, 1]
                }
            },
            {
                'name': 'Resize',
                'params': {
                    'size': [30, 30]
                }
            }
        ]
        cls.signal_transform = DataProcessor(transform_config)

    def test_different_modes(self):
        """Test ClassificationDataset object for different modes"""
        test_dataset_config = {
            'name': 'CIFAR10',
            'version': 'default',
            'mode': 'test'
        }
        train_dataset_config = {
            'name': 'CIFAR10',
            'version': 'default',
            'mode': 'train'
        }
        test_dataset = get_classification_dataset(BaseImageDataset)(
            DATA_ROOT, [test_dataset_config])
        train_dataset = get_classification_dataset(BaseImageDataset)(
            DATA_ROOT, [train_dataset_config])

        self.assertTrue(len(test_dataset.items) != len(train_dataset.items))

    def test_fraction(self):
        """Test creating ClassificationDataset object using fraction < 1"""
        dataset_config = {
            'name': 'CIFAR10',
            'version': 'default',
            'mode': 'test'
        }
        fraction = 0.5
        dataset = get_classification_dataset(BaseImageDataset)(
            DATA_ROOT, [dataset_config], fraction=fraction)
        self.assertEqual(5000, len(dataset.items))

    def test_dataset_no_transform(self):
        """Checks dataset using no transform"""
        dataset = get_classification_dataset(BaseImageDataset)(
            DATA_ROOT, self.dataset_config)

        instance = dataset[0]
        self.assertEqual(
            instance['item'].path,
            '/data/CIFAR10/processed/images/50000.png')
        self.assertTrue(isinstance(instance['signal'], torch.Tensor))
        self.assertEqual(instance['label'], [3])

    def test_dataset_with_target_transform(self):
        """Checks dataset with target transform"""
        dataset = get_classification_dataset(BaseImageDataset)(
            DATA_ROOT,
            self.dataset_config,
            target_transform=self.target_transform)

        instance = dataset[0]
        self.assertEqual(
            instance['item'].path,
            '/data/CIFAR10/processed/images/50000.png')
        self.assertTrue(isinstance(instance['signal'], torch.Tensor))
        self.assertEqual(instance['label'], 3)

    def test_dataset_with_signal_transform(self):
        """Checks dataset with signal transform"""
        dataset = get_classification_dataset(BaseImageDataset)(
            DATA_ROOT,
            self.dataset_config,
            signal_transform=self.signal_transform,
            target_transform=self.target_transform)

        instance = dataset[0]
        self.assertEqual(instance['signal'].shape, (3, 30, 30))


if __name__ == "__main__":
    unittest.main()
