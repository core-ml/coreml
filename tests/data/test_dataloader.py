"""Tests coreml.data.dataloader.py"""
from os.path import join, exists
import torch
import numpy as np
import unittest
from coreml.config import DATA_ROOT
from coreml.data.dataloader import get_dataloader


class DataloaderTestCase(unittest.TestCase):
    """Class to check the creation of DataLoader"""
    @classmethod
    def setUpClass(cls):
        if not exists(join(DATA_ROOT, 'CIFAR10')):
            subprocess.call(
                'python /workspace/coreml/tasks/data/classification/CIFAR10.py',
                shell=True)

    def test_classification_dataloader_2d(self):
        """Test get_dataloader for classification with each input being 2D"""
        cfg = {
            'root': DATA_ROOT,
            'data_type': 'image',
            'dataset': {
                'name': 'classification_dataset',
                'params': {
                    'test': {
                        'fraction': 0.1
                    }
                },
                'config': [
                    {
                        'name': 'CIFAR10',
                        'version': 'default',
                        'mode': 'test'
                    }
                ]
            },
            'target_transform': {
                'name': 'classification',
                'params': {
                    'classes': [
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
                    ]
                }
            },
            'signal_transform': {
                'test': [
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
            },
            'sampler': {
                'test': {
                    'name': 'default'
                }
            },
            'collate_fn': {
                'name': 'classification_collate'
            }
        }
        batch_size = 8

        dataloader, _ = get_dataloader(
            cfg, 'test', batch_size=batch_size, shuffle=False, drop_last=False)

        iterator = iter(dataloader)
        batch = next(iterator)
        signals, labels = batch['signals'], batch['labels']

        self.assertIsInstance(signals, torch.Tensor)
        self.assertIsInstance(labels, torch.Tensor)
        self.assertEqual(signals.dtype, torch.float32)
        self.assertEqual(labels.dtype, torch.float32)
        self.assertEqual(len(signals), len(labels))
        self.assertEqual(len(signals.shape), 4)
        self.assertTrue(signals.shape, (batch_size, 3, 30, 30))


if __name__ == "__main__":
    unittest.main()
