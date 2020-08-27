"""Tests coreml.data.data_module.py"""
from os.path import join, exists
import multiprocessing as mp
import torch
import numpy as np
import unittest
from coreml.config import DATA_ROOT
from coreml.data.data_module import DataModule


class DataModuleTestCase(unittest.TestCase):
    """Class to check the creation of DataModule"""
    @classmethod
    def setUpClass(cls):
        if not exists(join(DATA_ROOT, 'CIFAR10')):
            subprocess.call(
                'python /workspace/coreml/coreml/data/process/CIFAR10.py',
                shell=True)

    def test_classification_data_module(self):
        """Test get_dataloader for classification"""
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
                'train': [
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
                ],
                'val': [
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
                ],
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
                'train': {
                    'name': 'default'
                },
                'val': {
                    'name': 'default'
                },
                'test': {
                    'name': 'default'
                }
            },
            'collate_fn': {
                'name': 'classification_collate'
            }
        }
        batch_size = 8
        data_module = DataModule(cfg, batch_size, mp.cpu_count())

        train_dataloader = data_module.train_dataloader()
        batch = next(iter(train_dataloader))
        signals, labels = batch['signals'], batch['labels']
        self.assertTrue(signals.shape, (batch_size, 3, 30, 30))

        val_dataloader = data_module.val_dataloader()
        batch = next(iter(val_dataloader))
        signals, labels = batch['signals'], batch['labels']
        self.assertTrue(signals.shape, (batch_size, 3, 30, 30))

        test_dataloader = data_module.test_dataloader()
        batch = next(iter(test_dataloader))
        signals, labels = batch['signals'], batch['labels']
        self.assertTrue(signals.shape, (batch_size, 3, 30, 30))


if __name__ == "__main__":
    unittest.main()
