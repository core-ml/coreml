"""Tests coreml.optimizer"""
import torch
import unittest
from coreml.modules.nn import NeuralNetworkModule
from coreml.modules.optimization import optimizer_factory


class OptimizerTestCase(unittest.TestCase):
    """Class to check the creation of Optimizer"""
    @classmethod
    def setUpClass(cls):
        cfg = {
            'network': {
                'config': [
                    {
                        'name': 'Conv2d',
                        'params': {
                            "in_channels": 1,
                            "out_channels": 16,
                            "kernel_size": [3, 7]
                        }
                    },
                    {
                        'name': 'BatchNorm2d',
                        'params': {
                            "num_features": 16
                        }
                    },
                    {
                        'name': 'LeakyReLU',
                        'params': {}
                    }
                ]
            }
        }
        cls.network = NeuralNetworkModule(cfg)

    def test_adam(self):
        """Test creation of a Adam optimizer"""
        optimizer_name = 'Adam'
        optimizer_args = {
            'params': self.network.parameters(),
            'lr': 0.0003,
            'weight_decay': 0.0005
        }

        optimizer = optimizer_factory.create(optimizer_name, **optimizer_args)
        self.assertTrue(optimizer_name in optimizer.__str__())

    def test_sgd(self):
        """Test creation of a SGD optimizer"""
        optimizer_name = 'SGD'
        optimizer_args = {
            'params': self.network.parameters(),
            'lr': 0.0003,
            'weight_decay': 0.0005
        }

        optimizer = optimizer_factory.create(optimizer_name, **optimizer_args)
        self.assertTrue(optimizer_name in optimizer.__str__())

    def test_lookahead(self):
        """Test creation of a LookAhead + RAdam optimizer"""
        optimizer_name = 'RAdam'
        optimizer_args = {
            'lr': 0.0003,
            'weight_decay': 0.0005
        }

        lookahead_cfg = {
            'params': self.network.parameters(),
            'optimizer': {
                'name': optimizer_name,
                'args': optimizer_args
            }
        }

        optimizer = optimizer_factory.create('Lookahead', **lookahead_cfg)
        self.assertTrue('Lookahead' in optimizer.__str__())


if __name__ == "__main__":
    unittest.main()
