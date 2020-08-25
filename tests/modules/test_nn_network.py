"""Tests coreml.modules.nn.NeuralNetworkModule network"""
import torch
import unittest
from copy import deepcopy
from torch.nn import Conv2d, BatchNorm2d, ReLU
from numpy.testing import assert_array_equal
from coreml.modules.nn import NeuralNetworkModule


class NeuralNetworkModuleNetworkTestCase(unittest.TestCase):
    """Class to check the creation of NeuralNetworkModule's network"""
    @classmethod
    def setUpClass(cls):
        network_config = [
            {
                "name": "Conv2d",
                "params": {
                    "in_channels": 1,
                    "out_channels": 16,
                    "kernel_size": [3, 7]
                }
            },
            {
                "name": "BatchNorm2d",
                "params": {
                    "num_features": 16
                }
            },
            {
                "name": "ReLU",
                "params": {}
            },
            {
                "name": "AdaptiveAvgPool2d",
                "params": {
                    'output_size': [1, 1]
                }
            },
            {
                "name": "Flatten",
                "params": {}
            },
            {
                "name": "Linear",
                "params": {
                    "in_features": 16,
                    "out_features": 2
                }
            },
        ]
        cls.config = {
            'network': {
                'config': network_config
            }
        }

    def test_cnn_creation(self):
        """Test creation of a CNN using NeuralNetworkModule"""
        network = NeuralNetworkModule(self.config)
        self.assertEqual(
            len(self.config['network']['config']), len(network.blocks))

        # test Conv2d
        self.assertIsInstance(network.blocks[0], Conv2d)
        self.assertEqual(network.blocks[0].in_channels, 1)
        self.assertEqual(network.blocks[0].out_channels, 16)
        self.assertEqual(network.blocks[0].kernel_size, [3, 7])

        # test BatchNorm2d
        self.assertIsInstance(network.blocks[1], BatchNorm2d)
        self.assertEqual(network.blocks[1].num_features, 16)

        # test ReLU
        self.assertIsInstance(network.blocks[2], ReLU)

    def test_init(self):
        """Test weight initialization of a NeuralNetworkModule"""
        network_init = {
            "weight": {
                "name": "constant",
                "params": {
                    "val": 0
                }
            },
            "bias": {
                "name": "constant",
                "params": {
                    "val": 0
                }
            },
            "bn_weight": {
                "name": "constant",
                "params": {
                    "val": 0
                }
            },
            "bn_bias": {
                "name": "constant",
                "params": {
                    "val": 0
                }
            }
        }

        self.init_config = deepcopy(self.config)
        self.init_config['network']['init'] = network_init

        network = NeuralNetworkModule(self.init_config)

        for m in network.modules():
            if isinstance(m, BatchNorm2d) or isinstance(m, Conv2d):
                assert_array_equal(m.weight.detach().numpy(), 0)
                assert_array_equal(m.bias.detach().numpy(), 0)

        network = NeuralNetworkModule(self.config)

        for m in network.modules():
            if isinstance(m, BatchNorm2d):
                assert_array_equal(m.weight.detach().numpy(), 1)
                assert_array_equal(m.bias.detach().numpy(), 0)

    def test_freeze_layers(self):
        """Test freezing layers of a network"""
        config = deepcopy(self.config)
        network_config = config['network']['config']

        # freeze the first 2 layers
        for index in [0, 1]:
            network_config[index]['requires_grad'] = False

        network = NeuralNetworkModule(config)
        network.freeze()
        for name, param in network.named_parameters():
            if '0_' in name or '1_' in name:
                self.assertEqual(param.requires_grad, False)
            else:
                self.assertEqual(param.requires_grad, True)
        dummy = torch.zeros(4, 1, 128, 40)
        out = network(dummy)
        self.assertEqual(out.shape, torch.Size([4, 2]))

    def test_freeze_backbone_layers(self):
        """Test freezing layers of backbone in a network"""
        config = deepcopy(self.config)
        network_config = config['network']['config']

        # set the backbone as resnet18
        network_config[:2] = [
            {
                "name": "resnet18",
                "params": {
                    "in_channels": 1,
                    "pretrained": True,
                }
            },

            {
                "name": "BatchNorm2d",
                "params": {
                    "num_features": 512
                }
            }
        ]

        network_config[-1] = {
            "name": "Linear",
            "params": {
                "in_features": 512,
                "out_features": 2
            }
        }

        # freeze the first 2 layers
        for index in [0, 1]:
            network_config[index]['requires_grad'] = False

        network = NeuralNetworkModule(config)
        network.freeze()
        for name, param in network.named_parameters():
            if '0_' in name or '1_' in name:
                self.assertEqual(param.requires_grad, False)
            else:
                self.assertEqual(param.requires_grad, True)
        dummy = torch.zeros(4, 1, 128, 40)
        out = network(dummy)
        self.assertEqual(out.shape, torch.Size([4, 2]))

    def test_cnn_forward(self):
        """Test forward pass of a CNN using NeuralNetworkModule"""
        network = NeuralNetworkModule(self.config)
        dummy = torch.zeros(4, 1, 128, 40)
        out = network(dummy)
        self.assertEqual(out.shape, torch.Size([4, 2]))


if __name__ == "__main__":
    unittest.main()
