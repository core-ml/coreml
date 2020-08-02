"""Tests coreml.networks.backbones.efficientnet"""
import torch
import torch.nn as nn
import unittest
from coreml.networks.backbones.efficientnet import efficientnet_b4, \
    efficientnet_b0, efficientnet_b7


class EfficientNetTestCase(unittest.TestCase):
    """Class to check the EfficientNet backbone"""
    def test_efficientnet_b4(self):
        """Test efficientnet_b4"""
        net = efficientnet_b4(num_classes=2, in_channels=1)
        dummy = torch.ones((128, 1, 96, 64))
        out = net(dummy)
        self.assertTrue(out.shape, (128, 2))

    def test_efficientnet_b0(self):
        """Test efficientnet_b0"""
        net = efficientnet_b0(num_classes=2, in_channels=1)
        dummy = torch.ones((128, 1, 96, 64))
        out = net(dummy)
        self.assertTrue(out.shape, (128, 2))

    def test_efficientnet_b7(self):
        """Test efficientnet_b7"""
        net = efficientnet_b7(num_classes=2, in_channels=1)
        dummy = torch.ones((128, 1, 96, 64))
        out = net(dummy)
        self.assertTrue(out.shape, (128, 2))


if __name__ == "__main__":
    unittest.main()
