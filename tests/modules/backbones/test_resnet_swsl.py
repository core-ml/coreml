"""Tests coreml.models.backbones.resnet_swsl"""
import torch
import torch.nn as nn
import unittest
from coreml.networks.backbones.resnet_swsl import resnext50_32x4d_swsl


class ResnetTestCase(unittest.TestCase):
    """Class to check the Resnet-based backbones trained in SSL/SWSL manner"""
    def test_resnext50_32x4d_swsl(self):
        """Test resnext50_32x4d_swsl"""
        dummy = torch.empty((1, 3, 224, 224))
        net = resnext50_32x4d_swsl(pretrained=True)
        out = net(dummy)
        self.assertEqual(out.shape, (1, 2048, 7, 7))


if __name__ == "__main__":
    unittest.main()
