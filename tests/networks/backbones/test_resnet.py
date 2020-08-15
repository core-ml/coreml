"""Tests cac.models.backbones.resnet"""
import torch
import torch.nn as nn
import unittest
from coreml.networks.backbones.resnet import resnet18, resnet34, resnet50, \
    resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, SeResnet


class ResnetTestCase(unittest.TestCase):
    """Class to check the Resnet-based backbones"""
    def test_resnet18(self):
        """Test resnet-18"""
        dummy = torch.empty((1, 3, 224, 224))
        net = resnet18(pretrained=True)
        out = net(dummy)
        self.assertEqual(out.shape, (1, 512, 7, 7))

    def test_resnet34(self):
        """Test resnet-34"""
        dummy = torch.empty((1, 3, 224, 224))
        net = resnet34(pretrained=True)
        out = net(dummy)
        self.assertEqual(out.shape, (1, 512, 7, 7))

    def test_resnet50(self):
        """Test resnet-50"""
        dummy = torch.empty((1, 3, 224, 224))
        net = resnet50(pretrained=True)
        out = net(dummy)
        self.assertEqual(out.shape, (1, 2048, 7, 7))

    def test_resnet101(self):
        """Test resnet-101"""
        dummy = torch.empty((1, 3, 224, 224))
        net = resnet101(pretrained=True)
        out = net(dummy)
        self.assertEqual(out.shape, (1, 2048, 7, 7))

    def test_resnet152(self):
        """Test resnet-152"""
        dummy = torch.empty((1, 3, 224, 224))
        net = resnet152(pretrained=True)
        out = net(dummy)
        self.assertEqual(out.shape, (1, 2048, 7, 7))

    def test_resnext50_32x4d(self):
        """Test resnext50_32x4d"""
        dummy = torch.empty((1, 3, 224, 224))
        net = resnext50_32x4d(pretrained=True)
        out = net(dummy)
        self.assertEqual(out.shape, (1, 2048, 7, 7))

    def test_resnext101_32x8d(self):
        """Test resnext101_32x8d"""
        dummy = torch.empty((1, 3, 224, 224))
        net = resnext101_32x8d(pretrained=True)
        out = net(dummy)
        self.assertEqual(out.shape, (1, 2048, 7, 7))

    def test_seresnet50(self):
        """Test seresnet-50"""
        dummy = torch.empty((1, 3, 224, 224))
        net = SeResnet(
            variant='seresnet50', num_classes=1, return_features=True)
        out = net(dummy)
        self.assertEqual(out.shape, (1, 2048, 7, 7))

    def test_resnet18_in_channels_1(self):
        """Test resnet-18 with in_channels=1"""
        dummy = torch.empty((1, 1, 224, 224))
        net = resnet18(in_channels=1, pretrained=True)
        out = net(dummy)
        self.assertEqual(out.shape, (1, 512, 7, 7))

    def test_resnet18_activation(self):
        """Test resnet-18 with different activation function"""
        activation = {
            'name': 'PReLU',
            'params': {}
        }
        net = resnet18(in_channels=1, pretrained=True, activation=activation)
        for name, module in net.named_modules():
            if name == 'activation':
                assert isinstance(module, nn.PReLU)

            if name == 'layer1.0':
                assert isinstance(module.activation, nn.PReLU)


if __name__ == "__main__":
    unittest.main()
