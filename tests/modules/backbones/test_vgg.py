"""Tests cac.models.backbones.resnet"""
import torch
import unittest
from coreml.networks.backbones.vgg import vgg11, vgg13, vgg16, vgg19, \
    vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn


class VGGTestCase(unittest.TestCase):
    """Class to check the VGG-based backbones"""
    def test_vgg11(self):
        """Test vgg11"""
        dummy = torch.empty((1, 3, 224, 224))
        net = vgg11(pretrained=True)
        out = net(dummy)
        self.assertEqual(out.shape, (1, 512, 7, 7))

    def test_vgg13(self):
        """Test vgg13"""
        dummy = torch.empty((1, 3, 224, 224))
        net = vgg13(pretrained=True)
        out = net(dummy)
        self.assertEqual(out.shape, (1, 512, 7, 7))

    def test_vgg16(self):
        """Test vgg16"""
        dummy = torch.empty((1, 3, 224, 224))
        net = vgg16(pretrained=True)
        out = net(dummy)
        self.assertEqual(out.shape, (1, 512, 7, 7))

    def test_vgg19(self):
        """Test vgg19"""
        dummy = torch.empty((1, 3, 224, 224))
        net = vgg19(pretrained=True)
        out = net(dummy)
        self.assertEqual(out.shape, (1, 512, 7, 7))

    def test_vgg11_bn(self):
        """Test vgg11_bn"""
        dummy = torch.empty((1, 3, 224, 224))
        net = vgg11_bn(pretrained=True)
        out = net(dummy)
        self.assertEqual(out.shape, (1, 512, 7, 7))

    def test_vgg13_bn(self):
        """Test vgg13_bn"""
        dummy = torch.empty((1, 3, 224, 224))
        net = vgg13_bn(pretrained=True)
        out = net(dummy)
        self.assertEqual(out.shape, (1, 512, 7, 7))

    def test_vgg16_bn(self):
        """Test vgg16_bn"""
        dummy = torch.empty((1, 3, 224, 224))
        net = vgg16_bn(pretrained=True)
        out = net(dummy)
        self.assertEqual(out.shape, (1, 512, 7, 7))

    def test_vgg19_bn(self):
        """Test vgg19_bn"""
        dummy = torch.empty((1, 3, 224, 224))
        net = vgg19_bn(pretrained=True)
        out = net(dummy)
        self.assertEqual(out.shape, (1, 512, 7, 7))

    def test_vgg16_in_channels_1(self):
        """Test vgg16 with in_channels=1"""
        dummy = torch.empty((1, 1, 224, 224))
        net = vgg16(in_channels=1, pretrained=True)
        out = net(dummy)
        self.assertEqual(out.shape, (1, 512, 7, 7))


if __name__ == "__main__":
    unittest.main()
