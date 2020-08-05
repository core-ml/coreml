"""Defines Factory object to register various layers"""
from typing import Any, List
import math
import torch
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn import Conv2d, Linear, BatchNorm2d, ReLU,\
    LeakyReLU, MaxPool2d, AdaptiveAvgPool2d, Flatten, Dropout,\
    Sigmoid, Conv1d, BatchNorm1d, MaxPool1d, AdaptiveAvgPool1d, \
    GroupNorm, PReLU, Module, Softmax
from coreml.factory import Factory
from coreml.networks.backbones.resnet import resnet18, resnet34, resnet50, \
    resnet101, resnet152, resnext50_32x4d, resnext101_32x8d
from coreml.networks.backbones.vgg import vgg11, vgg13, vgg16, vgg19, \
    vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from coreml.networks.backbones.efficientnet import EfficientNet


class Swish(Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


factory = Factory()
factory.register_builder('Conv2d', Conv2d)
factory.register_builder('Linear', Linear)
factory.register_builder('BatchNorm2d', BatchNorm2d)
factory.register_builder('ReLU', ReLU)
factory.register_builder('PReLU', PReLU)
factory.register_builder('LeakyReLU', LeakyReLU)
factory.register_builder('MaxPool2d', MaxPool2d)
factory.register_builder('AdaptiveAvgPool2d', AdaptiveAvgPool2d)
factory.register_builder('Flatten', Flatten)
factory.register_builder('Dropout', Dropout)
factory.register_builder('Sigmoid', Sigmoid)
factory.register_builder('Softmax', Softmax)
factory.register_builder('Conv1d', Conv1d)
factory.register_builder('BatchNorm1d', BatchNorm1d)
factory.register_builder('MaxPool1d', MaxPool1d)
factory.register_builder('AdaptiveAvgPool1d', AdaptiveAvgPool1d)
factory.register_builder('GroupNorm', GroupNorm)
factory.register_builder('resnet18', resnet18)
factory.register_builder('resnet34', resnet34)
factory.register_builder('resnet50', resnet50)
factory.register_builder('resnet101', resnet101)
factory.register_builder('resnet152', resnet152)
factory.register_builder('resnext50_32x4d', resnext50_32x4d)
factory.register_builder('resnext101_32x8d', resnext101_32x8d)
factory.register_builder('vgg11', vgg11)
factory.register_builder('vgg13', vgg13)
factory.register_builder('vgg16', vgg16)
factory.register_builder('vgg19', vgg19)
factory.register_builder('vgg11_bn', vgg11_bn)
factory.register_builder('vgg13_bn', vgg13_bn)
factory.register_builder('vgg16_bn', vgg16_bn)
factory.register_builder('vgg19_bn', vgg19_bn)
factory.register_builder('Swish', Swish)
factory.register_builder('efficientnet', EfficientNet)
