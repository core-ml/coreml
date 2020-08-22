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
from coreml.modules.backbones.resnet import resnet18, resnet34, resnet50, \
    resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, SeResnet
from coreml.modules.backbones.vgg import vgg11, vgg13, vgg16, vgg19, \
    vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from coreml.modules.backbones.resnet_swsl import resnext50_32x4d_swsl
from coreml.modules.backbones.efficientnet import EfficientNet


class Swish(Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


layer_factory = Factory()
layer_factory.register_builder('Conv2d', Conv2d)
layer_factory.register_builder('Linear', Linear)
layer_factory.register_builder('BatchNorm2d', BatchNorm2d)
layer_factory.register_builder('ReLU', ReLU)
layer_factory.register_builder('PReLU', PReLU)
layer_factory.register_builder('LeakyReLU', LeakyReLU)
layer_factory.register_builder('MaxPool2d', MaxPool2d)
layer_factory.register_builder('AdaptiveAvgPool2d', AdaptiveAvgPool2d)
layer_factory.register_builder('Flatten', Flatten)
layer_factory.register_builder('Dropout', Dropout)
layer_factory.register_builder('Sigmoid', Sigmoid)
layer_factory.register_builder('Softmax', Softmax)
layer_factory.register_builder('Conv1d', Conv1d)
layer_factory.register_builder('BatchNorm1d', BatchNorm1d)
layer_factory.register_builder('MaxPool1d', MaxPool1d)
layer_factory.register_builder('AdaptiveAvgPool1d', AdaptiveAvgPool1d)
layer_factory.register_builder('GroupNorm', GroupNorm)
layer_factory.register_builder('resnet18', resnet18)
layer_factory.register_builder('resnet34', resnet34)
layer_factory.register_builder('resnet50', resnet50)
layer_factory.register_builder('resnet101', resnet101)
layer_factory.register_builder('resnet152', resnet152)
layer_factory.register_builder('resnext50_32x4d', resnext50_32x4d)
layer_factory.register_builder('resnext101_32x8d', resnext101_32x8d)
layer_factory.register_builder('vgg11', vgg11)
layer_factory.register_builder('vgg13', vgg13)
layer_factory.register_builder('vgg16', vgg16)
layer_factory.register_builder('vgg19', vgg19)
layer_factory.register_builder('vgg11_bn', vgg11_bn)
layer_factory.register_builder('vgg13_bn', vgg13_bn)
layer_factory.register_builder('vgg16_bn', vgg16_bn)
layer_factory.register_builder('vgg19_bn', vgg19_bn)
layer_factory.register_builder('Swish', Swish)
layer_factory.register_builder('efficientnet', EfficientNet)
layer_factory.register_builder('resnext50_32x4d_swsl', resnext50_32x4d_swsl)
layer_factory.register_builder('seresnet', SeResnet)
