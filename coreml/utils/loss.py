"""Defines loss functions"""
import torch.nn as nn
from coreml.factory import Factory

loss_factory = Factory()
loss_factory.register_builder('cross-entropy', nn.CrossEntropyLoss)
loss_factory.register_builder('binary-cross-entropy', nn.BCEWithLogitsLoss)
