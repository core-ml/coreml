"""Defines Factory object to register various models"""
from coreml.factory import Factory
from coreml.models.binary_classification import BinaryClassificationModel

factory = Factory()
factory.register_builder(
    'classification', BinaryClassificationModel)
