"""Defines Factory object to register various models"""
from coreml.factory import Factory
from coreml.models.binary_classification import BinaryClassificationModel
from coreml.models.multiclass_classification import \
    MultiClassClassificationModel

factory = Factory()
factory.register_builder(
    'classification', BinaryClassificationModel)
factory.register_builder(
    'multiclass-classification', MultiClassClassificationModel)
