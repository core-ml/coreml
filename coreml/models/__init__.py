"""Defines Factory object to register various models"""
from coreml.factory import Factory
from coreml.models.classification import ClassificationModelBuilder

factory = Factory()
factory.register_builder(
    'classification', ClassificationModelBuilder())
