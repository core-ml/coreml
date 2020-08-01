"""Defines Factory object to register various datasets"""
from coreml.factory import Factory
from coreml.data.classification import ClassificationDatasetBuilder

factory = Factory()
factory.register_builder(
    "classification_dataset", ClassificationDatasetBuilder())
