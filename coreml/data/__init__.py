"""Defines Factory object to register various datasets"""
from coreml.factory import Factory
from coreml.data.classification import ClassificationDatasetBuilder

dataset_factory = Factory()
dataset_factory.register_builder(
    "classification_dataset", ClassificationDatasetBuilder())
