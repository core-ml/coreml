"""Defines Factory object to register various LightningModules"""
from coreml.factory import Factory
from coreml.modules.nn import NeuralNetworkModule
from coreml.modules.classification import BinaryClassificationModule, \
    MultiClassClassificationModule

lm_factory = Factory()
lm_factory.register_builder('neural-net', NeuralNetworkModule)
lm_factory.register_builder(
    'binary-classification', BinaryClassificationModule)
lm_factory.register_builder(
    'multiclass-classification', MultiClassClassificationModule)
