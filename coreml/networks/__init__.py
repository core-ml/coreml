"""Defines Factory object to register various networks"""
from coreml.factory import Factory
from coreml.networks.nn import NeuralNetworkBuilder

factory = Factory()
factory.register_builder('neural_net', NeuralNetworkBuilder())
