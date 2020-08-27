"""Defines Factory object to register various optimizers"""
from typing import Generator
from torch.nn.parameter import Parameter
from torch.optim import Adam, SGD, AdamW, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CyclicLR, \
    OneCycleLR, MultiStepLR
from timm import optim
from coreml.factory import Factory


class Lookahead(optim.Lookahead):
    """Refer to timm/optim/lookahead.py for information on alpha and k

    :param params: parameters of the module whose weights are optimized
    :type params: Generator[Parameter, None, None]
    :param optimizer: Dictionary containing the name of the optimizer
        and the optimizer params
    :type optimizer: dict
    """
    def __init__(
            self, params: Generator[Parameter, None, None], optimizer: dict,
            alpha=0.5, k=6):
        optimizer['args']['params'] = params
        optimizer = optimizer_factory.create(
            optimizer['name'], **optimizer['args'])
        super(Lookahead, self).__init__(optimizer, alpha, k)


optimizer_factory = Factory()
optimizer_factory.register_builder('SGD', SGD)
optimizer_factory.register_builder('Adam', Adam)
optimizer_factory.register_builder('AdamW', AdamW)
optimizer_factory.register_builder('RAdam', optim.RAdam)
optimizer_factory.register_builder('Lookahead', Lookahead)

scheduler_factory = Factory()
scheduler_factory.register_builder('ReduceLROnPlateau', ReduceLROnPlateau)
scheduler_factory.register_builder('StepLR', StepLR)
scheduler_factory.register_builder('MultiStepLR', MultiStepLR)
scheduler_factory.register_builder('CyclicLR', CyclicLR)
scheduler_factory.register_builder('1cycle', OneCycleLR)
