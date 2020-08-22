"""Defines the factory object for all the initializers"""
from torch.nn.init import kaiming_uniform_, kaiming_normal_, ones_, zeros_, \
    normal_, constant_, xavier_uniform_, xavier_normal_
from coreml.factory import Factory

init_factory = Factory()
init_factory.register_builder('kaiming_uniform', kaiming_uniform_)
init_factory.register_builder('kaiming_normal', kaiming_normal_)
init_factory.register_builder('ones', ones_)
init_factory.register_builder('zeros', zeros_)
init_factory.register_builder('normal', normal_)
init_factory.register_builder('constant', constant_)
init_factory.register_builder('xavier_uniform', xavier_uniform_)
init_factory.register_builder('xavier_normal', xavier_normal_)
