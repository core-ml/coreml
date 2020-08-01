"""Custom sampler for loading data"""
import random
from typing import List, Any
from collections import defaultdict
import numpy as np
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset
from coreml.factory import Factory


class DataSampler(Sampler):
    """Custom sampler to decide the ordering of samples within an epoch

    This retains the functionality of the default PyTorch sampler.
    Added here to serve as the base for adding more functionality.

    :param dataset: the dataset object from which to sample
    :type dataset: :class:`~torch.utils.data.Dataset`
    :param shuffle: decides the functionality for the sampler,
        defaults to True
    :type shuffle: bool, optional
    :param seed: random seed to use for sampling, defaults to 0
    :type seed: int, optional
    :param kwargs: additional params as dict
    :type kwargs: dict
    """
    def __init__(self, dataset: Dataset, shuffle: bool = True, seed: int = 0,
                 **kwargs):
        super(DataSampler, self).__init__(dataset)
        self.dataset = dataset
        self.shuffle = shuffle
        random.seed(seed)
        self.len = len(dataset)

    def load_fn(self):
        """Default behaviour as :class:`~torch.utils.sampler.Sampler`"""
        indices = np.arange(self.len)
        if self.shuffle:
            random.shuffle(indices)

        return indices

    def __iter__(self):
        return iter(self.load_fn())

    def __len__(self):
        return self.len


sampler_factory = Factory()
sampler_factory.register_builder('default', DataSampler)