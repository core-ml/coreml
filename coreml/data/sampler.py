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


class ClassificationDataSampler(DataSampler):
    """Custom sampler to decide the ordering of samples for classification

    :param dataset: the dataset object from which to sample
    :type dataset: :class:`~torch.utils.data.Dataset`
    :param shuffle: decides the functionality for the sampler,
        defaults to True
    :type shuffle: bool, optional
    :param seed: random seed to use for sampling, defaults to 0
    :type seed: int, optional
    :param target_transform: defines the transformation to be applied on the
        raw targets to make them processable; if label_index is provided,
        target_transform.transforms[label_index] is used instead; defaults to
        None
    :type target_transform: Any
    :param mode: mode of sampling; choices are [`default`, `balanced`]; for
        `default`, it matches the default sampling behaviour. For `balanced`,
        it ensures class balance per batch and drops the examples; defaults
        to `default`
    :type mode: str, optional
    """
    def __init__(self, dataset: Dataset, shuffle: bool = True, seed: int = 0,
                 target_transform: Any = None, mode: str = 'default'):
        super(ClassificationDataSampler, self).__init__(dataset, shuffle, seed)
        self._check_params(dataset, shuffle, target_transform, mode)
        self.mode = mode

        if mode == 'balanced':
            self.target_transform = target_transform
            labels = [item.label['classification'] for item in dataset.items]
            self.labels = np.array([target_transform(
                label) for label in labels])

            _, indices = np.unique(self.labels, return_inverse=True)

            # tracks the list of indices corresponding to each label
            self.label_indices_map = defaultdict(list)

            for index, class_index in enumerate(indices):
                self.label_indices_map[class_index].append(index)

            # tracks the minimum number of examples across classes
            self.min_count = min(
                [len(indices) for _, indices in self.label_indices_map.items()])
            self.load_fn = self.load_balanced

            # length = number of classes * min_count
            self.len = self.min_count * len(self.label_indices_map)

    def load_balanced(self):
        """
        Returns a list of indices with class balance per batch.
        It returns K * C indices where C is the number of classes and K
        is the minimum number of examples across classes.
        """
        if self.shuffle:
            for key in self.label_indices_map:
                random.shuffle(self.label_indices_map[key])

        indices = []

        for i in range(self.min_count):
            # need to use `sorted` here to ensure that the ordering of keys is
            # not affected by which key was created first
            indices.extend([subindices[i] for _, subindices in sorted(
                self.label_indices_map.items())])

        return indices

    @staticmethod
    def _check_params(dataset, shuffle, target_transform, mode):
        assert mode in ['default', 'balanced']
        if mode == 'default':
            return

        assert isinstance(dataset.items[0].label, dict)
        assert hasattr(target_transform, 'classes')


sampler_factory = Factory()
sampler_factory.register_builder('default', DataSampler)
sampler_factory.register_builder('classification', ClassificationDataSampler)
