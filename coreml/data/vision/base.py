"""Defines base dataset class for loading audio datasets."""
import random
from abc import abstractmethod
from typing import List

from tqdm import tqdm
from torch.utils.data import Dataset
from coreml.utils.typing import DatasetConfigDict
from coreml.data.vision.image import Image
from coreml.data.base_dataset import BaseDataset
from coreml.data.utils import read_dataset_from_config

random.seed(0)


class BaseImageDataset(BaseDataset):
    """
    Defines the base image dataset object that needs to be inherited
    by any task-specific dataset class for images.

    :param data_root: directory where data versions reside
    :type data_root: str
    :param dataset_config: defines the config for the data to be loaded.
        The config is specified by a list of dicts, with each dict representing
        (dataset_name, dataset_version, mode [train, test, val])
    :type dataset_config: DatasetConfigDict
    :param fraction: fraction of the data to load, defaults to 1.0
    :type fraction: float
    """
    def load_items(self):
        """Load all image data items as specified by self.dataset_config"""
        self.items = []

        for dataset_config in self.dataset_config:
            data_info = read_dataset_from_config(
                self.data_root, dataset_config)

            for i in tqdm(range(len(data_info['file'])), desc='Loading items'):
                path, label = data_info['file'][i], data_info['label'][i]

                image_item = Image(path=path, label=label)
                self.items.append(image_item)
