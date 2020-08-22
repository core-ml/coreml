"""Contains functions for loading data"""
# pylint: disable=no-member
import logging
from functools import partial
from collections import defaultdict
from typing import Tuple, Dict, List
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from coreml.data import dataset_factory
from coreml.data.sampler import sampler_factory
from coreml.data.transforms import DataProcessor, annotation_factory
from coreml.utils.logger import color


def classification_collate(batch: Tuple[Dict]) -> Dict:
    """Collate function for classification model

    :param batch: A tuple of dicts of processed signals, labels and items
    :type batch: Tuple[Dict]
    :returns: A dict containing:
        1) tensor, batch of processed signals
        2) tensor, batch of corresponding labels
        3) list, items corresponding to each instance
    """
    signals = []
    labels = []
    items = []

    for data_point in batch:
        signals.append(data_point['signal'])
        labels.append(data_point['label'])
        items.append(data_point['item'])

    collated_batch = {
        'signals': torch.stack(signals),
        'labels': torch.Tensor(labels),
        'items': items
    }

    return collated_batch


def get_dataloader(
        cfg: Dict, mode: str, batch_size: int,
        num_workers: int = 10, shuffle: bool = True, drop_last: bool = True
        ) -> DataLoader:
    """Creates the DataLoader

    :param cfg: config specifying the dataloader
    :type cfg: Dict
    :param mode: mode/split to load; one of {'train', 'test', 'val'}
    :type mode: str
    :param batch_size: number of instances in each batch
    :type batch_size: int
    :param num_workers: number of cpu workers to use, defaults to 10
    :type num_workers: int
    :param shuffle: whether to shuffle the data, defaults to True
    :type shuffle: bool, optional
    :param drop_last: whether to include last batch containing sample
        less than the batch size, defaults to True
    :type drop_last: bool, optional
    :returns: the DataLoader object
    """
    logging.info(color('Creating {} DataLoader'.format(mode), 'blue'))

    # define target transform
    target_transform = None
    if 'target_transform' in cfg:
        target_transform = annotation_factory.create(
            cfg['target_transform']['name'],
            **cfg['target_transform']['params'])

    # define signal transform
    signal_transform = None
    if 'signal_transform' in cfg:
        signal_transform = DataProcessor(cfg['signal_transform'][mode])

    # define Dataset object
    dataset_params = cfg['dataset']['params'].get(mode, {})

    dataset_params.update({
        'target_transform': target_transform,
        'signal_transform': signal_transform,
        'mode': mode,
        'data_type': cfg['data_type'],
        'data_root': cfg['root'],
        'dataset_config': cfg['dataset']['config']
    })

    dataset = dataset_factory.create(cfg['dataset']['name'], **dataset_params)

    # to load entire dataset in one batch
    if batch_size == -1:
        batch_size = len(dataset)

    # define sampler
    sampler_cfg = cfg['sampler'].get(mode, {'name': 'default'})
    sampler_params = sampler_cfg.get('params', {})
    sampler_params.update({
        'dataset': dataset,
        'shuffle': shuffle,
        'target_transform': target_transform
    })
    sampler = sampler_factory.create(sampler_cfg['name'], **sampler_params)

    # define the collate function for accumulating a batch
    collate_fn = partial(eval(cfg['collate_fn']['name']),
                         **cfg['collate_fn'].get('params', {}))

    # define DataLoader object
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=collate_fn,
        pin_memory=True)

    return dataloader
