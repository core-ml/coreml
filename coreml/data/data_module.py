"""Defines DataModule class extending LightningDataModule"""
from typing import Dict
import pytorch_lightning as pl
from coreml.data.dataloader import get_dataloader


class DataModule(pl.LightningDataModule):
    """Extends LightningDataModule

    :param data_config: config specifying the params of each dataloader
    :type cfg: Dict
    :param batch_size: number of instances in each batch
    :type batch_size: int
    :param num_workers: number of cpu workers to use
    :type num_workers: int
    :param train_mode: key for the train data split, defaults to 'train'
    :type train_mode: str, optional
    :param val_mode: key for the validation data split, defaults to 'val'
    :type val_mode: str, optional
    :param test_mode: key for the test data split, defaults to 'test'
    :type test_mode: str, optional
    """
    def __init__(
            self, data_config: Dict, batch_size: int, num_workers: int,
            train_mode: str = 'train', val_mode: str = 'val',
            test_mode: str = 'test'):
        super().__init__()
        self.data_config = data_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_mode = train_mode
        self.val_mode = val_mode
        self.test_mode = test_mode

    def train_dataloader(self):
        return get_dataloader(
            self.data_config, self.train_mode,
            self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=False)

    def val_dataloader(self):
        return get_dataloader(
            self.data_config, self.val_mode,
            self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False)

    def test_dataloader(self):
        return get_dataloader(
            self.data_config, self.test_mode,
            self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False)
