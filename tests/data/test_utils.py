"""Tests coreml.data.utils"""
import unittest
import subprocess
from os.path import join, exists
import torch
from coreml.config import DATA_ROOT
from coreml.data.utils import read_dataset_from_config


class DataUtilTestCase(unittest.TestCase):
    """Class to run tests on coreml.data.utils.py"""
    @classmethod
    def setUpClass(cls):
        if not exists(join(DATA_ROOT, 'CIFAR10')):
            subprocess.call(
                'python /workspace/coreml/coreml/data/process/CIFAR10.py',
                shell=True)

    def test_read_dataset_from_config(self):
        dataset_config = {
            'name': 'CIFAR10',
            'version': 'default',
            'mode': 'test'
        }
        dataset_info = read_dataset_from_config(DATA_ROOT, dataset_config)

        self.assertIn('file', dataset_info.keys())
        self.assertIn('label', dataset_info.keys())


if __name__ == "__main__":
    unittest.main()
