"""Tests cac.data.transforms.DataProcessor"""
import unittest
import math
import numpy as np
import torch
from numpy.testing import assert_array_equal, assert_raises, \
    assert_array_almost_equal
from coreml.data.transforms import DataProcessor


class DataProcessorTestCase(unittest.TestCase):
    """Class to run tests on DataProcessor"""
    def test_resize_1d(self):
        """Checks Resize transform with 1D data"""
        target_size = (1, 1000)
        config = [
            {
                'name': 'Resize',
                'params': {'size': target_size}
            }
        ]

        processor = DataProcessor(config)
        dummy_input = torch.zeros(8000)
        transformed_input = processor(dummy_input)

        self.assertEqual(transformed_input.shape, (1000,))

    def test_resize_2d(self):
        """Checks Resize transform with 2D data"""
        target_size = (128, 20)
        config = [
            {
                'name': 'Resize',
                'params': {'size': target_size}
            }
        ]

        processor = DataProcessor(config)
        dummy_input = torch.zeros((128, 50))
        transformed_input = processor(dummy_input)

        self.assertEqual(transformed_input.shape, target_size)

    def test_resize_3d(self):
        """Checks Resize transform with 3D data"""
        target_size = (128, 20)
        config = [
            {
                'name': 'Resize',
                'params': {'size': target_size}
            }
        ]

        processor = DataProcessor(config)
        dummy_input = torch.zeros((2, 128, 50))
        transformed_input = processor(dummy_input)

        self.assertEqual(transformed_input.shape, (2, *target_size))

    def test_transpose(self):
        """Checks Transpose"""
        dummy = torch.ones((10, 20))
        config = [
            {
                'name': 'Transpose',
                'params': {
                    'dim0': 0,
                    'dim1': 1
                }
            }
        ]
        processor = DataProcessor(config)

        t_signal = processor(dummy)
        self.assertEqual(t_signal.shape, (20, 10))

    def test_permute(self):
        """Checks Permute"""
        dummy = torch.ones((10, 20, 3))
        config = [
            {
                'name': 'Permute',
                'params': {
                    'order': [2, 0, 1]
                }
            }
        ]
        processor = DataProcessor(config)
        t_signal = processor(dummy)
        self.assertEqual(t_signal.shape, (3, 10, 20))


if __name__ == "__main__":
    unittest.main()
