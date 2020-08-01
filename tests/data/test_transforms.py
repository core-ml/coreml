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


if __name__ == "__main__":
    unittest.main()
