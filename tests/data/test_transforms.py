"""Tests coreml.data.transforms.DataProcessor"""
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

    def test_subtract_mean_dims_aligned(self):
        """Tests subtract mean dims already aligned"""
        dummy = torch.tensor([[0.1, 0.5, 0.6], [0.2, 0.4, 0.3]])
        mean = [0.1, 0.2, 0.2]

        config = [
            {
                'name': 'SubtractMean',
                'params': {
                    'mean': mean
                }
            }
        ]
        processor = DataProcessor(config)

        t_dummy = processor(dummy)
        target = torch.tensor([[0.0, 0.3, 0.4], [0.1, 0.2, 0.1]])
        assert_array_almost_equal(target, t_dummy)

    def test_subtract_mean_align_dim(self):
        """Tests subtract mean with dim specified"""
        dummy = torch.tensor([[0.1, 0.5, 0.6], [0.2, 0.4, 0.3]]).T
        mean = [0.1, 0.2, 0.2]

        config = [
            {
                'name': 'SubtractMean',
                'params': {
                    'mean': mean,
                    'dim': 0
                }
            }
        ]
        processor = DataProcessor(config)

        t_dummy = processor(dummy)
        target = torch.tensor([[0.0, 0.3, 0.4], [0.1, 0.2, 0.1]]).T
        assert_array_almost_equal(target, t_dummy)

    def test_random_vertical_flip_2d(self):
        """Tests RandomVerticalFlip on 2D input"""
        dummy = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]])  # 3 x 3
        expected = torch.tensor([[0., 1., 1.],
                                 [0., 0., 0.],
                                 [0., 0., 0.]])  # 3 x 3

        config = [
            {
                'name': 'RandomVerticalFlip',
                'params': {
                    'p': 1.
                }
            }
        ]
        processor = DataProcessor(config)

        t_dummy = processor(dummy)
        assert_array_almost_equal(t_dummy, expected)

        config = [
            {
                'name': 'RandomVerticalFlip',
                'params': {
                    'p': 0.
                }
            }
        ]
        processor = DataProcessor(config)

        t_dummy = processor(dummy)
        assert_array_almost_equal(dummy, t_dummy)

    def test_random_vertical_flip_3d_1_channel(self):
        """Tests RandomVerticalFlip on 3D input with 1 channel"""
        dummy = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]])     # 3 x 3
        dummy = dummy.unsqueeze(0)               # 1 x 3 x 3
        expected = torch.tensor([[0., 1., 1.],
                                 [0., 0., 0.],
                                 [0., 0., 0.]])  # 3 x 3
        expected = expected.unsqueeze(0)         # 1 x 3 x 3

        config = [
            {
                'name': 'RandomVerticalFlip',
                'params': {
                    'p': 1.
                }
            }
        ]
        processor = DataProcessor(config)

        t_dummy = processor(dummy)
        assert_array_almost_equal(t_dummy, expected)

    def test_random_vertical_flip_3d_2_channels(self):
        """Tests RandomVerticalFlip on 3D input with 2 channels"""
        dummy = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]])           # 3 x 3
        dummy = dummy.unsqueeze(0).expand(2, -1, -1)   # 2 x 3 x 3
        expected = torch.tensor([[0., 1., 1.],
                                 [0., 0., 0.],
                                 [0., 0., 0.]])             # 3 x 3
        expected = expected.unsqueeze(0).expand(2, -1, -1)  # 2 x 3 x 3

        config = [
            {
                'name': 'RandomVerticalFlip',
                'params': {
                    'p': 1.
                }
            }
        ]
        processor = DataProcessor(config)

        t_dummy = processor(dummy)
        assert_array_almost_equal(t_dummy, expected)

    def test_random_horizontal_flip_2d(self):
        """Tests RandomHorizontalFlip on 2D input"""
        dummy = torch.tensor([[1., 0., 0.],
                              [1., 0., 0.],
                              [0., 0., 0.]])  # 3 x 3
        expected = torch.tensor([[0., 0., 1.],
                                 [0., 0., 1.],
                                 [0., 0., 0.]])  # 3 x 3

        config = [
            {
                'name': 'RandomHorizontalFlip',
                'params': {
                    'p': 1.
                }
            }
        ]
        processor = DataProcessor(config)

        t_dummy = processor(dummy)
        assert_array_almost_equal(t_dummy, expected)

        config = [
            {
                'name': 'RandomHorizontalFlip',
                'params': {
                    'p': 0.
                }
            }
        ]
        processor = DataProcessor(config)

        t_dummy = processor(dummy)
        assert_array_almost_equal(dummy, t_dummy)

    def test_random_horizontal_flip_3d_1_channel(self):
        """Tests RandomHorizontalFlip on 3D input with 1 channel"""
        dummy = torch.tensor([[1., 0., 0.],
                              [1., 0., 0.],
                              [0., 0., 0.]])  # 3 x 3
        dummy = dummy.unsqueeze(0)               # 1 x 3 x 3
        expected = torch.tensor([[0., 0., 1.],
                                 [0., 0., 1.],
                                 [0., 0., 0.]])  # 3 x 3
        expected = expected.unsqueeze(0)         # 1 x 3 x 3

        config = [
            {
                'name': 'RandomHorizontalFlip',
                'params': {
                    'p': 1.
                }
            }
        ]
        processor = DataProcessor(config)

        t_dummy = processor(dummy)
        assert_array_almost_equal(t_dummy, expected)

    def test_random_horizontal_flip_3d_2_channels(self):
        """Tests RandomHorizontalFlip on 3D input with 2 channels"""
        dummy = torch.tensor([[1., 0., 0.],
                              [1., 0., 0.],
                              [0., 0., 0.]])  # 3 x 3
        dummy = dummy.unsqueeze(0).expand(2, -1, -1)   # 2 x 3 x 3
        expected = torch.tensor([[0., 0., 1.],
                                 [0., 0., 1.],
                                 [0., 0., 0.]])  # 3 x 3
        expected = expected.unsqueeze(0).expand(2, -1, -1)  # 2 x 3 x 3

        config = [
            {
                'name': 'RandomHorizontalFlip',
                'params': {
                    'p': 1.
                }
            }
        ]
        processor = DataProcessor(config)

        t_dummy = processor(dummy)
        assert_array_almost_equal(t_dummy, expected)

    def test_random_erasing_2d(self):
        """Tests RandomErasing on 2D input"""
        dummy = torch.rand((3, 3))  # 3 x 3

        config = [
            {
                'name': 'RandomErasing',
                'params': {
                    'p': 1.,
                    'scale': (.001, .001),
                    'ratio': (.2, .2)
                }
            }
        ]
        processor = DataProcessor(config)

        t_dummy = processor(dummy)
        self.assertEqual(t_dummy.shape, dummy.shape)

        config = [
            {
                'name': 'RandomErasing',
                'params': {
                    'p': 0.
                }
            }
        ]
        processor = DataProcessor(config)

        t_dummy = processor(dummy)
        assert_array_almost_equal(dummy, t_dummy)


if __name__ == "__main__":
    unittest.main()
