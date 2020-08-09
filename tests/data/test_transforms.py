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

    def test_rescale_transform(self):
        """Checks Rescale transform"""
        config = [
            {
                'name': 'Rescale',
                'params': {'value': 255}
            }
        ]

        processor = DataProcessor(config)
        dummy_signal = torch.ones(100) * 255.
        transformed_signal = processor(dummy_signal)
        assert_array_equal(transformed_signal, 1.0)

    def test_random_rotation(self):
        """Tests RandomRotation"""
        dummy = torch.tensor([[1., 0., 0., 2.],
                              [0., 0., 0., 0.],
                              [0., 1., 2., 0.],
                              [0., 0., 1., 2.]])  # 4 x 4

        expected = torch.tensor(
            [[0.9824, 0.0088, 0.0000, 1.9649],
             [0.0000, 0.0029, 0.0000, 0.0176],
             [0.0029, 1.0000, 1.9883, 0.0000],
             [0.0000, 0.0088, 1.0117, 1.9649]])  # 1 x 4 x 4

        config = [
            {
                'name': 'RandomRotation',
                'params': {
                    'degrees': 45.
                }
            }
        ]
        torch.manual_seed(0)
        processor = DataProcessor(config)

        t_dummy = processor(dummy)
        assert_array_almost_equal(t_dummy, expected, decimal=4)

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
        dummy = torch.tensor(
            [[0.1, 0.5, 0.6], [0.2, 0.4, 0.3]]).T.unsqueeze(-1)
        mean = [0.1, 0.2, 0.2]

        config = [
            {
                'name': 'Normalize',
                'params': {
                    'mean': mean,
                    'std': [1, 1, 1]
                }
            }
        ]
        processor = DataProcessor(config)

        t_dummy = processor(dummy)
        target = torch.tensor(
            [[0.0, 0.3, 0.4], [0.1, 0.2, 0.1]]).T.unsqueeze(-1)
        assert_array_almost_equal(target, t_dummy)

    def test_subtract_mean_align_dim(self):
        """Tests subtract mean with dim specified"""
        dummy = torch.tensor([[0.1, 0.5, 0.6], [0.2, 0.4, 0.3]]).unsqueeze(-1)
        mean = [0.1, 0.2, 0.2]

        config = [
            {
                'name': 'Normalize',
                'params': {
                    'mean': mean,
                    'std': [1, 1, 1],
                    'dim': 1
                }
            }
        ]
        processor = DataProcessor(config)

        t_dummy = processor(dummy)
        target = torch.tensor([[0.0, 0.3, 0.4], [0.1, 0.2, 0.1]]).unsqueeze(-1)
        assert_array_almost_equal(target, t_dummy)

    def test_subtract_mean_imagenet(self):
        """Tests subtract mean from specified dataset"""
        dummy = torch.tensor(
            [[0.1, 0.5, 0.6], [0.2, 0.4, 0.3]]).T.unsqueeze(-1)
        mean = [0.485, 0.456, 0.406]

        config = [
            {
                'name': 'Normalize',
                'params': {
                    'mean': 'imagenet',
                    'std': [1, 1, 1]
                }
            }
        ]

        processor = DataProcessor(config)

        t_dummy = processor(dummy)
        target = torch.tensor([
            [-0.3850,  0.0440,  0.1940],
            [-0.2850, -0.0560, -0.1060]]).T.unsqueeze(-1)
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

    def test_colorjitter(self):
        """Tests ColorJiter"""
        torch.manual_seed(0)
        dummy = torch.rand(3, 5, 5)
        expected = torch.tensor([
         [[0.6863, 0.9709, 0.2014, 0.3054, 0.4860],
          [0.8358, 1.0000, 1.0000, 0.6502, 0.7154],
          [0.1962, 0.3835, 0.9929, 0.4433, 0.4729],
          [0.7237, 0.5893, 0.9653, 0.1961, 0.3046],
          [0.8852, 1.0000, 0.7024, 0.4536, 0.6027]],
         [[1.0000, 1.0000, 0.8778, 0.7666, 1.0000],
          [0.3821, 1.0000, 0.2783, 0.3303, 0.3239],
          [0.6277, 0.7232, 1.0000, 0.9284, 0.9818],
          [0.6332, 0.4211, 0.7802, 0.9193, 1.0000],
          [0.2849, 0.4944, 1.0000, 0.5254, 1.0000]],

         [[0.9540, 1.0000, 0.9027, 0.6902, 0.9508],
          [0.7502, 0.6776, 1.0000, 0.5409, 0.8339],
          [0.9316, 0.9055, 0.1823, 0.3414, 0.4848],
          [0.6290, 0.9019, 1.0000, 1.0000, 1.0000],
          [0.7763, 0.6276, 0.5818, 1.0000, 1.0000]]])

        config = [
            {
                'name': 'ColorJitter',
                'params': {
                    'brightness': 0.2,
                    'contrast': 0.2,
                    'hue': 0.2,
                    'saturation': 0.2
                }
            }
        ]
        processor = DataProcessor(config)
        t_dummy = processor(dummy)
        assert_array_almost_equal(t_dummy, expected, decimal=4)

    def test_randomaffine(self):
        """Tests RandomAffine"""
        torch.manual_seed(0)
        dummy = torch.rand(3, 5, 5)
        expected = torch.tensor([
         [[0.2763, 0.3109, 0.3721, 0.4599, 0.5413],
          [0.1560, 0.1800, 0.2358, 0.3223, 0.4009],
          [0.2102, 0.0790, 0.0983, 0.2009, 0.2835],
          [0.2832, 0.1709, 0.0971, 0.1024, 0.1890],
          [0.3726, 0.2669, 0.1878, 0.1354, 0.1174]],

         [[0.5799, 0.6210, 0.6119, 0.5527, 0.4610],
          [0.6795, 0.7326, 0.7336, 0.6579, 0.5626],
          [0.7319, 0.8646, 0.8574, 0.7499, 0.6526],
          [0.7694, 0.8488, 0.8583, 0.8301, 0.7307],
          [0.7562, 0.7516, 0.7588, 0.7776, 0.7972]],

         [[0.4247, 0.3421, 0.2600, 0.1785, 0.1672],
          [0.3765, 0.2933, 0.2111, 0.1983, 0.2205],
          [0.3562, 0.2546, 0.2259, 0.2647, 0.2896],
          [0.3591, 0.3045, 0.3194, 0.3469, 0.3747],
          [0.4069, 0.4114, 0.4245, 0.4462, 0.4756]]])

        config = [
            {
                'name': 'RandomAffine',
                'params': {
                    'degrees': 45,
                    'translate': [0.1, 0.1],
                    'scale': [1, 8],
                    'shear': [2, 2]
                }
            }
        ]
        processor = DataProcessor(config)
        t_dummy = processor(dummy)
        assert_array_almost_equal(t_dummy, expected, decimal=4)

    def test_random_crop_2d(self):
        """Tests RandomCrop on 2D input"""
        dummy = torch.rand((3, 3))  # 3 x 3

        config = [
            {
                'name': 'RandomCrop',
                'params': {
                    'size': [4, 4],
                    'padding': 2,
                    'padding_mode': 'reflect'
                }
            }
        ]
        processor = DataProcessor(config)

        t_dummy = processor(dummy)
        self.assertEqual(t_dummy.shape, (4, 4))


if __name__ == "__main__":
    unittest.main()
