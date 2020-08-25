"""Tests coreml.utils.array"""
import unittest
import numpy as np
import wandb
import torch
from coreml.utils.array import is_scalar


class ArrayTestCase(unittest.TestCase):
    """Class to run tests on array util functions"""
    def test_is_scalar_float(self):
        """Checks the function is_scalar with float"""
        self.assertTrue(is_scalar(1.))

    def test_is_scalar_int(self):
        """Checks the function is_scalar with int"""
        self.assertTrue(is_scalar(1))

    def test_is_scalar_tensor_scalar(self):
        """Checks the function is_scalar with tensor scalar"""
        self.assertTrue(is_scalar(torch.tensor(1.)))

    def test_is_scalar_tensor_non_scalar(self):
        """Checks the function is_scalar with tensor non-scalar"""
        self.assertFalse(is_scalar(torch.Tensor([1.])))


if __name__ == "__main__":
    unittest.main()
