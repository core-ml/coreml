"""Tests coreml.networks.activations"""
import unittest
import torch
import torch.nn.functional as F
from numpy.testing import assert_array_equal
from coreml.networks.activations import factory as activation_factory


class ActivationTestCase(unittest.TestCase):
    """Class to check the creation of activation functions"""
    def test_relu(self):
        """Test creation of a ReLU activation"""
        activation_name = 'ReLU'
        args = {}

        activation = activation_factory.create(activation_name, **args)
        self.assertEqual(activation._get_name(), activation_name)

        x = torch.ones(10) * -1
        y = activation(x)
        self.assertEqual(len(torch.nonzero(y, as_tuple=False)), 0)

    def test_prelu(self):
        """Test creation of a PReLU activation"""
        activation_name = 'PReLU'
        args = {}

        activation = activation_factory.create(activation_name, **args)
        self.assertEqual(activation._get_name(), activation_name)

    def test_leaky_relu(self):
        """Test creation of a LeakyReLU activation"""
        activation_name = 'LeakyReLU'
        args = {}

        activation = activation_factory.create(activation_name, **args)
        self.assertEqual(activation._get_name(), activation_name)

    def test_swish(self):
        """Test creation of a Swish activation"""
        activation_name = 'Swish'
        args = {}

        activation = activation_factory.create(activation_name, **args)
        self.assertEqual(activation._get_name(), activation_name)

        x = torch.empty(10)
        y = activation(x)
        assert_array_equal(y, x * torch.sigmoid(x))

    def test_sigmoid(self):
        """Test creation of a Sigmoid activation"""
        activation_name = 'Sigmoid'
        args = {}

        activation = activation_factory.create(activation_name, **args)
        self.assertEqual(activation._get_name(), activation_name)

        x = torch.empty(10)
        y = activation(x)
        assert_array_equal(y, torch.sigmoid(x))

    def test_softmax(self):
        """Test creation of a Softmax activation"""
        activation_name = 'Softmax'
        args = {}

        activation = activation_factory.create(activation_name, **args)
        self.assertEqual(activation._get_name(), activation_name)

        x = torch.empty(10, 2)
        y = activation(x)
        assert_array_equal(y, torch.softmax(x, -1))


if __name__ == "__main__":
    unittest.main()
