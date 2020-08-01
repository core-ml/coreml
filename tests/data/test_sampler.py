"""Tests coreml.data.sampler.DataSampler"""
# pylint: disable=no-member,invalid-name
import unittest
from torch.utils.data import Dataset
from numpy.testing import assert_array_equal
from coreml.data.sampler import DataSampler


class DummyDataset(Dataset):
    """Defines a dummy dataset for testing"""
    def __init__(self):
        self.len = 10

    def __getitem__(self, index):
        return index

    def __len__(self):
        return self.len


class DataSamplerTestCase(unittest.TestCase):
    """Class to run tests on DataSampler"""
    @classmethod
    def setUpClass(cls):
        cls.dataset = DummyDataset()

    def test_no_shuffle(self):
        """Test sampling without shuffling"""
        sampler = DataSampler(
            self.dataset, shuffle=False)
        indices = list(sampler)
        true_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        # without shuffle=True, the ordering should be the default ordering
        self.assertEqual(indices, true_indices)

    def test_with_shuffle(self):
        """Test sampling with shuffling"""
        sampler = DataSampler(
            self.dataset, shuffle=True)
        indices = list(sampler)
        true_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        # with shuffle=True, the ordering should be different from the
        # default ordering
        self.assertNotEqual(indices, true_indices)


if __name__ == "__main__":
    unittest.main()
