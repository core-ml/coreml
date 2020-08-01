"""Tests pestdetection.data.image.Image"""
import unittest
from os.path import join
import torch
import cv2
import numpy as np
from coreml.data.vision.image import Image


class ImageTestCase(unittest.TestCase):
    """Class to run tests on Image"""
    @classmethod
    def setUpClass(cls):
        dummy_image = np.ones((100, 100, 3))
        cls.dummy_image_path = '/tmp/image.png'
        cv2.imwrite(cls.dummy_image_path, dummy_image)

    def setUp(self):
        self.image = Image(self.dummy_image_path)

    def test_image_numpy(self):
        """Test default functionality of Image.read()"""
        np_image = self.image.read()
        self.assertIsInstance(np_image, np.ndarray)

    def test_image_tensor(self):
        """Test Image.read() with as_tensor=True"""
        tensor_image = self.image.read(as_tensor=True)
        self.assertIsInstance(tensor_image, torch.Tensor)

    def test_label_non_dict(self):
        """Test creating Image object with wrong label format"""
        with self.assertRaises(AssertionError):
            image = Image(self.dummy_image_path, label='a')


if __name__ == "__main__":
    unittest.main()
