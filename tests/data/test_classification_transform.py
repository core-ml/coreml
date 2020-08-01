"""Tests coreml.data.transforms.ClassificationAnnotationTransform"""
import unittest
from coreml.data.transforms import ClassificationAnnotationTransform


class ClassificationAnnotationTransformTestCase(unittest.TestCase):
    """Class to run tests on ClassificationAnnotationTransform"""
    @classmethod
    def setUpClass(cls):
        binary_classes = ['x', 'y']
        multi_classes = ['x', 'y', 'z']
        cls.binary_transform = ClassificationAnnotationTransform(
            binary_classes)
        cls.multi_transform = ClassificationAnnotationTransform(multi_classes)

    def test_empty_target(self):
        """Checks the case when input target is empty"""
        target = []
        with self.assertRaises(ValueError):
            transform_target = self.binary_transform(target)

    def test_no_intersection_binary(self):
        """Checks the case when input target has no intersection"""
        target = ['a', 'b']
        with self.assertRaises(ValueError):
            transform_target = self.binary_transform(target)

    def test_one_intersection_binary(self):
        """Tests target has one intersection with binary_transform"""
        target = ['y']
        transform_target = self.binary_transform(target)
        self.assertEqual(transform_target, 1)

    def test_one_intersection_multi(self):
        """Tests when target has one intersection with multi_transform"""
        target = ['z', 'a']
        transform_target = self.multi_transform(target)
        self.assertEqual(transform_target, 2)

    def test_multiple_intersection_multi(self):
        """Tests when target has multiple intersections with multi_transform"""
        target = ['y', 'z']
        with self.assertRaises(ValueError):
            transform_target = self.multi_transform(target)


if __name__ == "__main__":
    unittest.main()
