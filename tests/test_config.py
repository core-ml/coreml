"""Tests cac.config.Config"""
import unittest
from coreml.config import Config


class ConfigTestCase(unittest.TestCase):
    """Class to run tests on Config"""
    @classmethod
    def setUpClass(cls):
        pass

    def test_default_config(self):
        """Test creating Config object"""
        version = 'configs/defaults/binary-cifar-classification.yml'
        cfg = Config(version)

        self.assertIn('data', dir(cfg))
        self.assertIn('model', dir(cfg))
        self.assertIn('network', dir(cfg))


if __name__ == "__main__":
    unittest.main()
