"""Tests coreml.models.binary_classification"""
from copy import deepcopy
from shutil import rmtree
from os.path import join
import unittest
from coreml.config import Config
from coreml.trainer import Trainer


class BinaryClassificationTestCase(unittest.TestCase):
    """Test training for binary classification task"""
    @classmethod
    def setUpClass(cls):
        version = 'configs/defaults/binary-cifar.yml'
        cls.cfg = Config(version)
        cls.cfg.trainer['num_workers'] = 10
        cls.cfg.trainer['params']['max_epochs'] = 1
        cls.cfg.trainer['params']['logger'] = None

    def test_1_fit(self):
        """Test trainer.fit()"""
        # remove existing checkpoints
        rmtree(self.cfg.checkpoint_dir)
        trainer = Trainer(self.cfg, **self.cfg.trainer['params'])
        trainer.fit()

    def test_2_evaluate(self):
        """Test trainer.evaluate()"""
        trainer = Trainer(self.cfg, **self.cfg.trainer['params'])
        trainer.evaluate(
            'test', ckpt_path=join(self.cfg.checkpoint_dir, 'epoch=0.ckpt'))


if __name__ == "__main__":
    unittest.main()
