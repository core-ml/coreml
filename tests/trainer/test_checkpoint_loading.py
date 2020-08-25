"""Tests loading model checkpoint"""
from copy import deepcopy
from shutil import rmtree
from os.path import join
import unittest
from coreml.config import Config
from coreml.trainer import Trainer


class CheckpointTestCase(unittest.TestCase):
    """Test checkpoint loading for various arguments"""
    @classmethod
    def setUpClass(cls):
        version = 'configs/defaults/binary-cifar.yml'
        cls.cfg = Config(version)
        cls.cfg.trainer['num_workers'] = 10
        cls.cfg.trainer['params']['max_epochs'] = 5
        cls.cfg.trainer['params']['logger'] = None

    def test_1_ckpt_last_path(self):
        """Test loading last checkpoint by path"""
        # remove existing checkpoints
        rmtree(self.cfg.checkpoint_dir)

        # first fit a trainer
        trainer = Trainer(self.cfg, **self.cfg.trainer['params'])
        trainer.fit()

        self.cfg.trainer['params']['resume_from_checkpoint'] = join(
            self.cfg.checkpoint_dir, 'epoch=4.ckpt')

        # define new trainer
        trainer = Trainer(self.cfg, **self.cfg.trainer['params'])
        # should load the last checkpoint
        trainer.evaluate('test')
        self.assertEqual(trainer.current_epoch, 5)

    def test_2_ckpt_diff_path(self):
        """Test loading checkpoint from different path"""
        self.cfg.trainer['params']['resume_from_checkpoint'] = join(
            self.cfg.checkpoint_dir, 'epoch=2.ckpt')

        # define new trainer
        trainer = Trainer(self.cfg, **self.cfg.trainer['params'])
        # should load the checkpoint from epoch 2
        trainer.evaluate('test')
        self.assertEqual(trainer.current_epoch, 3)


if __name__ == "__main__":
    unittest.main()
