"""Defines model checkpoint class"""
from typing import Dict
from os.path import join, basename
import numpy as np
from natsort import natsorted
from glob import glob

from coreml.utils.logger import color


class ModelCheckpoint(object):
    """Define model checkpoint class that tracks metrics and saves checkpoints

    :param ckpt_dir: folder in which checkpoints will be stored
    :type ckpt_dir: str
    :param period: number of epochs between two saving consecutive ckpts
    :type period: int
    :param monitor: metric to track for deciding whether to save checkpoint
    :type monitor: str
    :param monitor_mode: indicates whether improvement is max/min
    :type monitor_mode: str
    """
    def __init__(
            self, ckpt_dir: str, period: int, monitor: str,
            monitor_mode: str):
        super(ModelCheckpoint, self).__init__()
        self.ckpt_dir = ckpt_dir
        self.period = period
        self.monitor = monitor
        self.monitor_mode = monitor_mode

        # if monitor metric is like accuracy, initialize by -infinity
        # if monitor metric is like loss, initialize by +infinity
        default_value = ((monitor_mode == 'min') - (monitor_mode == 'max')) * np.inf
        self.best_metric_dict = {
            monitor: default_value
        }

    def get_saved_checkpoint_path(
            self, load_best: bool = False,
            epoch: int = -1) -> str:
        """Returns the filename of saved checkpoint for a given epoch.

        :param load_best: whether to load best checkpoint or not, defaults to
            False
        :type load_best: bool, optional
        :param epoch: epoch whose checkpoint is to be loaded; only used
            if `load_best=False`, defaults to -1
        :type epoch: int, optional

        :return: path to the required checkpoint
        """
        if load_best:
            fname = 'best_ckpt.pth.tar'
        else:
            available_ckpts = natsorted(
                glob(join(self.ckpt_dir, '[0-9]*_ckpt.pth.tar')))
            if epoch == -1:
                fname = basename(available_ckpts[-1])
            else:
                fname = '{}_ckpt.pth.tar'.format(epoch)

        return join(self.ckpt_dir, fname)

    def update_best_metric(
            self, epoch_counter: int, epoch_metric_dict: Dict) -> Dict:
        """Updates current best metric values with current epoch metrics

        :param epoch_counter: number of current epoch
        :type epoch_counter: int
        :param epoch_metric_dict: dict containing the current epoch metrics
        :type epoch_metric_dict: Dict

        :return: dict containing save status incl. filepath to save the model
        """
        assert self.monitor in epoch_metric_dict, \
            'Metric {} not computed'.format(self.monitor)

        old_value = self.best_metric_dict[self.monitor]
        new_value = epoch_metric_dict[self.monitor]

        if self.monitor_mode == 'max':
            improvement_indicator = (new_value > old_value)
        elif self.monitor_mode == 'min':
            improvement_indicator = (new_value < old_value)

        save_status = {'save': False}
        if improvement_indicator:
            self.best_metric_dict[self.monitor] = new_value
            info = '[{}] {} improved from {} to {}'.format(
                color('Saving best model', 'red'), self.monitor, old_value,
                new_value)
            save_status.update({
                'save': True,
                'path': join(self.ckpt_dir, 'best_ckpt.pth.tar'),
                'info': info
            })
        else:
            info = '[{}] {} did not improve from {}'.format(
                color('Saving regular model', 'red'), self.monitor, old_value)

        if not ((epoch_counter + 1) % self.period):
            save_status.update({
                'save': True,
                'path': join(self.ckpt_dir, '{}_ckpt.pth.tar'.format(
                    epoch_counter)),
                'info': info
            })

        return save_status
