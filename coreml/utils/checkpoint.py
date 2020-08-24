"""Util functions for model checkpoints"""
from os.path import join, basename
from glob import glob
from natsort import natsorted


def get_last_saved_checkpoint_path(load_dir: str) -> str:
    """Returns the filename of the last saved checkpoint for a given dir.

    :param load_dir: directory from where to load checkpoints
    :type load_dir: str
    :return: path to the required checkpoint
    """
    available_ckpts = natsorted(glob(join(load_dir, '*.ckpt')))

    # return last saved checkpoint
    return join(load_dir, basename(available_ckpts[-1]))
