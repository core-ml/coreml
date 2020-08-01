"""Holds util functions for reading and writing various types of files"""
# pylint: disable=invalid-name,no-member

import pickle
from typing import Iterable, Dict
import yaml
import numpy as np
import cv2
from PIL import Image


def load_txt(path: str):
    """
    Read a text file containing a list of strings.

    :param path: str, filepath to read
    :return: list of strings
    """
    with open(path, 'r') as f:
        data = f.readlines()
        data = [d.strip() for d in data]

    return list(filter(len, data))


def save_txt(path: str, values: Iterable[str]):
    """
    Write a list of strings to a text file.

    :param path: str, filepath to write to
    :param values: list, list of strings to write to `path`
    """
    with open(path, 'w') as f:
        for value in values:
            f.write(value + '\n')


def read_img(path: str, order: str = 'RGB'):
    """
    Read image from the path as either 'BGR' or 'RGB'.

    :param path: str, image path to read
    :param order: str, choice of whether to load the image as
                  in 'BGR' format or 'RGB', default='BGR'
    :return: np.ndarray, image as 'BGR' or 'RGB' based on `order`
    """
    im = Image.open(path)
    im = np.asarray(im)

    if order == 'BGR':
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

    return im


class PrettySafeLoader(yaml.SafeLoader):
    """Custom loader for reading YAML files"""
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))


PrettySafeLoader.add_constructor(
    u'tag:yaml.org,2002:python/tuple',
    PrettySafeLoader.construct_python_tuple)


def read_yml(path: str):
    """
    Read params from a yml file.

    :param path: filepath to read
    :return: dict, contents stored in `path`
    """
    with open(path, 'r') as f:
        data = yaml.load(f, Loader=PrettySafeLoader)

    return data


def save_yml(path: str, data: Dict):
    """
    Save params in the given yml file path.

    :param path: str, filepath to write to
    :param data: dict, data to store
    """
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def load_pkl(path: str, load_mode: str = 'rb'):
    """
    Read a pickle file.

    :param path: str, filepath to read
    :param load_mode: str, read mode
    :return: contents of the pickle file
    """
    return pickle.load(open(path, load_mode))


def save_pkl(path: str, obj):
    """
    Save an object in a pickle file.

    :param path: str, filepath to read
    :param obj: data to store
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
