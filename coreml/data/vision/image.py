"""Defines the Image class"""
from typing import Union, List, Tuple
import numpy as np
import torch
from coreml.utils.io import read_img
from coreml.utils.typing import LabelDict


class Image:
    """Defines an image containing annotations"""
    def __init__(self, path: str, label: LabelDict = None):
        """Constructor for the class

        :param path: path to image
        :type path: str
        :param label: dictionary of labels for different tasks,
            defaults to None
        :type label: LabelDict, optional
        """
        self.path = path

        # ensure that label is of the right data type
        if label is not None:
            assert isinstance(label, dict)

        self.label = label

    def load(
            self, mode: str = 'RGB', as_tensor: bool = False
            ) -> Union[np.ndarray, torch.Tensor]:
        """Read the image

        :param mode: 'RGB' or 'BGR', defaults to 'RGB'
        :type mode: str, optional
        :param as_tensor: whether to return a `torch.Tensor` object; returns
            a np.ndarray object if `False`, defaults to `False`
        :type as_tensor: bool, optional
        :returns: image either as a nunpy array or torch.Tensor
        """
        image = read_img(self.path, mode)
        self.height, self.width = image.shape[:2]

        if as_tensor:
            image = torch.from_numpy(image.copy()).float()

        return {
            'signal': image
        }
