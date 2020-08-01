from typing import List, Any, Union, Tuple
import torch
import torch.nn.functional as F
from coreml.factory import Factory
from coreml.utils.typing import TransformDict


class Resize:
    """Resize the given input to a particular size

    Example:
    >>> signal = load_signal()
    >>> signal.shape
    (128, 156)
    >>> t_signal = Resize((128, 128))(signal)
    >>> t_signal.shape
    (128, 128)

    :param size: desired size after resizing
    :type size: Union[int, Tuple[int]]
    """
    def __init__(self, size: Union[int, Tuple[int]]):
        self.size = size

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        # F.interpolate takes batch input and each input in a batch
        # should atleast be a 3D tensor so, 1D and 2D inputs both need
        # to be converted to 3D

        # if input is 1D, convert it to 2D
        if len(signal.shape) == 1:
            signal = signal.unsqueeze(0)

        # if the output of the previous command
        # is 2D, convert it to 3D
        if len(signal.shape) == 2:
            signal = signal.unsqueeze(0)

        # needs batch input
        signal = F.interpolate(
            signal.unsqueeze(0), self.size, mode='bilinear',
            align_corners=False)
        return signal.squeeze()


class Transpose:
    """Interchange two specified axes in the input

    :param dim0: first dimension to swap
    :type dim0: float
    :param dim1: second dimension to swap
    :type dim1: float
    """
    def __init__(self, dim0: int, dim1: int):
        self.dim0 = dim0
        self.dim1 = dim1

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        return signal.transpose(self.dim0, self.dim1)


class Permute:
    """Permute the axes order in the input

    :param order: list containing the order after permutation
    :type order: List
    """
    def __init__(self, order: List):
        self.order = order

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        return signal.permute(*self.order)


class Compose:
    """Composes several transforms together to be applied on raw signal

    :param transforms: list of transforms to apply on the signal
    :type transforms: List[Any]
    """
    def __init__(self, transforms: List[Any]):
        self.transforms = transforms

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            signal = t(signal)
        return signal


class DataProcessor:
    """Defines class for on-the-fly transformations on a given input

    :param config: list of dictionaries, each specifying a
        transformation to be applied on a given input
    :type config: List[TransformDict]
    """
    def __init__(self, config: List[TransformDict]):
        super(DataProcessor, self).__init__()

        transforms = []
        for transform in config:
            transforms.append(
                transform_factory.create(
                    transform['name'], **transform['params']))

        self.transform = Compose(transforms)

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        return self.transform(signal)


transform_factory = Factory()
transform_factory.register_builder('Compose', Compose)
transform_factory.register_builder('Resize', Resize)
transform_factory.register_builder('Transpose', Transpose)
transform_factory.register_builder('Permute', Permute)


class ClassificationAnnotationTransform:
    """
    Transforms the input label to the appropriate target value
    for single-label classification.

    :param classes: list of relevant classes for classification
    :type classes: List[str]
    """
    def __init__(self, classes: List[str]):
        assert isinstance(classes, list)
        self.classes = classes

    def __call__(self, target:  List[str]) -> int:
        # find the intersection between target and self.classes
        intersection = [
            _target for _target in target if _target in self.classes]

        # ensure that only one of the relevant classes is present
        # in the target at max
        if len(intersection) > 1:
            raise ValueError(
                'target contains more than 1 overlapping class with self.classes')

        # if one intersection, then return the corresponding index
        if len(intersection) == 1:
            return self.classes.index(intersection[0])

        raise ValueError(
            'target contains has no overlapping class with self.classes')


annotation_factory = Factory()
annotation_factory.register_builder(
    "classification", ClassificationAnnotationTransform)
