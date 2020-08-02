from typing import List, Any, Union, Tuple, Callable
import torch
import torch.nn.functional as F
import kornia
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


class SubtractMean:
    """Subtract the specified mean value from the images

    :param mean: mean value to be subtracted from each input;.
    :type mean: Union[Tuple[int], Float[int], torch.Tensor]
    :param dim: dimension along which the mean should be subtracted,
        defaults to -1
    :type dim: int
    """
    def __init__(self, mean: Union[Tuple[int], List[int], torch.Tensor],
                 dim: int = -1):
        mean, dim = self._check_params(mean, dim)
        self.mean = mean
        self.dim = dim

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply the transformation

        :param signal: signal to be augmented
        :type signal: np.ndarray
        :returns: mean-subtracted signal
        """
        self._check_input(signal)
        signal = signal.transpose(self.dim, -1)
        signal -= self.mean
        return signal.transpose(self.dim, -1)

    @staticmethod
    def _check_params(mean, dim):
        assert isinstance(mean, (list, tuple, torch.Tensor))
        if not isinstance(mean, torch.Tensor):
            mean = torch.FloatTensor(mean)
        return mean, dim

    def _check_input(self, signal):
        assert isinstance(signal, torch.Tensor)
        channels = signal.shape[self.dim]
        assert channels == len(self.mean), "input does not match mean shape"


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


class RandomFlip:
    """Randomly flips the input based on the flip class

    :param flip_cls: kornia augmentation class for flipping
    :type flip_cls: `kornia.augmentation.AugmentationBase`
    :param p: probability of the input being flipped; defaults to 0.5
    :type p: float
    """
    def __init__(
            self, flip_cls: kornia.augmentation.AugmentationBase,
            p: float = 0.5):
        self.transform = flip_cls(p=p)

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        self._check_input(signal)
        ndim = len(signal.shape)

        # B, C, H, W
        signal = self.transform(signal)
        signal = signal.squeeze()

        if ndim == 2:
            # signal is 2-dimensional
            return signal

        # ndim = 3
        if len(signal.shape) == 2:
            # input signal had 1 as its channel dimension
            # which got removed due to squeeze()
            signal = signal.unsqueeze(0)

        return signal

    @staticmethod
    def _check_input(signal):
        assert isinstance(signal, torch.Tensor)
        assert len(signal.shape) in [2, 3]


class RandomVerticalFlip(RandomFlip):
    """Randomly flips the input along the vertical axis

    :param p: probability of the input being flipped; defaults to 0.5
    :type p: float
    """
    def __init__(
            self, p: float = 0.5):
        super(RandomVerticalFlip, self).__init__(
            kornia.augmentation.RandomVerticalFlip, p)


class RandomHorizontalFlip(RandomFlip):
    """Randomly flips the input along the horizontal axis

    :param p: probability of the input being flipped; defaults to 0.5
    :type p: float
    """
    def __init__(
            self, p: float = 0.5):
        super(RandomVerticalFlip, self).__init__(
            kornia.augmentation.RandomHorizontalFlip, p)


transform_factory = Factory()
transform_factory.register_builder('Compose', Compose)
transform_factory.register_builder('Resize', Resize)
transform_factory.register_builder('Transpose', Transpose)
transform_factory.register_builder('Permute', Permute)
transform_factory.register_builder('SubtractMean', SubtractMean)
transform_factory.register_builder('RandomVerticalFlip', RandomVerticalFlip)
transform_factory.register_builder(
    'RandomHorizontalFlip', RandomHorizontalFlip)
# transform_factory.register_builder('RandomErasing', RandomErasing)


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
