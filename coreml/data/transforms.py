from typing import List, Any, Union, Tuple, Callable, Optional
import torch
import torch.nn.functional as F
import kornia
from coreml.factory import Factory
from coreml.utils.typing import TransformDict

# dataset statistics for standard datasets
imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
cifar_stats = ([0.491, 0.482, 0.447], [0.247, 0.243, 0.261])
mnist_stats = ([0.131], [0.308])


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


class Rescale:
    """Rescale the given input by a particular value

    Example:
    >>> signal = torch.ones(100) * 255
    >>> signal.max()
    (255)
    >>> t_signal = Rescale(255.)(signal)
    >>> t_signal.max()
    (1.)

    :param value: value to scale the input by
    :type value: int
    """
    def __init__(self, value: int):
        assert value
        self.value = value

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        return signal / self.value


class KorniaBase:
    """Base class to apply any kornia augmentation

    :param aug: kornia augmentation class to be used
    :type aug: `kornia.augmentation.AugmentationBase`
    :param kwargs: arguments for the given augmentation
    :type kwargs: dict
    """
    def __init__(
            self, flip_cls: kornia.augmentation.AugmentationBase,
            **kwargs):
        self.transform = flip_cls(**kwargs)

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        self._check_input(signal)
        ndim = len(signal.shape)

        # if input is 1D, convert it to 2D
        if ndim == 1:
            signal = signal.unsqueeze(0)

        # if the output of the previous command
        # is 2D, convert it to 3D
        if len(signal.shape) == 2:
            signal = signal.unsqueeze(0)

        # add batch dimension
        signal = self.transform(signal.unsqueeze(0))
        # returns B, C, H, W

        # remove batch dimension
        signal = signal.squeeze(0)

        # if the input was 2D, squeeze one dimension
        if ndim < 3:
            signal = signal.squeeze(0)

        # if the input was 1D, squeeze another dimension
        if ndim < 2:
            signal = signal.squeeze(0)

        return signal

    @staticmethod
    def _check_input(signal):
        assert len(signal.shape) in [1, 2, 3]
        assert isinstance(signal, torch.Tensor)


class Resize(KorniaBase):
    """Resize the given input to a particular size

    Wrapper for `kornia.geometry.transform.affwarp.Resize`

    Example:
    >>> signal = load_signal()
    >>> signal.shape
    (128, 156)
    >>> t_signal = Resize((128, 128))(signal)
    >>> t_signal.shape
    (128, 128)

    :param size: desired size after resizing
    :type size: Union[int, Tuple[int, int]]
    """
    def __init__(
            self, size: Union[int, Tuple[int, int]],):
        super(Resize, self).__init__(
            kornia.geometry.transform.affwarp.Resize, size=size)


class RandomCrop(KorniaBase):
    """Randomly crop the given input to a particular size

    Wrapper for `kornia.augmentation.RandomCrop`.

    Example:
    >>> signal = load_signal()
    >>> signal.shape
    (156, 156)
    >>> t_signal = RandomCrop((128, 128))(signal)
    >>> t_signal.shape
    (128, 128)

    :param size: desired size after resizing
    :type size: Union[int, Tuple[int, int]]
    :param padding: Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of
            length 4 is provided, it is used to pad left, top, right,
            bottom borders respectively. If a sequence of length 2 is provided,
            it is used to pad left/right, top/bottom borders, respectively.
    :type padding: Optional[Union[int, Tuple[int, int], Tuple[int, int, int, int]]]
    :param fill: Pixel fill value for constant fill. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant.
            defaults to 0.
    :type fill: int
    :param padding_mode: Type of padding. Should be: constant, edge, reflect or
            symmetric. defaults to constant
    :type padding_mode: str
    """
    def __init__(
            self, size: Tuple[int, int],
            padding: Optional[Union[int, Tuple[int, int], Tuple[int, int, int, int]]] = None,
            fill: int = 0, padding_mode: str = 'constant'):
        super(RandomCrop, self).__init__(
            kornia.augmentation.RandomCrop, size=size, padding=padding,
            fill=fill, padding_mode=padding_mode)


class RandomAffine(KorniaBase):
    """Random affine transformation of the image keeping center invariant

    Wrapper for `kornia.augmentation.RandomAffine`. Refer to
    torchvision.transforms to understand the meaning of each argument.

    :param degrees: defaults to 0
    :type degrees: Union[float, Tuple[float, float]]
    :param translate: defaults to None
    :type translate: Optional[Tuple[float, float]
    :param scale: defaults to None
    :type scale: Optional[Tuple[float, float]
    :param shear: defaults to None
    :type shear: Optional[Union[float, Tuple[float, float]]]
    """
    def __init__(
            self, degrees: Union[float, Tuple[float, float]] = 0,
            translate: Optional[Tuple[float, float]] = None,
            scale: Optional[Tuple[float, float]] = None,
            shear: Optional[Union[float, Tuple[float, float]]] = None):
        super(RandomAffine, self).__init__(
            kornia.augmentation.RandomAffine, degrees=degrees,
            translate=translate, scale=scale, shear=shear)


class ColorJitter(KorniaBase):
    """Change the brightness, contrast, saturation and hue randomly

    Wrapper for `kornia.augmentation.ColorJitter`. Refer to
    torchvision.transforms to understand the meaning of each argument.

    :param brightness: defaults to 0
    :type brightness: Union[torch.Tensor, float, Tuple[float, float], List[float]]
    :param contrast: defaults to 0
    :type contrast: Union[torch.Tensor, float, Tuple[float, float], List[float]]
    :param saturation: defaults to 0
    :type saturation: Union[torch.Tensor, float, Tuple[float, float], List[float]]
    :param hue: defaults to 0
    :type hue: Union[torch.Tensor, float, Tuple[float, float], List[float]]
    """
    def __init__(
            self, brightness: Union[torch.Tensor, float, Tuple[float, float], List[float]] = 0.,
            contrast: Union[torch.Tensor, float, Tuple[float, float], List[float]] = 0.,
            saturation: Union[torch.Tensor, float, Tuple[float, float], List[float]] = 0.,
            hue: Union[torch.Tensor, float, Tuple[float, float], List[float]] = 0.):
        super(ColorJitter, self).__init__(
            kornia.augmentation.ColorJitter, brightness=brightness,
            contrast=contrast, saturation=saturation, hue=hue)


class Normalize(KorniaBase):
    """Normalize an input with given mean and standard deviation

    Wrapper for `kornia.enhance.Normalize`

    :param mean: mean for each channel. Also accepts string in one of
        ['imagenet', 'cifar', 'mnist'] to use mean from those datasets
    :type mean: Union[torch.Tensor, float, str]
    :param std: standard deviations for each channel, Also accepts string in
        one of ['imagenet', 'cifar', 'mnist'] to use std from those datasets
    :type std: Union[torch.Tensor, float, str]
    :param dim: dimension along which to normalize, defaults to 0
    :type dim: int
    """
    def __init__(
            self, mean: Union[torch.Tensor, float, str],
            std: Union[torch.Tensor, float, str],
            dim: int = 0):
        mean, std = self._check_params(mean, std)
        super(Normalize, self).__init__(
            kornia.enhance.Normalize, mean=mean, std=std)
        self.dim = dim

    @staticmethod
    def _check_params(mean, std):
        if isinstance(mean, str):
            assert mean in ['imagenet', 'cifar', 'mnist']
            mean = locals()(f'{mean}_stats')[0]

        if isinstance(std, str):
            assert std in ['imagenet', 'cifar', 'mnist']
            std = locals()(f'{std}_stats')[1]

        if not isinstance(mean, torch.Tensor):
            mean = torch.FloatTensor(mean)

        if not isinstance(std, torch.Tensor):
            std = torch.FloatTensor(std)
        return mean, std

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        signal = signal.transpose(self.dim, 0)
        signal = super(Normalize, self).__call__(signal)
        return signal.transpose(self.dim, 0)


class RandomVerticalFlip(KorniaBase):
    """Randomly flips the input along the vertical axis

    Wrapper for `kornia.augmentation.RandomVerticalFlip`

    :param p: probability of the input being flipped; defaults to 0.5
    :type p: float
    """
    def __init__(
            self, p: float = 0.5):
        super(RandomVerticalFlip, self).__init__(
            kornia.augmentation.RandomVerticalFlip, p=p)


class RandomRotation(KorniaBase):
    """Randomly flips the input along the vertical axis

    Wrapper for `kornia.augmentation.RandomRotation`

    :param p: probability of the input being flipped; defaults to 0.5
    :type p: float
    """
    def __init__(
            self, degrees: Union[torch.Tensor, float, Tuple[float, float],
                                 List[float]]):
        super(RandomRotation, self).__init__(
            kornia.augmentation.RandomRotation, degrees=degrees)


class RandomHorizontalFlip(KorniaBase):
    """Randomly flips the input along the horizontal axis

    Wrapper for `kornia.augmentation.RandomHorizontalFlip`

    :param p: probability of the input being flipped; defaults to 0.5
    :type p: float
    """
    def __init__(
            self, p: float = 0.5):
        super(RandomHorizontalFlip, self).__init__(
            kornia.augmentation.RandomHorizontalFlip, p=p)


class RandomErasing(KorniaBase):
    """
    Erases a random selected rectangle for each image in the batch, putting the
    value to zero. The rectangle will have an area equal to the original image
    area multiplied by a value uniformly sampled between the range
    [scale[0], scale[1]) and an aspect ratio sampled between
    [ratio[0], ratio[1])

    Wrapper for `kornia.augmentation.RandomErasing`

    :param p: probability that the random erasing operation will be performed.
        defaults to 0.5
    :type p: float
    :param scale: range of proportion of erased area against input image.
        defaults to (0.02, 0.33)
    :type scale: Tuple[float, float]
    :param ratio: range of aspect ratio of erased area.
        defaults to (0.3, 3.3)
    :type ratio: Tuple[float, float]
    """
    def __init__(
            self, p: float = 0.5, scale: Tuple[float, float] = (0.02, 0.33),
            ratio: Tuple[float, float] = (0.3, 3.3)):
        self._check_params(scale, ratio)
        super(RandomErasing, self).__init__(
            kornia.augmentation.RandomErasing, p=p,
            scale=scale, ratio=ratio)

    @staticmethod
    def _check_params(scale, ratio):
        assert isinstance(scale, (list, tuple))
        assert isinstance(ratio, (list, tuple))

        for r in ratio:
            assert isinstance(r, float), "Requires float ratio value"

        for s in scale:
            assert isinstance(s, float), "Requires float scale value"


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
transform_factory.register_builder('RandomAffine', RandomAffine)
transform_factory.register_builder('Transpose', Transpose)
transform_factory.register_builder('Permute', Permute)
transform_factory.register_builder('Normalize', Normalize)
transform_factory.register_builder('Rescale', Rescale)
transform_factory.register_builder('RandomRotation', RandomRotation)
transform_factory.register_builder('ColorJitter', ColorJitter)
transform_factory.register_builder('RandomVerticalFlip', RandomVerticalFlip)
transform_factory.register_builder(
    'RandomHorizontalFlip', RandomHorizontalFlip)
transform_factory.register_builder('RandomErasing', RandomErasing)
transform_factory.register_builder('RandomCrop', RandomCrop)


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
