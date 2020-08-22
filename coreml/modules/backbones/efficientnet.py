from timm.models import efficientnet
from coreml.modules.backbones.base import BaseTimmModel


class EfficientNet(BaseTimmModel):
    """EfficientNet with all its variants

    :param variant: specific architecture to use (b0-b7)
    :param variant: str
    :param num_classes: number of classes to be predicted
    :param num_classes: int
    :param in_channels: number of input channels, defaults to 3
    :param in_channels: int, optional
    :param return_features: whether to only return features during inference,
        defaults to False
    :param return_features: bool, optional
    """
    def __init__(
            self, variant: str, num_classes: int, in_channels: int = 3,
            return_features: bool = False):
        method = getattr(efficientnet, variant)
        super(EfficientNet, self).__init__(
            method, variant, num_classes, in_channels, return_features)
