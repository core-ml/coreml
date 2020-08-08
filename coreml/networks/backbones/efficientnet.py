import torch
from efficientnet_pytorch import EfficientNet as _EfficientNet


class EfficientNet(torch.nn.Module):
    """EfficientNet with all its variants

    :param variant: specific architecture to use (b0-b7)
    :param variant: str
    :param num_classes: number of classes to be predicted
    :param num_classes: int
    :param in_channles: number of input channels, defaults to 3
    :param in_channles: int, optional
    :param return_features: whether to only return features during inference,
        defaults to False
    :param return_features: bool, optional
    """
    def __init__(
            self, variant: str, num_classes: int, in_channels: int = 3,
            return_features: bool = False):
        super(EfficientNet, self).__init__()
        self.net = _EfficientNet.from_pretrained(
            variant, num_classes=num_classes, in_channels=in_channels)
        self.return_features = return_features

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.return_features:
            return self.net.extract_features(input)

        return self.net(input)
