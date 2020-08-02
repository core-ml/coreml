from efficientnet_pytorch import EfficientNet


def _efficientnet(name, num_classes, in_channels):
    return EfficientNet.from_pretrained(
        name, num_classes=num_classes, in_channels=in_channels)


def efficientnet_b4(num_classes, in_channels):
    return _efficientnet('efficientnet-b4', num_classes, in_channels)


def efficientnet_b0(num_classes, in_channels):
    return _efficientnet('efficientnet-b0', num_classes, in_channels)


def efficientnet_b7(num_classes, in_channels):
    return _efficientnet('efficientnet-b7', num_classes, in_channels)
