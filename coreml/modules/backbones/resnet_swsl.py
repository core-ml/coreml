# Loading models from https://pytorch.org/hub/facebookresearch_semi-supervised-ImageNet1K-models_resnext
# modified to work as a backbone

from torch.hub import load_state_dict_from_url
from coreml.modules.backbones.resnet import ResNet, Bottleneck


semi_weakly_supervised_model_urls = {
    'resnext50_32x4d': 'https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext50_32x4-72679e44.pth',
}


def _resnext(url, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    state_dict = load_state_dict_from_url(url, progress=progress)
    model.load_state_dict(state_dict, strict=False)
    return model


def resnext50_32x4d_swsl(progress=True, pretrained=True, **kwargs):
    """Constructs a semi-weakly supervised ResNeXt-50 32x4 model pre-trained on
        1B weakly supervised image dataset and finetuned on ImageNet.
       `"Billion-scale Semi-Supervised Learning for Image Classification"
       <https://arxiv.org/abs/1905.00546>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to
            stderr.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnext(
        semi_weakly_supervised_model_urls['resnext50_32x4d'],
        Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)
