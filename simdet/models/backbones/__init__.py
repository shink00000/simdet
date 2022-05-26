from .efficientnet import EfficientNet
from .resnet import ResNet, ResNeXt

BACKBONES = {
    'EfficientNet': EfficientNet,
    'ResNet': ResNet,
    'ResNeXt': ResNeXt
}
