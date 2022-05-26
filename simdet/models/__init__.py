from .efficientdet import EfficientDet
from .fcos import FCOS
from .retinanet import RetinaNet

MODELS = {
    'EfficientDet': EfficientDet,
    'FCOS': FCOS,
    'RetinaNet': RetinaNet
}
