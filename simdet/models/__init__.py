from .detr import DETR
from .efficientdet import EfficientDet
from .fcos import FCOS
from .retinanet import RetinaNet

MODELS = {
    'DETR': DETR,
    'EfficientDet': EfficientDet,
    'FCOS': FCOS,
    'RetinaNet': RetinaNet
}
