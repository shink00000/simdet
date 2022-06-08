from .detr_encoder import DETREncoder
from .fcos_encoder import FCOSEncoder
from .retina_encoder import RetinaEncoder

ENCODERS = {
    'DETREncoder': DETREncoder,
    'FCOSEncoder': FCOSEncoder,
    'RetinaEncoder': RetinaEncoder
}
