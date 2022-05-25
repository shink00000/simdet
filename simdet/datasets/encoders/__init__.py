from .fcos_encoder import FCOSEncoder
from .retina_encoder import RetinaEncoder

ENCODERS = {
    'FCOSEncoder': FCOSEncoder,
    'RetinaEncoder': RetinaEncoder
}
