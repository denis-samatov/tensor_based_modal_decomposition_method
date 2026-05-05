from . import _warn

_warn()

from TBMD.config import SensorPlacementConfig as TensorQRConfig  # noqa: F401,E402
from TBMD.core.sensor_placement.tensor_qr_factorization import *  # noqa: F401,F403,E402
