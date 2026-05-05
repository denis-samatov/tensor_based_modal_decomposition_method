"""Legacy module namespace for TBMD 1.x imports.

The implementations now live under :mod:`TBMD.core`.
"""

from __future__ import annotations

import warnings


def _warn() -> None:
    warnings.warn(
        "TBMD.modules is deprecated; import from TBMD.core instead.",
        DeprecationWarning,
        stacklevel=2,
    )


_warn()

from TBMD.core.decomposition.hosvd import *  # noqa: F401,F403,E402
from TBMD.core.modal_processor.modes import *  # noqa: F401,F403,E402
from TBMD.core.sensor_placement.tensor_qr_factorization import *  # noqa: F401,F403,E402
from TBMD.core.reconstruction.tensor_compressive_sensing import *  # noqa: F401,F403,E402
from TBMD.core.decomposition.geometry_aware import (  # noqa: F401,E402
    GeometryAwareConfig,
    GeometryAwareTuckerDecomposer,
)
from TBMD.core.sensor_placement.geometry_aware import (  # noqa: F401,E402
    GeometricQRConfig,
    GeometryAwareTensorQR,
)
from TBMD.core.reconstruction.geometry_aware import (  # noqa: F401,E402
    GeometryAwareCSConfig,
    GeometryAwareTensorCS,
)
from TBMD.digital_twin import *  # noqa: F401,F403,E402
from TBMD.config import SensorPlacementConfig as TensorQRConfig  # noqa: F401,E402
