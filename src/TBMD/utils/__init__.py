"""Legacy utility namespace for TBMD 1.x imports."""

from __future__ import annotations

import warnings


def _warn() -> None:
    warnings.warn(
        "TBMD.utils is deprecated; import from TBMD.core or TBMD.visualization instead.",
        DeprecationWarning,
        stacklevel=2,
    )


_warn()
