# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LoggerKit: routes Warp's host-side diagnostics through Omniverse Kit's
Carbonite (``carb``) logging functions.

Frameworks running inside Kit install this via :func:`warp.set_logger` so that
Warp messages appear in Kit's log viewer with proper severity, timestamps,
and source attribution.  Outside Kit (e.g. unit tests), ``carb`` is
unavailable and the implementation falls back to ``print()`` /
``warnings.warn()``.
"""

import sys
import warnings

from warp._src.logger import _format_warning


def _warp_showwarning_stdout(message, category, filename, lineno, file=None, line=None):
    """Format and write a Warp warning to sys.stdout (for Kit compatibility)."""
    sys.stdout.write(_format_warning(message, category, filename, lineno, line))


class LoggerKit:
    """Logger for Omniverse Kit and similar frameworks.

    Routes output through Kit's Carbonite (``carb``) logging functions so that
    Warp messages appear in Kit's log viewer with proper severity, timestamps,
    and source attribution.  Falls back to ``print()`` if ``carb`` is not
    available (e.g. unit tests outside a Kit process).
    """

    def _carb(self):
        """Lazily import carb (only available inside a Kit process)."""
        try:
            import carb  # noqa: PLC0415

            return carb
        except ImportError:
            return None

    def debug(self, message: str) -> None:
        carb = self._carb()
        if carb is not None:
            carb.log_verbose(message)
        else:
            print(message)

    def info(self, message: str) -> None:
        carb = self._carb()
        if carb is not None:
            carb.log_info(message)
        else:
            print(message)

    def warning(self, message: str, category=None, stacklevel: int = 1) -> None:
        carb = self._carb()
        if carb is not None:
            cat_name = category.__name__ if category is not None else "UserWarning"
            carb.log_warn(f"Warp {cat_name}: {message}")
        else:
            with warnings.catch_warnings():
                warnings.showwarning = _warp_showwarning_stdout
                warnings.warn(message, category, stacklevel=stacklevel + 1)

    def error(self, message: str) -> None:
        carb = self._carb()
        if carb is not None:
            carb.log_error(f"Warp Error: {message}")
        else:
            # Stay on stdout: Kit's test infrastructure treats writes to
            # sys.stderr as hard test failures.
            print("Warp Error: " + message)
