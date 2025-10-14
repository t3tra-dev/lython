"""
Lython - Python compiler toolchain based on LLVM
~~~~~~~~~~~~~~~~~~~

:copyright: (c) 2024-present t3tra
:license: MIT, see LICENSE for more details.

"""

__title__ = "lython"
__author__ = "t3tra"
__license__ = "MIT"
__copyright__ = "Copyright 2024-present t3tra"
__version__ = "0.1.0"

__path__ = __import__("pkgutil").extend_path(__path__, __name__)

import logging
from typing import Literal, NamedTuple

from . import mlir

__all__: list[str] = ["mlir"]


class VersionInfo(NamedTuple):
    major: int
    minor: int
    micro: int
    releaselevel: Literal["alpha", "beta", "candidate", "final"]
    serial: int


version_info: VersionInfo = VersionInfo(
    major=0, minor=1, micro=0, releaselevel="alpha", serial=0
)

logging.getLogger(__name__).addHandler(logging.NullHandler())

del logging, NamedTuple, Literal, VersionInfo
