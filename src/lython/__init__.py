"""
lython: A transpiler that converts Python code to LLVM IR and compiles it to machine code
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


def get_codegen():
    from . import codegen

    return codegen


def get_compiler():
    from . import compiler

    return compiler


__all__ = ["get_codegen", "get_compiler"]


class VersionInfo(NamedTuple):
    major: int
    minor: int
    micro: int
    releaselevel: Literal["alpha", "beta", "candidate", "final"]
    serial: int


version_info: VersionInfo = VersionInfo(
    major=0, minor=0, micro=1, releaselevel="final", serial=0
)

logging.getLogger(__name__).addHandler(logging.NullHandler())

del logging, NamedTuple, Literal, VersionInfo
