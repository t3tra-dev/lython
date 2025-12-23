# cmake -P vendor_mlir.cmake -DSRC_DIR=... -DDEST_DIR=...

cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED SRC_DIR)
    message(FATAL_ERROR "SRC_DIR must be defined")
endif()

if(NOT DEFINED DEST_DIR)
    message(FATAL_ERROR "DEST_DIR must be defined")
endif()

if(NOT EXISTS "${SRC_DIR}")
    message(FATAL_ERROR "Source directory does not exist: ${SRC_DIR}")
endif()

message(STATUS "Vendoring MLIR Python modules")
message(STATUS "  Source: ${SRC_DIR}")
message(STATUS "  Destination: ${DEST_DIR}")

if(EXISTS "${DEST_DIR}")
    file(REMOVE_RECURSE "${DEST_DIR}")
endif()

file(COPY "${SRC_DIR}/" DESTINATION "${DEST_DIR}")

set(INIT_PY_CONTENT [=[
from __future__ import annotations

import sys as _sys
from importlib import import_module as _import_module

from . import _mlir_libs
from . import dialects
from . import execution_engine
from . import ir
from . import passmanager
from . import rewrite

_sys.modules.setdefault("mlir", _sys.modules[__name__])

_sys.modules.setdefault("mlir._mlir_libs", _mlir_libs)
_sys.modules.setdefault("mlir.dialects", dialects)
_sys.modules.setdefault("mlir.execution_engine", execution_engine)
_sys.modules.setdefault("mlir.ir", ir)
_sys.modules.setdefault("mlir.passmanager", passmanager)
_sys.modules.setdefault("mlir.rewrite", rewrite)

try:
    _mlir_pkg = _import_module("._mlir_libs._mlir", __name__)
    _sys.modules.setdefault("mlir._mlir_libs._mlir", _mlir_pkg)
except Exception:
    pass

# Import and alias individual dialect modules to ensure consistent registration
# This prevents "Dialect namespace already registered" errors when importing
# from both lython.mlir.dialects and mlir.dialects paths
from .dialects import func as _func_module
from .dialects import arith as _arith_module
from .dialects import cf as _cf_module

_sys.modules.setdefault("mlir.dialects.func", _func_module)
_sys.modules.setdefault("mlir.dialects.arith", _arith_module)
_sys.modules.setdefault("mlir.dialects.cf", _cf_module)

__all__ = [
    "ir",
    "execution_engine",
    "passmanager",
    "rewrite",
    "dialects",
    "_mlir_libs",
]

del _sys, _import_module
]=])

file(WRITE "${DEST_DIR}/__init__.py" "${INIT_PY_CONTENT}")

message(STATUS "Vendoring complete")
