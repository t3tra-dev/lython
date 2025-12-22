#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)

SRC_MLIR="$ROOT_DIR/src/lython/mlir"
BUILD_MLIR="$ROOT_DIR/third_party/llvm-project/build/tools/mlir/python_packages/mlir_core/mlir"

if [[ ! -d "$BUILD_MLIR" ]]; then
  echo "error: MLIR build artifacts not found: $BUILD_MLIR"
  echo "hint: run ./build_mlir.sh first, then:"
  echo "  cd third_party/llvm-project/build && cmake --build . --target MLIRPythonModules"
  exit 1
fi

echo "Sync MLIR -> $SRC_MLIR"
rm -rf "$SRC_MLIR"
# rsync があれば高速・属性維持、なければ cp -a
if command -v rsync >/dev/null 2>&1; then
  rsync -a --delete "$BUILD_MLIR/" "$SRC_MLIR/"
else
  mkdir -p "$SRC_MLIR"
  (cd "$BUILD_MLIR" && tar cf - .) | (cd "$SRC_MLIR" && tar xpf -)
fi

# パッケージとして認識されるように __init__.py を作成
cat > "$SRC_MLIR/__init__.py" << 'EOF'
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
EOF

echo "Done."
