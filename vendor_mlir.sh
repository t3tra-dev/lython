#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(pwd)

SRC_MLIR="$ROOT_DIR/src/lython/mlir"
BUILD_MLIR="$ROOT_DIR/third_party/llvm-project/build/tools/mlir/python_packages/mlir_core/mlir"

if [[ ! -d "$BUILD_MLIR" ]]; then
  echo "error: MLIR build artifacts not found: $BUILD_MLIR"
  echo "hint: run ./build.sh first"
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

# パッケージとして認識されるよう保険で __init__.py を作成
[[ -f "$SRC_MLIR/__init__.py" ]] || : > "$SRC_MLIR/__init__.py"

echo "Done."
