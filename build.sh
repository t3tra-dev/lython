#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)

MLIR_BUILD_DIR="$ROOT_DIR/third_party/llvm-project/build"
if [[ ! -d "$MLIR_BUILD_DIR" ]]; then
    echo "[ERROR] MLIR build directory not found: $MLIR_BUILD_DIR"
    echo "Please run ./build_mlir.sh first to build MLIR"
    exit 1
fi

if [[ ! -f "$MLIR_BUILD_DIR/python_packages/mlir_core/mlir/_mlir_libs/_mlir.so" ]] && [[ ! -f "$MLIR_BUILD_DIR/python_packages/mlir_core/mlir/_mlir_libs/_mlir.dylib" ]]; then
    echo "[WARNING] MLIR Python modules may not be built. Consider running:"
    echo "  cd $MLIR_BUILD_DIR && cmake --build . --target MLIRPythonModules"
fi

BUILD_DIR=${BUILD_DIR:-"$ROOT_DIR/build"}
mkdir -p "$BUILD_DIR"

CMAKE_GENERATOR_NAME=""
if [[ -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    CMAKE_GENERATOR_NAME=$(grep -E '^CMAKE_GENERATOR:INTERNAL=' "$BUILD_DIR/CMakeCache.txt" | sed -E 's/^CMAKE_GENERATOR:INTERNAL=//')
fi

if [[ -z "$CMAKE_GENERATOR_NAME" ]]; then
    if command -v ninja >/dev/null 2>&1; then
        CMAKE_GENERATOR_NAME="Ninja"
    else
        CMAKE_GENERATOR_NAME="Unix Makefiles"
    fi
fi

cd "$BUILD_DIR"

PYTHON_BIN=${PYTHON_BIN:-"$(command -v python3)"}
if [[ -f "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
fi

echo "Using Python: $PYTHON_BIN"
echo "Using MLIR from: $MLIR_BUILD_DIR"
echo "Using CMake generator: $CMAKE_GENERATOR_NAME"

echo "Configuring Lython with CMake..."
cmake -G "$CMAKE_GENERATOR_NAME" "$ROOT_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DMLIR_DIR="$MLIR_BUILD_DIR/lib/cmake/mlir" \
    -DLLVM_DIR="$MLIR_BUILD_DIR/lib/cmake/llvm" \
    -DPython3_EXECUTABLE="$PYTHON_BIN"

echo "Building Lython PyDialect..."
cmake --build . --target PyDialectPythonBindings

echo "Build completed successfully!"
echo "PyDialect TableGen files generated in: $BUILD_DIR"
