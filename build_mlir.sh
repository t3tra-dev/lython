#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT_DIR/third_party/llvm-project"

mkdir -p build
cd build

PYTHON_BIN=${PYTHON_BIN:-"$(command -v python3)"}
if [[ -f "$ROOT_DIR/.venv/bin/python" ]]; then
	PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
fi

PY_BASE_PREFIX=$("$PYTHON_BIN" -c "import sys; print(sys.base_prefix)")
echo "Python base_prefix: $PY_BASE_PREFIX"

PY_INCLUDE_DIR=$("$PYTHON_BIN" -c "import sysconfig; print(sysconfig.get_paths().get('include',''))")
PY_LIBDIR=$("$PYTHON_BIN" -c "import sysconfig; print((sysconfig.get_config_var('LIBDIR') or sysconfig.get_config_var('LIBPL') or ''))")
PY_LDLIB=$("$PYTHON_BIN" -c "import sysconfig; print(sysconfig.get_config_var('LDLIBRARY') or '')")
if [[ -z "$PY_INCLUDE_DIR" && "$OS" == linux ]] && command -v pkg-config >/dev/null 2>&1; then
	if pkg-config --exists python-3.12; then
		PY_INCLUDE_DIR=$(pkg-config --cflags-only-I python-3.12 | sed -E 's/ *-I//g' | awk '{print $1}')
	elif pkg-config --exists python3; then
		PY_INCLUDE_DIR=$(pkg-config --cflags-only-I python3 | sed -E 's/ *-I//g' | awk '{print $1}')
	fi
fi
if [[ -n "$PY_LIBDIR" && -n "$PY_LDLIB" && -f "$PY_LIBDIR/$PY_LDLIB" ]]; then
	PY_LIBRARY="$PY_LIBDIR/$PY_LDLIB"
else
	PY_LIBRARY=""
fi
echo "Python include: $PY_INCLUDE_DIR"
echo "Python library: ${PY_LIBRARY:-<auto>}"

echo "Using Python: $PYTHON_BIN"

UNAME_S=$(uname -s || echo unknown)
case "$UNAME_S" in
Linux) OS=linux ;;
Darwin) OS=macos ;;
*) OS=other ;;
esac

NB_CMAKE_DIR=""
NB_INCLUDE_DIR=""
if "$PYTHON_BIN" -c "import nanobind,sys;print(nanobind.cmake_dir())" >/dev/null 2>&1; then
	NB_CMAKE_DIR=$("$PYTHON_BIN" -c "import nanobind;print(nanobind.cmake_dir(), end='')")
	NB_INCLUDE_DIR=$("$PYTHON_BIN" -c "import nanobind;print(nanobind.include_dir(), end='')")
	echo "Found nanobind in Python: cmake_dir=$NB_CMAKE_DIR include_dir=$NB_INCLUDE_DIR"
fi

if [[ -z "${NB_CMAKE_DIR}" || -z "${NB_INCLUDE_DIR}" ]]; then
	for PFX in /opt/homebrew /usr/local /usr; do
		if [[ -d "$PFX/share/nanobind/cmake" && -d "$PFX/share/nanobind/include" ]]; then
			NB_CMAKE_DIR="$PFX/share/nanobind/cmake"
			NB_INCLUDE_DIR="$PFX/share/nanobind/include"
			echo "Using system nanobind: cmake_dir=$NB_CMAKE_DIR include_dir=$NB_INCLUDE_DIR"
			break
		fi
		for CAND in "$PFX/lib/cmake/nanobind" "$PFX/lib64/cmake/nanobind" "$PFX/share/cmake/nanobind"; do
			if [[ -z "$NB_CMAKE_DIR" && -d "$CAND" ]]; then
				NB_CMAKE_DIR="$CAND"
			fi
		done
		if [[ -z "$NB_INCLUDE_DIR" && -d "$PFX/include/nanobind" ]]; then
			NB_INCLUDE_DIR="$PFX/include"
		fi
		if [[ -n "$NB_CMAKE_DIR" && -n "$NB_INCLUDE_DIR" ]]; then
			echo "Using system nanobind: cmake_dir=$NB_CMAKE_DIR include_dir=$NB_INCLUDE_DIR"
			break
		fi
	done
fi

if [[ -z "${NB_INCLUDE_DIR}" ]] && command -v pkg-config >/dev/null 2>&1; then
	if pkg-config --exists nanobind; then
		NB_INCLUDE_DIR=$(pkg-config --cflags-only-I nanobind | sed -E 's/ *-I//g' | awk '{print $1}')
		echo "Using pkg-config nanobind include: $NB_INCLUDE_DIR"
	fi
fi

if [[ -z "${NB_CMAKE_DIR}" || -z "${NB_INCLUDE_DIR}" ]]; then
	echo "[ERROR] nanobind not found. Install one of:"
	echo "  - In your venv:  $PYTHON_BIN -m pip install -U nanobind"
	echo "  - Or via Homebrew: brew install nanobind"
	exit 1
fi

PYBIND11_DIR=""
if "$PYTHON_BIN" -c "import pybind11,sys;print(pybind11.get_cmake_dir())" >/dev/null 2>&1; then
	PYBIND11_DIR=$("$PYTHON_BIN" -c "import pybind11;print(pybind11.get_cmake_dir(), end='')")
	echo "Found pybind11 in Python: $PYBIND11_DIR"
else
	for PFX in /opt/homebrew /usr/local /usr; do
		for CAND in "$PFX/share/cmake/pybind11" "$PFX/lib/cmake/pybind11" "$PFX/lib64/cmake/pybind11"; do
			if [[ -d "$CAND" ]]; then
				PYBIND11_DIR="$CAND"
				echo "Using system pybind11: $PYBIND11_DIR"
				break 2
			fi
		done
	done
fi

if ! (
	ls /opt/homebrew/lib/cmake/tsl-robin-map/tsl-robin-map*.cmake >/dev/null 2>&1 || \
	ls /usr/local/lib/cmake/tsl-robin-map/tsl-robin-map*.cmake >/dev/null 2>&1 || \
	ls /usr/lib/cmake/tsl-robin-map/tsl-robin-map*.cmake >/dev/null 2>&1 || \
	ls /usr/lib64/cmake/tsl-robin-map/tsl-robin-map*.cmake >/dev/null 2>&1
); then
	echo "[NOTICE] Missing tsl-robin-map CMake package. If configuration fails, install it:"
	if [[ "$OS" == macos ]]; then
		echo "  macOS (Homebrew): brew install robin-map"
	elif [[ "$OS" == linux ]]; then
		echo "  Linux: package varies by distro; build from https://github.com/Tessil/robin-map if unavailable"
	fi
fi

echo "Configuring with CMake..."
cmake -G Ninja ../llvm \
	-DLLVM_ENABLE_PROJECTS=mlir \
	-DLLVM_TARGETS_TO_BUILD=host \
	-DMLIR_ENABLE_BINDINGS_PYTHON=ON \
	-DMLIR_ENABLE_EXECUTION_ENGINE=ON \
	-DLLVM_ENABLE_RTTI=ON \
	-DLLVM_ENABLE_EH=ON \
	-DLLVM_INCLUDE_BENCHMARKS=OFF \
	-DLLVM_INCLUDE_TESTS=OFF \
	-DCMAKE_BUILD_TYPE=Release \
	-DPython3_EXECUTABLE="${PYTHON_BIN}" \
	-DPython_EXECUTABLE="${PYTHON_BIN}" \
	-DPython3_ROOT_DIR="${PY_BASE_PREFIX}" \
	-DPython_ROOT_DIR="${PY_BASE_PREFIX}" \
	-DPython3_FIND_VIRTUALENV=ONLY \
	-DPython_FIND_VIRTUALENV=ONLY \
	${PY_INCLUDE_DIR:+-DPython_INCLUDE_DIR="${PY_INCLUDE_DIR}"} \
	${PY_INCLUDE_DIR:+-DPython_INCLUDE_DIRS="${PY_INCLUDE_DIR}"} \
	${PY_INCLUDE_DIR:+-DPython3_INCLUDE_DIRS="${PY_INCLUDE_DIR}"} \
	${PY_LIBRARY:+-DPython_LIBRARY="${PY_LIBRARY}"} \
	${PY_LIBRARY:+-DPython3_LIBRARY="${PY_LIBRARY}"} \
	-Dnanobind_DIR="${NB_CMAKE_DIR}" \
	-Dnanobind_INCLUDE_DIR="${NB_INCLUDE_DIR}" \
	-DMLIR_PYTHON_BINDINGS_NANOBIND_INCLUDES="${NB_INCLUDE_DIR}" \
	${PYBIND11_DIR:+-Dpybind11_DIR="${PYBIND11_DIR}"}

echo "Building MLIR Python modules..."
cmake --build . --target MLIRPythonModules
