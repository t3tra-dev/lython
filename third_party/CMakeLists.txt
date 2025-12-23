cmake_minimum_required(VERSION 3.20)
project(lython_third_party LANGUAGES C CXX)

include(ExternalProject)

find_package(Python3 3.12 EXACT REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Python3_EXECUTABLE: ${Python3_EXECUTABLE}")
message(STATUS "Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3_LIBRARIES: ${Python3_LIBRARIES}")
message(STATUS "Python3_VERSION: ${Python3_VERSION}")

if(NOT Python3_VERSION_MAJOR EQUAL 3 OR NOT Python3_VERSION_MINOR EQUAL 12)
    message(FATAL_ERROR "Python 3.12 is required, but found ${Python3_VERSION}")
endif()

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import nanobind; print(nanobind.cmake_dir(), end='')"
    OUTPUT_VARIABLE NANOBIND_CMAKE_DIR
    RESULT_VARIABLE NANOBIND_RESULT
    ERROR_QUIET
)
if(NOT NANOBIND_RESULT EQUAL 0)
    find_path(NANOBIND_CMAKE_DIR
        NAMES nanobind-config.cmake
        PATHS
            /opt/homebrew/share/nanobind/cmake
            /opt/homebrew/lib/cmake/nanobind
            /usr/local/share/nanobind/cmake
            /usr/local/lib/cmake/nanobind
            /usr/share/nanobind/cmake
            /usr/lib/cmake/nanobind
    )
endif()

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import nanobind; print(nanobind.include_dir(), end='')"
    OUTPUT_VARIABLE NANOBIND_INCLUDE_DIR
    RESULT_VARIABLE NANOBIND_INCLUDE_RESULT
    ERROR_QUIET
)
if(NOT NANOBIND_INCLUDE_RESULT EQUAL 0)
    find_path(NANOBIND_INCLUDE_DIR
        NAMES nanobind/nanobind.h
        PATHS
            /opt/homebrew/include
            /opt/homebrew/share/nanobind/include
            /usr/local/include
            /usr/local/share/nanobind/include
            /usr/include
    )
endif()

if(NOT NANOBIND_CMAKE_DIR OR NOT NANOBIND_INCLUDE_DIR)
    message(FATAL_ERROR
        "nanobind not found. Install with:\n"
        "  pip install nanobind\n"
        "  or: brew install nanobind"
    )
endif()
message(STATUS "nanobind cmake dir: ${NANOBIND_CMAKE_DIR}")
message(STATUS "nanobind include dir: ${NANOBIND_INCLUDE_DIR}")

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir(), end='')"
    OUTPUT_VARIABLE PYBIND11_DIR
    RESULT_VARIABLE PYBIND11_RESULT
    ERROR_QUIET
)
if(NOT PYBIND11_RESULT EQUAL 0)
    find_path(PYBIND11_DIR
        NAMES pybind11Config.cmake
        PATHS
            /opt/homebrew/share/cmake/pybind11
            /opt/homebrew/lib/cmake/pybind11
            /usr/local/share/cmake/pybind11
            /usr/local/lib/cmake/pybind11
            /usr/share/cmake/pybind11
            /usr/lib/cmake/pybind11
    )
endif()
if(PYBIND11_DIR)
    message(STATUS "pybind11 dir: ${PYBIND11_DIR}")
else()
    message(STATUS "pybind11 not found (optional)")
endif()

set(LLVM_PROJECT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llvm-project")
set(LLVM_PROJECT_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llvm-project/build")

set(LLVM_CMAKE_ARGS
    -G Ninja
    -DLLVM_ENABLE_PROJECTS=mlir
    -DLLVM_TARGETS_TO_BUILD=host
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON
    -DMLIR_ENABLE_EXECUTION_ENGINE=ON
    -DLLVM_ENABLE_RTTI=ON
    -DLLVM_ENABLE_EH=ON
    -DLLVM_INCLUDE_BENCHMARKS=OFF
    -DLLVM_INCLUDE_TESTS=OFF
    -DCMAKE_BUILD_TYPE=Release
    -DPython3_EXECUTABLE=${Python3_EXECUTABLE}
    -DPython_EXECUTABLE=${Python3_EXECUTABLE}
    -DPython3_ROOT_DIR=${Python3_ROOT_DIR}
    -DPython_ROOT_DIR=${Python3_ROOT_DIR}
    -Dnanobind_DIR=${NANOBIND_CMAKE_DIR}
    -Dnanobind_INCLUDE_DIR=${NANOBIND_INCLUDE_DIR}
    -DMLIR_PYTHON_BINDINGS_NANOBIND_INCLUDES=${NANOBIND_INCLUDE_DIR}
)

if(Python3_INCLUDE_DIRS)
    list(APPEND LLVM_CMAKE_ARGS
        -DPython_INCLUDE_DIR=${Python3_INCLUDE_DIRS}
        -DPython_INCLUDE_DIRS=${Python3_INCLUDE_DIRS}
        -DPython3_INCLUDE_DIRS=${Python3_INCLUDE_DIRS}
    )
endif()

if(Python3_LIBRARIES)
    list(APPEND LLVM_CMAKE_ARGS
        -DPython_LIBRARY=${Python3_LIBRARIES}
        -DPython3_LIBRARY=${Python3_LIBRARIES}
    )
endif()

if(PYBIND11_DIR)
    list(APPEND LLVM_CMAKE_ARGS -Dpybind11_DIR=${PYBIND11_DIR})
endif()

ExternalProject_Add(llvm_mlir
    SOURCE_DIR ${LLVM_PROJECT_SOURCE_DIR}/llvm
    BINARY_DIR ${LLVM_PROJECT_BUILD_DIR}
    CMAKE_ARGS ${LLVM_CMAKE_ARGS}
    BUILD_COMMAND ${CMAKE_COMMAND} --build . --target
        llvm-tblgen
        mlir-tblgen
        mlir-linalg-ods-yaml-gen
        MLIRPythonModules
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS
        ${LLVM_PROJECT_BUILD_DIR}/bin/llvm-tblgen
        ${LLVM_PROJECT_BUILD_DIR}/bin/mlir-tblgen
        ${LLVM_PROJECT_BUILD_DIR}/lib/cmake/llvm/LLVMConfig.cmake
        ${LLVM_PROJECT_BUILD_DIR}/lib/cmake/mlir/MLIRConfig.cmake
)

set(LLVM_TABLEGEN_EXE "${LLVM_PROJECT_BUILD_DIR}/bin/llvm-tblgen" CACHE FILEPATH "llvm-tblgen executable" FORCE)
set(MLIR_TABLEGEN_EXE "${LLVM_PROJECT_BUILD_DIR}/bin/mlir-tblgen" CACHE FILEPATH "mlir-tblgen executable" FORCE)
set(MLIR_LINALG_ODS_YAML_GEN "${LLVM_PROJECT_BUILD_DIR}/bin/mlir-linalg-ods-yaml-gen" CACHE FILEPATH "mlir-linalg-ods-yaml-gen executable" FORCE)

set(LLVM_DIR "${LLVM_PROJECT_BUILD_DIR}/lib/cmake/llvm" CACHE PATH "LLVM CMake directory" FORCE)
set(MLIR_DIR "${LLVM_PROJECT_BUILD_DIR}/lib/cmake/mlir" CACHE PATH "MLIR CMake directory" FORCE)

message(STATUS "LLVM_TABLEGEN_EXE: ${LLVM_TABLEGEN_EXE}")
message(STATUS "MLIR_TABLEGEN_EXE: ${MLIR_TABLEGEN_EXE}")
message(STATUS "LLVM_DIR: ${LLVM_DIR}")
message(STATUS "MLIR_DIR: ${MLIR_DIR}")

set(MLIR_PYTHON_SRC "${LLVM_PROJECT_BUILD_DIR}/tools/mlir/python_packages/mlir_core/mlir")

get_filename_component(LYTHON_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
if(EXISTS "${LYTHON_ROOT}/src/lython")
    set(LYTHON_MLIR_DIR "${LYTHON_ROOT}/src/lython/mlir")
else()
    set(LYTHON_MLIR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src/lython/mlir")
endif()

message(STATUS "MLIR Python modules source: ${MLIR_PYTHON_SRC}")
message(STATUS "MLIR Python modules destination: ${LYTHON_MLIR_DIR}")

set(VENDOR_MLIR_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/cmake/vendor_mlir.cmake")

ExternalProject_Add_Step(llvm_mlir vendor_python_modules
    COMMAND ${CMAKE_COMMAND}
        -DSRC_DIR=${MLIR_PYTHON_SRC}
        -DDEST_DIR=${LYTHON_MLIR_DIR}
        -P ${VENDOR_MLIR_SCRIPT}
    DEPENDEES build
    COMMENT "Vendoring MLIR Python modules to ${LYTHON_MLIR_DIR}"
)
