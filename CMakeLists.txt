cmake_minimum_required(VERSION 3.20)
project(Lython LANGUAGES C CXX)

# コンパイラ設定
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# LLVM/MLIR設定 (submodule使用)
set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "")
set(LLVM_TARGETS_TO_BUILD "host" CACHE STRING "")
set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "")
set(LLVM_ENABLE_RTTI ON CACHE BOOL "")
set(LLVM_ENABLE_BINDINGS_PYTHON OFF CACHE BOOL "")
set(MLIR_ENABLE_BINDINGS_PYTHON OFF CACHE BOOL "")
set(LLVM_BUILD_TOOLS ON CACHE BOOL "")
set(MLIR_BUILD_TOOLS ON CACHE BOOL "")
set(LLVM_ENABLE_ABI_BREAKING_CHECKS OFF CACHE BOOL "")

message(STATUS "Building LLVM/MLIR 18 from submodule")

# LLVM/MLIRをsubmoduleからビルド
add_subdirectory(third_party/llvm-project/llvm)

# Dialectライブラリ
add_subdirectory(src/dialects)

# Python Dialect用 mlir-opt
add_executable(python-mlir-opt python_mlir_opt.cpp)
target_link_libraries(python-mlir-opt
    PRIVATE
    MLIRPythonDialect
    MLIROptLib
    MLIRParser
    MLIRIR
    MLIRPass
    MLIRSupport
    LLVMSupport
)
target_include_directories(python-mlir-opt PRIVATE 
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_BINARY_DIR}/src/dialects
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)
add_dependencies(python-mlir-opt MLIRPythonDialect)
