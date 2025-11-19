cmake_minimum_required(VERSION 3.20)

project(lython VERSION 0.1.0
               DESCRIPTION "Python compiler toolchain based on LLVM"
               LANGUAGES CXX C)

# C++17が必要 (MLIRと合わせる)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(MLIR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llvm-project/build/lib/cmake/mlir")
set(LLVM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llvm-project/build/lib/cmake/llvm")

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})

add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(src)
add_subdirectory(tools)
