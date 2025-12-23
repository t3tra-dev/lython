cmake_minimum_required(VERSION 3.20)

project(lython VERSION 0.1.0
               DESCRIPTION "Python compiler toolchain based on LLVM"
               LANGUAGES CXX C)

# C++17が必要 (MLIRと合わせる)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python3 3.12 EXACT REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Python3_EXECUTABLE: ${Python3_EXECUTABLE}")
message(STATUS "Python3_VERSION: ${Python3_VERSION}")

if(NOT Python3_VERSION_MAJOR EQUAL 3 OR NOT Python3_VERSION_MINOR EQUAL 12)
    message(FATAL_ERROR "Python 3.12 is required, but found ${Python3_VERSION}")
endif()

set(LLVM_PROJECT_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llvm-project/build")
set(MLIR_DIR "${LLVM_PROJECT_BUILD_DIR}/lib/cmake/mlir")
set(LLVM_DIR "${LLVM_PROJECT_BUILD_DIR}/lib/cmake/llvm")

set(LLVM_TABLEGEN_EXE "${LLVM_PROJECT_BUILD_DIR}/bin/llvm-tblgen" CACHE FILEPATH "llvm-tblgen executable")
set(MLIR_TABLEGEN_EXE "${LLVM_PROJECT_BUILD_DIR}/bin/mlir-tblgen" CACHE FILEPATH "mlir-tblgen executable")

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
