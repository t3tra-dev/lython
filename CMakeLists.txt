cmake_minimum_required(VERSION 3.24)
project(Lython LANGUAGES C CXX)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

set(LLVM_ENABLE_PROJECTS "mlir;llvm")
set(LLVM_EXTERNAL_PROJECTS ${LLVM_ENABLE_PROJECTS})

include(cmake/LLVMExternalProject.cmake)

add_subdirectory(src)
