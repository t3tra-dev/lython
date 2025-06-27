include(FetchContent)

set(LLVM_PROJECT_GIT_REPOSITORY "https://github.com/llvm/llvm-project.git" CACHE STRING "")
set(LLVM_PROJECT_GIT_TAG "llvmorg-18.1.1" CACHE STRING "")

FetchContent_Declare(
    llvm-project
    GIT_REPOSITORY ${LLVM_PROJECT_GIT_REPOSITORY}
    GIT_TAG ${LLVM_PROJECT_GIT_TAG}
)

FetchContent_MakeAvailable(llvm-project)

set(LLVM_DIR ${llvm-project_BINARY_DIR}/lib/cmake/llvm)
set(MLIR_DIR ${llvm-project_BINARY_DIR}/lib/cmake/mlir)

list(APPEND CMAKE_PREFIX_PATH ${LLVM_DIR} ${MLIR_DIR})
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})

add_subdirectory(${llvm-project_SOURCE_DIR}/llvm ${llvm-project_BINARY_DIR}/llvm)
add_subdirectory(${llvm-project_SOURCE_DIR}/mlir ${llvm-project_BINARY_DIR}/mlir)
