add_library(LythonRuntime STATIC
    bltinmodule.cpp
    traceback.cpp
    native/exception.cpp
)

target_include_directories(LythonRuntime
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/third_party/llvm-project/llvm/include
    ${CMAKE_SOURCE_DIR}/third_party/llvm-project/mlir/include
)

# Optimize runtime for performance
target_compile_options(LythonRuntime PRIVATE -O3)
