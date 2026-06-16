add_library(LythonRuntime STATIC
    bltinmodule.cpp
    traceback.cpp
    stackguard.cpp
    native/exception.cpp
)

target_include_directories(LythonRuntime
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Optimize runtime for performance
target_compile_options(LythonRuntime PRIVATE -O3)
