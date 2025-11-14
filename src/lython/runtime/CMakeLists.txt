add_library(LythonRuntime STATIC
    lyrt_abi.cpp
)

target_include_directories(LythonRuntime
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
