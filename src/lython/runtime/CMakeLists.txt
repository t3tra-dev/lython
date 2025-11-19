add_library(LythonRuntime STATIC
    lyrt.cpp
    bltinmodule.cpp
    objects/object.cpp
    objects/unicode.cpp
    objects/tuple.cpp
    objects/dict.cpp
    objects/function.cpp
)

target_include_directories(LythonRuntime
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
