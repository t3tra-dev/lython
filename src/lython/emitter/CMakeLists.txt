add_library(LythonEmitter STATIC
    lib/ClosureAnalysis.cpp
    lib/EmitterCore.cpp
    lib/EmitterWith.cpp
    lib/Emitter.cpp
    lib/TypeSystem.cpp
)

target_include_directories(LythonEmitter
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
    ${CMAKE_SOURCE_DIR}/src/lython/dialects/cpp
    ${CMAKE_BINARY_DIR}/src/lython/dialects
)

target_link_libraries(LythonEmitter
    PUBLIC
    LythonParser
    MLIR
    LLVM
    PRIVATE
    PyDialect
)

add_dependencies(LythonEmitter PyDialectIncGen)
target_compile_options(LythonEmitter PRIVATE -frtti)
