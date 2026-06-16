add_library(LythonEmitter STATIC
    lib/BuilderAst.cpp
    lib/Builder.cpp
    lib/BuilderAsync.cpp
    lib/BuilderBoolOps.cpp
    lib/BuilderCalls.cpp
    lib/BuilderClasses.cpp
    lib/BuilderContainers.cpp
    lib/BuilderControl.cpp
    lib/BuilderEH.cpp
    lib/BuilderExpressions.cpp
    lib/BuilderFunctions.cpp
    lib/BuilderImports.cpp
    lib/BuilderLiterals.cpp
    lib/BuilderMatch.cpp
    lib/BuilderStatements.cpp
    lib/BuilderSupport.cpp
    lib/BuilderTensors.cpp
    lib/BuilderTypes.cpp
    lib/Emitter.cpp
    lib/TypeInference.cpp
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
    LythonProtocolEvidence
)

add_dependencies(LythonEmitter PyDialectIncGen)
target_compile_options(LythonEmitter PRIVATE -frtti)
