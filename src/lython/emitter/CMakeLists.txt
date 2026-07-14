add_library(LythonEmitter STATIC
    lib/ClosureAnalysis.cpp
    lib/EmitterCalls.cpp
    lib/EmitterClasses.cpp
    lib/EmitterControlFlow.cpp
    lib/EmitterCore.cpp
    lib/EmitterExceptions.cpp
    lib/EmitterExpressions.cpp
    lib/EmitterFunctions.cpp
    lib/EmitterImports.cpp
    lib/EmitterLoops.cpp
    lib/EmitterMatch.cpp
    lib/EmitterPrimitive.cpp
    lib/EmitterStatements.cpp
    lib/EmitterSupport.cpp
    lib/EmitterWith.cpp
    lib/PrimitiveTypes.cpp
    lib/Emitter.cpp
    lib/TypeConstraintSolver.cpp
    lib/TypeSystem.cpp
)

target_include_directories(LythonEmitter
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
    ${CMAKE_SOURCE_DIR}/src/lython/common
    ${CMAKE_SOURCE_DIR}/src/lython/dialects
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
