add_executable(LythonCpythonSpecGen
    tools/CpythonSpecGen.cpp
    lib/CpythonContract.cpp
    lib/CpythonSpec.cpp
    lib/Grammar.cpp
)

target_include_directories(LythonCpythonSpecGen
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(LythonCpythonSpecGen
    PRIVATE
    LYTHON_CPYTHON_314_PARSER_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
)

set(LYTHON_CPYTHON_SPEC_SNAPSHOT
    "${CMAKE_CURRENT_BINARY_DIR}/CpythonSpecSnapshot.inc")

add_custom_command(
    OUTPUT "${LYTHON_CPYTHON_SPEC_SNAPSHOT}"
    COMMAND LythonCpythonSpecGen "${LYTHON_CPYTHON_SPEC_SNAPSHOT}"
    DEPENDS
        LythonCpythonSpecGen
        "${CMAKE_CURRENT_SOURCE_DIR}/Tokens"
        "${CMAKE_CURRENT_SOURCE_DIR}/python.gram"
        "${CMAKE_CURRENT_SOURCE_DIR}/Python.asdl"
        "${CMAKE_CURRENT_SOURCE_DIR}/parser.c"
    VERBATIM
)

add_custom_target(LythonCpythonSpecSnapshot
    DEPENDS "${LYTHON_CPYTHON_SPEC_SNAPSHOT}")

add_library(LythonParser STATIC
    PegenShim.c
    parser.c
    lib/Ast.cpp
    lib/CpythonPegAdapter.cpp
    lib/CpythonSpec.cpp
    lib/GeneratedTokenStream.cpp
    lib/Grammar.cpp
    lib/Lexer.cpp
    lib/Parser.cpp
    lib/SourceEncoding.cpp
    lib/Token.cpp
    lib/UnicodeNames.cpp
    lib/UnicodeNormalize.cpp
    lib/UnicodeXid.cpp
)

add_dependencies(LythonParser LythonCpythonSpecSnapshot)

set_source_files_properties(parser.c PROPERTIES
    COMPILE_OPTIONS "-Wno-unused-but-set-variable;-Wno-incompatible-pointer-types-discards-qualifiers"
)
set_source_files_properties(PegenShim.c PROPERTIES
    COMPILE_OPTIONS "-Wno-cast-qual"
)

target_include_directories(LythonParser
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
)
