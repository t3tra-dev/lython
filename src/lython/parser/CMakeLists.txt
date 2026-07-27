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
    COMMAND ${LYTHON_HOST_TOOL_LAUNCHER}
        $<TARGET_FILE:LythonCpythonSpecGen> "${LYTHON_CPYTHON_SPEC_SNAPSHOT}"
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
# CpythonSpec.cpp は ~800KB の生成スナップショット (静的初期化子) を include して
# いる。データ表であって hot code ではないので、最適化しても得るものが無く、掛かる
# コストは大きい。
#
# 実測 (RelWithDebInfo、`build/.ninja_log`): このオブジェクト 1 個で
# **894.8 秒 / コールドビルド全体 907.3 秒** を占め、次に遅い TU は 11.3 秒 —
# 80 倍である。同じソースが `LythonCpythonSpecGen` では 1.7 秒でコンパイルされる。
# `00984f0` で `CMAKE_BUILD_TYPE` の既定を RelWithDebInfo にしたことで、この表が
# `-O2 -g` でコンパイルされるようになったのが原因 (RSS も 1 GB を超える)。
#
# Why NOT 最適化を残す: 交換レートを測った。`lyc parse` の起動は
# RelWithDebInfo (この TU が -O2) で 0.01 s、Debug (同 -O0) で 0.02 s。差 10 ms は
# **バイナリ全体**の最適化差であり、`loadGeneratedSpec()` に帰属する分はその一部に
# すぎない。**起動 10 ms 以下 対 コールドビルド約 880 秒**なので、-O0 が正しい。
#
# fuzzer ビルドでは加えて計装も外す (ASan + カバレッジとの組み合わせで数十分になる)。
# データ表なので計装を外しても探索価値は失わない。
if(LYTHON_ENABLE_FUZZERS)
    set_source_files_properties(lib/CpythonSpec.cpp PROPERTIES
        COMPILE_OPTIONS "-fno-sanitize=fuzzer-no-link,address;-O0"
    )
else()
    set_source_files_properties(lib/CpythonSpec.cpp PROPERTIES
        COMPILE_OPTIONS "-O0"
    )
endif()
set_source_files_properties(PegenShim.c PROPERTIES
    COMPILE_OPTIONS "-Wno-cast-qual"
)

target_include_directories(LythonParser
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
)
