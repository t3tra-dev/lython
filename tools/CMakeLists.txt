add_executable(lyc
    CLI.cpp
)

target_include_directories(lyc
    PUBLIC
    ${CMAKE_SOURCE_DIR}/src/lython/dialects/cpp
    ${CMAKE_BINARY_DIR}/src/lython/dialects
    ${CMAKE_SOURCE_DIR}/src/lython/lowering
    ${CMAKE_SOURCE_DIR}/src/lython/runtime
    ${CMAKE_SOURCE_DIR}/src/lython/parser/include
    ${CMAKE_SOURCE_DIR}/src/lython/emitter/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# 生成ヘッダ依存 (PyDialectIncGen) を明示
add_dependencies(lyc PyDialectIncGen)

target_link_libraries(lyc
    PRIVATE
    PyDialect
    LythonRuntimeLowering
    LythonParser
    LythonEmitter
    MLIR
    LLVM
)

install(TARGETS lyc
    RUNTIME DESTINATION bin
)

target_compile_options(lyc PRIVATE -frtti)
if(APPLE)
    target_link_options(lyc PRIVATE "-Wl,-no_warn_duplicate_libraries")
endif()

# ビルド成果物の出力先を build/bin に統一
set_target_properties(lyc PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Pre-lowered runtime cache: lyc compiles an empty module, which embeds and
# fully lowers (and verifies) every runtime MLIR object; the emitted LLVM IR
# is then linked into each subsequent compilation instead of re-lowering the
# runtime per compile. LYTHON_RUNTIME_PRELOWER=1 keeps the embedded symbols
# public and seals their safety metadata as build-time verified.
set(LYTHON_RUNTIME_PRELOWERED_IR "${CMAKE_BINARY_DIR}/runtime_prelowered.ll")
set(LYTHON_PRELOWER_INPUT "${CMAKE_BINARY_DIR}/lython_prelower_input.mlir")
file(WRITE "${LYTHON_PRELOWER_INPUT}" "module {}\n")
file(GLOB LYTHON_PRELOWER_RUNTIME_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/src/lython/runtime/objects/*.mlir"
)
file(WRITE "${LYTHON_RUNTIME_PRELOWERED_IR}"
    "; Runtime prelowering is disabled while PyDialect lowering is rebuilt.\n"
)
add_custom_target(LythonRuntimePrelowered ALL
    DEPENDS "${LYTHON_RUNTIME_PRELOWERED_IR}"
)
