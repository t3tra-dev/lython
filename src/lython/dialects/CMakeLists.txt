# TableGenファイルからC++コードを生成
set(LLVM_TARGET_DEFINITIONS tablegen/PyDialect.td)
mlir_tablegen(PyDialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(PyDialect.h.inc -gen-dialect-decls)
mlir_tablegen(PyOps.cpp.inc -gen-op-defs)
mlir_tablegen(PyOps.h.inc -gen-op-decls)
add_public_tablegen_target(PyDialectIncGen)

# Debug output: confirm generated file locations
message(STATUS "Generated files will be placed in: ${CMAKE_CURRENT_BINARY_DIR}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(LYTHON_PROTOCOL_RELATIONS_INC
  "${CMAKE_CURRENT_BINARY_DIR}/PyProtocolRelations.inc"
)
file(GLOB LYTHON_PROTOCOL_MANIFEST_MLIR CONFIGURE_DEPENDS
  "${CMAKE_SOURCE_DIR}/src/lython/runtime/lib/*.mlir"
)
list(PREPEND LYTHON_PROTOCOL_MANIFEST_MLIR
  "${CMAKE_SOURCE_DIR}/src/lython/runtime/typing.mlir"
)
add_custom_command(
  OUTPUT "${LYTHON_PROTOCOL_RELATIONS_INC}"
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_SOURCE_DIR}/tools/dialects/GenerateProtocolRelations.py"
          --input ${LYTHON_PROTOCOL_MANIFEST_MLIR}
          --output "${LYTHON_PROTOCOL_RELATIONS_INC}"
  DEPENDS
    "${CMAKE_SOURCE_DIR}/tools/dialects/GenerateProtocolRelations.py"
    ${LYTHON_PROTOCOL_MANIFEST_MLIR}
  COMMENT "Generating protocol relation tables from typing manifests"
  VERBATIM
)
add_custom_target(PyProtocolRelationsIncGen
  DEPENDS "${LYTHON_PROTOCOL_RELATIONS_INC}"
)

add_library(PyDialect STATIC
  cpp/PyDialect.cpp
  cpp/PyDialectTypes.cpp
  cpp/PyTypeObject.cpp
  cpp/PyVerifier/Verifier.cpp
)

add_dependencies(PyDialect PyDialectIncGen PyProtocolRelationsIncGen)

target_link_libraries(PyDialect
  PRIVATE
  MLIR
  LLVM
)

target_include_directories(PyDialect PUBLIC
  ${CMAKE_SOURCE_DIR}/src/lython/common
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

# MLIRのダイアレクト実装ではRTTIを有効化
target_compile_options(PyDialect PRIVATE -frtti)
