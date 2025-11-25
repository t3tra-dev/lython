# TableGenファイルからC++コードを生成
set(LLVM_TARGET_DEFINITIONS tablegen/PyDialect.td)
mlir_tablegen(PyDialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(PyDialect.h.inc -gen-dialect-decls)
mlir_tablegen(PyOps.cpp.inc -gen-op-defs)
mlir_tablegen(PyOps.h.inc -gen-op-decls)
add_public_tablegen_target(PyDialectIncGen)

set(PYTHON_BINDINGS_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/src/lython/mlir/dialects")
file(MAKE_DIRECTORY ${PYTHON_BINDINGS_OUTPUT_DIR})

add_custom_command(
  OUTPUT ${PYTHON_BINDINGS_OUTPUT_DIR}/_lython_ops_gen.py
  COMMAND ${MLIR_TABLEGEN_EXE} -gen-python-op-bindings -bind-dialect=py
          -I ${CMAKE_SOURCE_DIR}/third_party/llvm-project/mlir/include
          -I ${CMAKE_CURRENT_SOURCE_DIR}
          ${CMAKE_CURRENT_SOURCE_DIR}/tablegen/PyDialect.td
          > ${PYTHON_BINDINGS_OUTPUT_DIR}/_lython_ops_gen.py
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tablegen/PyDialect.td
  COMMENT "Generating Python bindings for Lython Py dialect"
  VERBATIM
)

# Debug output: confirm generated file locations
message(STATUS "Generated files will be placed in: ${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "Python bindings will be placed in: ${PYTHON_BINDINGS_OUTPUT_DIR}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_library(PyDialect STATIC
  cpp/PyDialect.cpp
  cpp/PyDialectTypes.cpp
  cpp/PyVerifier.cpp
)

add_dependencies(PyDialect PyDialectIncGen)

target_link_libraries(PyDialect
  PRIVATE
  MLIRIR
  MLIRSupport
)

target_include_directories(PyDialect PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

# MLIRのダイアレクト実装ではRTTIを有効化
target_compile_options(PyDialect PRIVATE -frtti)

add_custom_target(PyDialectPythonBindings ALL
  DEPENDS ${PYTHON_BINDINGS_OUTPUT_DIR}/_lython_ops_gen.py
)

add_dependencies(PyDialectPythonBindings PyDialect)

add_subdirectory(binding)
