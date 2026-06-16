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

add_library(PyDialect STATIC
  cpp/PyDialect.cpp
  cpp/PyDialectTypes.cpp
  cpp/PyTypeObject.cpp
  cpp/PyVerifier/Common.cpp
  cpp/PyVerifier/ClassFunc.cpp
  cpp/PyVerifier/Call.cpp
  cpp/PyVerifier/Casts.cpp
  cpp/PyVerifier/Async.cpp
  cpp/PyVerifier/ConstTupleDict.cpp
  cpp/PyVerifier/EH.cpp
  cpp/PyVerifier/Numeric.cpp
  cpp/PyVerifier/Union.cpp
  cpp/PyVerifier/Iteration.cpp
)

add_dependencies(PyDialect PyDialectIncGen)

target_link_libraries(PyDialect
  PRIVATE
  MLIR
  LLVM
)

target_include_directories(PyDialect PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

# MLIRのダイアレクト実装ではRTTIを有効化
target_compile_options(PyDialect PRIVATE -frtti)
