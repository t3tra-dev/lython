# Python Dialect ビルド設定

# MLIRとLLVMのインクルードディレクトリ設定
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src/dialects/include)

# TableGen ルール
set(LLVM_TARGET_DEFINITIONS include/Lython/PythonDialect/PythonOps.td)
mlir_tablegen(PythonOps.h.inc -gen-op-decls -I ${CMAKE_SOURCE_DIR}/third_party/llvm-project/mlir/include)
mlir_tablegen(PythonOps.cpp.inc -gen-op-defs -I ${CMAKE_SOURCE_DIR}/third_party/llvm-project/mlir/include)
add_public_tablegen_target(MLIRPythonOpsIncGen)

set(LLVM_TARGET_DEFINITIONS include/Lython/PythonDialect/PythonDialect.td)
mlir_tablegen(PythonDialect.h.inc -gen-dialect-decls -I ${CMAKE_SOURCE_DIR}/third_party/llvm-project/mlir/include)
mlir_tablegen(PythonDialect.cpp.inc -gen-dialect-defs -I ${CMAKE_SOURCE_DIR}/third_party/llvm-project/mlir/include)
add_public_tablegen_target(MLIRPythonDialectIncGen)

# Types TableGen ルール
set(LLVM_TARGET_DEFINITIONS include/Lython/PythonDialect/PythonTypes.td)
mlir_tablegen(PythonTypes.h.inc -gen-typedef-decls -I ${CMAKE_SOURCE_DIR}/third_party/llvm-project/mlir/include)
mlir_tablegen(PythonTypes.cpp.inc -gen-typedef-defs -I ${CMAKE_SOURCE_DIR}/third_party/llvm-project/mlir/include)
add_public_tablegen_target(MLIRPythonTypesIncGen)

# ライブラリターゲット
add_mlir_dialect_library(MLIRPythonDialect
  lib/PythonDialect/PythonDialect.cpp
  lib/PythonDialect/PythonOps.cpp
  lib/PythonDialect/PythonTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/src/dialects/include/Lython

  DEPENDS
  MLIRPythonOpsIncGen
  MLIRPythonDialectIncGen
  MLIRPythonTypesIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)

target_include_directories(MLIRPythonDialect PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
