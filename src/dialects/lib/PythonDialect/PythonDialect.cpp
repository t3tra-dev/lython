//===- PythonDialect.cpp - Python Dialect Implementation ----------------===//

#include "Lython/PythonDialect/PythonDialect.h"
#include "Lython/PythonDialect/PythonOps.h"
#include "Lython/PythonDialect/PythonTypes.h"

using namespace mlir;
using namespace mlir::python;

//===----------------------------------------------------------------------===//
// Python Dialect
//===----------------------------------------------------------------------===//

#include "PythonDialect.cpp.inc"

void PythonDialect::initialize() {
  // まず基本型のみ登録 (パラメータなし)
  addTypes<
    PythonI32Type,
    PythonI64Type,
    PythonF32Type,
    PythonF64Type,
    PythonBoolType
  >();

  // 操作の登録
  addOperations<
#define GET_OP_LIST
#include "PythonOps.cpp.inc"
  >();
}
