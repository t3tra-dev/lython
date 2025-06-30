//===- PythonOps.cpp - Python Dialect Operations -----------------------===//
//
// Python操作のC++実装
//
//===----------------------------------------------------------------------===//

#include "Lython/PythonDialect/PythonOps.h"
#include "Lython/PythonDialect/PythonDialect.h"
#include "Lython/PythonDialect/PythonTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::python;

//===----------------------------------------------------------------------===//
// Python Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "PythonOps.cpp.inc"

// Custom operations implementations can be added here if needed
