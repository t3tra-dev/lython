//===- PythonTypes.cpp - Python Dialect Types ---------------------------===//
//
// Python型システムのC++実装
//
//===----------------------------------------------------------------------===//

#include "Lython/PythonDialect/PythonTypes.h"
#include "Lython/PythonDialect/PythonDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::python;

//===----------------------------------------------------------------------===//
// Python Type Storage and Uniquing
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "PythonTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Types Implementation - 現在はTableGen標準実装を使用
//===----------------------------------------------------------------------===//
