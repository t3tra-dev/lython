//===- PythonTypes.h - Python Dialect Types -------------------*- C++ -*-===//
//
// Python型システム ヘッダーファイル
//
//===----------------------------------------------------------------------===//

#ifndef LYTHON_PYTHON_TYPES_H
#define LYTHON_PYTHON_TYPES_H

#include "mlir/IR/Types.h"
#include "mlir/IR/TypeSupport.h"

//===----------------------------------------------------------------------===//
// Python Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "PythonTypes.h.inc"

#endif // LYTHON_PYTHON_TYPES_H
