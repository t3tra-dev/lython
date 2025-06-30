//===- PythonOps.h - Python Dialect Operations ----------------*- C++ -*-===//
//
// Python操作宣言ヘッダー
//
//===----------------------------------------------------------------------===//

#ifndef LYTHON_PYTHON_OPS_H
#define LYTHON_PYTHON_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Lython/PythonDialect/PythonDialect.h"
#include "Lython/PythonDialect/PythonTypes.h"

//===----------------------------------------------------------------------===//
// Python Operations
//===----------------------------------------------------------------------===//

// BytecodeOpInterface compatibility layer for MLIR 18
namespace mlir {
struct BytecodeOpInterface {
    template <typename ConcreteOp> struct Trait {};
};
struct SymbolOpInterface {
    template <typename ConcreteOp> struct Trait {};
};

// Minimal compatibility stubs
struct DialectBytecodeReader {
    LogicalResult readAttribute(Attribute &) { return success(); }
    LogicalResult readOptionalAttribute(Attribute &) { return success(); }
};
struct DialectBytecodeWriter {
    void writeAttribute(const Attribute &) {}
    void writeOptionalAttribute(const Attribute &) {}
};
} // namespace mlir

#define GET_OP_CLASSES
#include "PythonOps.h.inc"

#endif // LYTHON_PYTHON_OPS_H
