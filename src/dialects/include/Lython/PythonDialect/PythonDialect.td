//===- PythonDialect.td - Python Dialect Definition -----*- tablegen -*-===//

#ifndef PYTHON_DIALECT
#define PYTHON_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Python Dialect Definition
//===----------------------------------------------------------------------===//

def PythonDialect : Dialect {
    let name = "py";
    let summary = "Python Dialect for Lython";
    let cppNamespace = "::mlir::python";
}

//===----------------------------------------------------------------------===//
// Base Python Operation Definition
//===----------------------------------------------------------------------===//

class PythonOp<string mnemonic, list<Trait> traits = []>
    : Op<PythonDialect, mnemonic, traits>;

#endif // PYTHON_DIALECT
