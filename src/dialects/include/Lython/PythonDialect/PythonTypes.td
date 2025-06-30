//===- PythonTypes.td - Python Dialect Types ------------*- tablegen -*-===//
//
// Python型システム定義 - lowering_sampleに基づく実装
//
//===----------------------------------------------------------------------===//

#ifndef PYTHON_TYPES
#define PYTHON_TYPES

include "Lython/PythonDialect/PythonDialect.td"

//===----------------------------------------------------------------------===//
// Python Type Base Class
//===----------------------------------------------------------------------===//

class PythonType<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<PythonDialect, name, traits> {
    let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Native Value Types (値セマンティクス) - lowering_sampleのネイティブ型
//===----------------------------------------------------------------------===//

def Py_I32Type : PythonType<"PythonI32", "i32"> {
    let summary = "32-bit signed integer (native)";
    let description = [{
    Native 32-bit signed integer type. A high-performance integer type used within @native functions.
    During lowering, it is directly converted to the standard MLIR i32 type.
    
    Example: `!py.i32` -> `i32` -> LLVM `i32`
  }];
}

def Py_I64Type : PythonType<"PythonI64", "i64"> {
    let summary = "64-bit signed integer (native)";
    let description = [{
    Native 64-bit signed integer type. Value semantics.
  }];
}

def Py_F32Type : PythonType<"PythonF32", "f32"> {
    let summary = "32-bit floating point (native)";
    let description = [{
    Native 32-bit floating-point type. Value semantics.
  }];
}

def Py_F64Type : PythonType<"PythonF64", "f64"> {
    let summary = "64-bit floating point (native)";
    let description = [{
    Native 64-bit floating-point type. Value semantics.
  }];
}

def Py_BoolType : PythonType<"PythonBool", "bool"> {
    let summary = "Boolean type (native)";
    let description = [{
    Native boolean type. Value semantics.
    During lowering, it is converted to the MLIR i1 type.
  }];
}

//===----------------------------------------------------------------------===//
// PyObject Reference Types (参照セマンティクス) - 文字列等
//===----------------------------------------------------------------------===//

def Py_ObjType : PythonType<"PythonObj", "obj"> {
    let summary = "PyObject reference type";
    let description = [{
    Reference type to PyObject. Reference semantics.
    Managed by GC and holds dynamic type information.
    During lowering, it is converted to runtime calls.
    
    Example: `!py.obj<str>` -> `rt.py_call @PyStr_*`
  }];

    let parameters = (ins "mlir::Type":$elementType);
    let assemblyFormat = "`<` $elementType `>`";
}

//===----------------------------------------------------------------------===//
// Container Types - リスト等の複合型
//===----------------------------------------------------------------------===//

def Py_ListType : PythonType<"PythonList", "list"> {
    let summary = "Python list type";
    let description = [{
    Python list type. Holds an element type as a type parameter.
    During lowering, it is converted to a memref<?xT> type.
    
    Example: `!py.list<i32>` -> `memref<?xi32>`
  }];

    let parameters = (ins "mlir::Type":$elementType);
    let assemblyFormat = "`<` $elementType `>`";
}

#endif // PYTHON_TYPES
