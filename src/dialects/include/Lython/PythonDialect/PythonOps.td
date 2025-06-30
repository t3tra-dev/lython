//===- PythonOps.td - Python Dialect Operations ---------*- tablegen -*-===//
//
// Python操作定義 - lowering_sampleパターンに基づく実装
//
//===----------------------------------------------------------------------===//

#ifndef PYTHON_OPS
#define PYTHON_OPS

include "Lython/PythonDialect/PythonDialect.td"
include "Lython/PythonDialect/PythonTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// Function Operations - 関数定義と戻り値
//===----------------------------------------------------------------------===//

def FuncOp : PythonOp<"func", [IsolatedFromAbove, Symbol]> {
    let summary = "Python function definition";
    let description = [{
    Python function definition. The @native attribute can specify native compilation.
    Refer to lowering_sample/native_int_add/1_pydia.mlir for implementation details.
    
    Example:
    ```mlir
    py.func @add(%a: !py.i32, %b: !py.i32) -> !py.i32 attributes {native = true} {
      %0 = py.add %a, %b : !py.i32
      py.return %0 : !py.i32
    }
    ```
  }];

    let arguments = (ins SymbolNameAttr:$sym_name, TypeAttr:$function_type,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs,
        DefaultValuedAttr<BoolAttr, "false">:$native);
    let regions = (region AnyRegion:$body);

    let builders = [OpBuilder<(ins "StringRef":$name, "Type":$type,
        CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>];

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
}

def ReturnOp : PythonOp<"return", [NoMemoryEffect, HasParent<"FuncOp">,
                                   ReturnLike, Terminator]> {
    let summary = "Python function return";
    let description = [{
    Return value from a Python function. The terminator operation for py.func.
  }];

    let arguments = (ins Variadic<AnyType>:$operands);

    let builders = [OpBuilder<(ins),
                              [{ build($_builder, $_state, std::nullopt); }]>];

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Constant Operations - 定数値
//===----------------------------------------------------------------------===//

def ConstantOp : PythonOp<"constant", [NoMemoryEffect]> {
    let summary = "Python constant value";
    let description = [{
    Python constant value. Supports integers, floating-point numbers, and booleans.
    Converts to appropriate MLIR constant operations during lowering.
    
    Example:
    ```mlir
    %0 = py.constant 42 : !py.i32
    %1 = py.constant 3.14 : !py.f64
    %2 = py.constant true : !py.bool
    ```
  }];

    let arguments = (ins AnyAttr:$value);
    let results = (outs AnyType:$result);

    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Arithmetic Operations - 算術演算
//===----------------------------------------------------------------------===//

def AddOp : PythonOp<"add", [NoMemoryEffect]> {
    let summary = "Python addition";
    let description = [{
    Python addition operation. Overloaded based on types.
    - Native types: Direct arithmetic operation (py.add -> arith.addi)
    - PyObject types: Runtime call (py.add -> rt.py_call @PyNumber_Add)
    
    Refer to lowering_sample/native_int_add and pystr_concat for details.
  }];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
}

def SubOp : PythonOp<"sub", [NoMemoryEffect]> {
    let summary = "Python subtraction";
    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);
    let hasCustomAssemblyFormat = 1;
}

def MulOp : PythonOp<"mul", [NoMemoryEffect]> {
    let summary = "Python multiplication";
    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);
    let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Comparison Operations - 比較演算
//===----------------------------------------------------------------------===//

def EqOp : PythonOp<"eq", [NoMemoryEffect]> {
    let summary = "Python equality comparison";
    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);
    let hasCustomAssemblyFormat = 1;
}

def LtOp : PythonOp<"lt", [NoMemoryEffect]> {
    let summary = "Python less than comparison";
    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);
    let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Variable Operations - 変数操作 (loop_sum参考)
//===----------------------------------------------------------------------===//

def VarOp : PythonOp<"var"> {
    let summary = "Python variable declaration";
    let description = [{
    Python variable declaration. Creates a mutable variable.
    Refer to lowering_sample/loop_sum/1_pydia.mlir for the variable pattern.
  }];

    let arguments = (ins AnyType:$initial_value);
    let results = (outs AnyType:$var);
    let hasCustomAssemblyFormat = 1;
}

def AssignOp : PythonOp<"assign"> {
    let summary = "Python variable assignment";
    let arguments = (ins AnyType:$var, AnyType:$value);
    let hasCustomAssemblyFormat = 1;
}

def LoadOp : PythonOp<"load", [Pure]> {
    let summary = "Python variable load";
    let arguments = (ins AnyType:$var);
    let results = (outs AnyType:$result);
    let hasCustomAssemblyFormat = 1;
}

#endif // PYTHON_OPS
