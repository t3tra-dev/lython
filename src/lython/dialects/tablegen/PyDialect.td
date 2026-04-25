include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def Py_Dialect : Dialect {
  let name = "py";
  let description = "Python Dialect for Lython";
  let cppNamespace = "::py";
  let hasConstantMaterializer = 0;
  let useDefaultTypePrinterParser = 0;
  let extraClassDeclaration = [{
    ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;
    void printType(::mlir::Type type, ::mlir::DialectAsmPrinter &printer) const override;
  }];
}

// Basic type predicates.
def Py_IntType     : Type<CPred<"::py::isPyIntType($_self)">,     "py.int">;
def Py_FloatType   : Type<CPred<"::py::isPyFloatType($_self)">,   "py.float">;
def Py_BoolType    : Type<CPred<"::py::isPyBoolType($_self)">,    "py.bool">;
def Py_StrType     : Type<CPred<"::py::isPyStrType($_self)">,     "py.str">;
def Py_ObjectType  : Type<CPred<"::py::isPyObjectType($_self)">,  "py.object">;
def Py_NoneType    : Type<CPred<"::py::isPyNoneType($_self)">,    "py.none">;
def Py_TupleType   : Type<CPred<"::py::isPyTupleType($_self)">,   "py.tuple">;
def Py_DictType    : Type<CPred<"::py::isPyDictType($_self)">,    "py.dict">;
def Py_ListType    : Type<CPred<"::py::isPyListType($_self)">,    "py.list">;
def Py_ClassType   : Type<CPred<"::py::isPyClassType($_self)">,   "py.class">;
def Py_ExceptionType : Type<CPred<"::py::isPyExceptionType($_self)">, "py.exception">;
def Py_TracebackType : Type<CPred<"::py::isPyTracebackType($_self)">, "py.traceback">;
def Py_LocationType  : Type<CPred<"::py::isPyLocationType($_self)">,  "py.location">;
def Py_FuncType    : Type<CPred<"::py::isPyFuncType($_self)">,    "py.func">;
def Py_FuncSigType : Type<CPred<"::py::isPyFuncSigType($_self)">, "py.funcsig">;
def Py_PrimFuncType: Type<CPred<"::py::isPyPrimFuncType($_self)">,"py.prim.func">;
def Py_CallableType: Type<CPred<"::py::isCallableType($_self)">,  "callable">;

class Py_Op<string mnemonic, list<Trait> traits = []> :
    Op<Py_Dialect, mnemonic, traits>;

def Py_ReturnOp : Py_Op<"return", [Terminator]> {
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";
}

def Py_FuncOp : Py_Op<"func", [
    Symbol,
    IsolatedFromAbove
  ]> {
  let summary = "Python level function definition";
  let description = [{
    Defines a Python-visible function object. The `function_type` attribute
    carries the MLIR function signature used within the body region. When
    `has_vararg`/`has_kwargs` are true, the final one or two inputs are
    expected to be `!py.tuple` / `!py.dict` respectively.

    v3: The `specialize` attribute enables polymorphic specialization.
    When set, multiple specialized versions of this function can be generated
    for different *args element types, enabling type-specific optimizations
    while controlling code bloat.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<Py_FuncSigType>:$function_type,
    OptionalAttr<UnitAttr>:$has_vararg,
    OptionalAttr<UnitAttr>:$has_kwargs,
    OptionalAttr<ArrayAttr>:$arg_names,
    OptionalAttr<ArrayAttr>:$kwonly_names,
    OptionalAttr<ArrayAttr>:$closure_types,
    OptionalAttr<UnitAttr>:$specialize,
    OptionalAttr<UnitAttr>:$maythrow,
    OptionalAttr<UnitAttr>:$nothrow
  );

  let regions = (region AnyRegion:$body);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 0;
}

def Py_ClassOp : Py_Op<"class", [
    Symbol,
    SymbolTable,
    IsolatedFromAbove,
    NoTerminator
  ]> {
  let summary = "Python class definition";
  let description = [{
    Defines a statically resolved class. The class symbol owns the nominal
    class type and optional fixed field schema metadata.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrArrayAttr>:$field_names,
    OptionalAttr<TypeArrayAttr>:$field_types
  );

  let regions = (region SizedRegion<1>:$body);

  let hasVerifier = 1;
}

def Py_CallOp : Py_Op<"call"> {
  let summary = "Invoke a Python function";
  let description = [{
    Calls a Python-level function.
    Positional arguments are packed into a `!py.tuple` while keyword
    arguments are carried in a `!py.dict` or `!py.none`.
  }];

  let arguments = (ins
    Py_CallableType:$callable,
    Py_TupleType:$posargs,
    AnyTypeOf<[Py_DictType, Py_NoneType]>:$kwargs,
    OptionalAttr<UnitAttr>:$vectorcall_hint
  );

  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat =
    "$callable `:` type($callable) `(` $posargs `:` type($posargs) `,` "
    "$kwargs `:` type($kwargs) `)` attr-dict `:` type($results)";

  let hasVerifier = 1;
}

def Py_CallVectorOp : Py_Op<"call.vector"> {
  let summary = "Invoke a Python callable using vectorcall-style arguments";
  let description = [{
    Supplies positional arguments along with parallel tuples of keyword names
    and values. This mirrors CPython's vectorcall protocol and avoids the need
    to materialise a dictionary at call sites.
  }];

  let arguments = (ins
    Py_CallableType:$callable,
    Py_TupleType:$posargs,
    Py_TupleType:$kwnames,
    Py_TupleType:$kwvalues,
    OptionalAttr<UnitAttr>:$vectorcall_hint
  );

  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat =
    "$callable `:` type($callable) `(` $posargs `:` type($posargs) `,` "
    "$kwnames `:` type($kwnames) `,` $kwvalues `:` type($kwvalues) `)` "
    "attr-dict `:` type($results)";

  let hasVerifier = 1;
}

def Py_InvokeOp : Py_Op<"invoke", [
    Terminator,
    AttrSizedOperandSegments
  ]> {
  let summary = "Invoke a Python callable with EH successors";
  let description = [{
    EH-aware call equivalent to `py.call.vector`. On success, control
    transfers to the normal destination. On exception, control transfers
    to the unwind destination.
  }];

  let arguments = (ins
    Py_CallableType:$callable,
    Py_TupleType:$posargs,
    Py_TupleType:$kwnames,
    Py_TupleType:$kwvalues,
    Variadic<AnyType>:$normalDestOperands,
    Variadic<AnyType>:$unwindDestOperands
  );

  let successors = (successor AnySuccessor:$normalDest,
                               AnySuccessor:$unwindDest);

  let hasVerifier = 1;
}

def Py_TryOp : Py_Op<"try"> {
  let summary = "Try/Except/Finally structured control flow";

  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$try_region,
                        AnyRegion:$except_region,
                        AnyRegion:$finally_region);

  let hasVerifier = 1;
}

def Py_TryYieldOp : Py_Op<"try.yield", [Terminator]> {
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";
}

def Py_ExceptYieldOp : Py_Op<"except.yield", [Terminator]> {
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";
}

def Py_FinallyYieldOp : Py_Op<"finally.yield", [Terminator]> {
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def Py_RaiseOp : Py_Op<"raise", [Terminator]> {
  let arguments = (ins Py_ExceptionType:$exception);
  let assemblyFormat = "$exception `:` type($exception) attr-dict";
  let hasVerifier = 1;
}

def Py_RaiseCurrentOp : Py_Op<"raise.current", [Terminator]> {
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def Py_ExceptionNewOp : Py_Op<"exception.new"> {
  let summary = "Create a py.exception with SSA payload";

  let arguments = (ins
    Py_ClassType:$type,
    Py_StrType:$message,
    Py_TupleType:$args,
    Py_ExceptionType:$cause,
    Py_ExceptionType:$context,
    Py_TracebackType:$traceback,
    Py_LocationType:$location,
    Py_DictType:$extras
  );

  let results = (outs Py_ExceptionType:$result);
  let hasVerifier = 1;
}

def Py_ExceptionNullOp : Py_Op<"exception.null", [Pure]> {
  let summary = "Create a null py.exception value";
  let results = (outs Py_ExceptionType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Py_TracebackNullOp : Py_Op<"traceback.null", [Pure]> {
  let summary = "Create a null py.traceback value";
  let results = (outs Py_TracebackType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Py_LocationCurrentOp : Py_Op<"location.current", [Pure]> {
  let summary = "Materialize the current source location payload";
  let results = (outs Py_LocationType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Py_ExceptMatchOp : Py_Op<"except.match", [Pure]> {
  let arguments = (ins Py_ExceptionType:$exception, Py_ClassType:$type);
  let results = (outs I1:$result);
  let assemblyFormat = "$exception `,` $type `:` type($exception) `,` type($type) attr-dict";
  let hasVerifier = 1;
}

def Py_NativeCallOp : Py_Op<"native_call"> {
  let summary = "Invoke a native (func.func) implementation";
  let description = [{
    Calls into a primitive, native function defined with `func.func`.
    The callee must not accept or return `!py.*` types.
  }];

  let arguments = (ins
    Py_PrimFuncType:$callee,
    Variadic<AnyType>:$args
  );

  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat =
    "$callee `:` type($callee) `(` $args `:` type($args) `)` attr-dict `:` type($results)";

  let hasVerifier = 1;
}

def Py_MakeFunctionOp : Py_Op<"make_function", [AttrSizedOperandSegments]> {
  let summary = "Materialize a !py.func value from a py.func symbol";
  let description = [{
    Produces a `!py.func` SSA value corresponding to the referenced `py.func`
    symbol. The result type must match the signature recorded on that symbol.

    v2: Optionally accepts runtime metadata:
    - defaults: Tuple of default values for positional parameters
    - kwdefaults: Dict of default values for keyword-only parameters
    - closure: Tuple of captured closure variables
    - annotations: Dict of type annotations
    - module: Module name where the function is defined
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$target,
    Optional<Py_TupleType>:$defaults,
    Optional<Py_DictType>:$kwdefaults,
    Optional<Py_TupleType>:$closure,
    Optional<Py_DictType>:$annotations,
    Optional<Py_StrType>:$module
  );

  let results = (outs Py_FuncType:$result);

  let assemblyFormat = "$target (`defaults` `(` $defaults^ `:` type($defaults) `)`)? "
                       "(`kwdefaults` `(` $kwdefaults^ `:` type($kwdefaults) `)`)? "
                       "(`closure` `(` $closure^ `:` type($closure) `)`)? "
                       "(`annotations` `(` $annotations^ `:` type($annotations) `)`)? "
                       "(`module` `(` $module^ `:` type($module) `)`)? "
                       "attr-dict `:` type($result)";

  let hasVerifier = 1;
}

def Py_FuncObjectOp : Py_Op<"func.object", [Pure]> {
  let summary = "Create a !py.func reference to an existing py.func symbol";
  let description = [{
    Produces a lightweight callable handle without attaching defaults or
    closure state. This is ideal for simply referencing an existing py.func
    definition.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$target
  );

  let results = (outs Py_FuncType:$result);

  let assemblyFormat = "$target attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def Py_MakeNativeOp : Py_Op<"make_native", [Pure]> {
  let summary = "Materialize a !py.prim.func value from a func.func symbol";
  let description = [{
    Produces a `!py.prim.func` SSA value referencing a primitive `func.func`
    symbol. The referenced function must not use any `!py.*` types.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$target
  );

  let results = (outs Py_PrimFuncType:$result);

  let assemblyFormat = "$target attr-dict `:` type($result)";

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Tuple / Dict / List construction
//===----------------------------------------------------------------------===//

def Py_TupleCreateOp : Py_Op<"tuple.create", []> {
  let summary = "Create a Python tuple from elements";
  let description = [{
    Constructs a `!py.tuple` value from a variadic list of elements.
    All elements must be of compatible types. If elements are heterogeneous,
    they should be upcast to `!py.object` first.

    Example:
      %t = py.tuple.create %a, %b : (!py.int, !py.int) -> !py.tuple<!py.int>
      %t2 = py.tuple.create %x, %y : (!py.object, !py.object) -> !py.tuple<!py.object>
  }];

  let arguments = (ins
    Variadic<AnyType>:$elements
  );

  let results = (outs Py_TupleType:$result);

  let assemblyFormat = "$elements attr-dict `:` functional-type($elements, $result)";
  let hasVerifier = 1;
}

def Py_TupleEmptyOp : Py_Op<"tuple.empty", [Pure]> {
  let summary = "Create an empty Python tuple";
  let description = [{
    Produces an empty tuple value for cases where no positional or keyword
    arguments are supplied.
  }];

  let results = (outs Py_TupleType:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def Py_DictEmptyOp : Py_Op<"dict.empty", []> {
  let summary = "Create an empty Python dictionary";
  let description = [{
    Creates an empty `!py.dict<K, V>` value with the specified key and value types.

    Example:
      %d = py.dict.empty : !py.dict<!py.str, !py.object>
  }];

  let results = (outs Py_DictType:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
}

def Py_DictInsertOp : Py_Op<"dict.insert"> {
  let summary = "Insert a key-value pair into a dictionary";
  let description = [{
    Inserts or updates a key-value pair in a Python dictionary.
    Returns the updated dictionary (note: semantically creates a new dict
    in SSA form, but may be optimized to in-place update at lowering).

    Example:
      %d2 = py.dict.insert %d[%key] = %value : !py.dict<!py.str, !py.object>
  }];

  let arguments = (ins
    Py_DictType:$dict,
    AnyType:$key,
    AnyType:$value
  );

  let results = (outs Py_DictType:$result);

  let assemblyFormat = "$dict `:` type($dict) `[` $key `:` type($key) `]` `=` $value `:` type($value) attr-dict `->` type($result)";
  let hasVerifier = 1;
}

def Py_DictGetOp : Py_Op<"dict.get", []> {
  let summary = "Get a value from a typed Python dictionary";
  let description = [{
    Loads the value associated with a key from a typed dictionary.
  }];

  let arguments = (ins
    Py_DictType:$dict,
    AnyType:$key
  );

  let results = (outs AnyType:$result);

  let assemblyFormat = "$dict `:` type($dict) `[` $key `:` type($key) `]` attr-dict `->` type($result)";
  let hasVerifier = 1;
}

def Py_ListNewOp : Py_Op<"list.new", []> {
  let summary = "Create an empty typed Python list";
  let description = [{
    Creates an empty `!py.list<T>` value. The element type is carried by the
    result type and remains available for static verification and lowering.
  }];

  let results = (outs Py_ListType:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def Py_ListAppendOp : Py_Op<"list.append"> {
  let summary = "Append an element to a typed Python list";
  let description = [{
    Appends an element whose type matches the list element type.
  }];

  let arguments = (ins
    Py_ListType:$list,
    AnyType:$value
  );

  let assemblyFormat = "$list `:` type($list) `,` $value `:` type($value) attr-dict";
  let hasVerifier = 1;
}

def Py_ListRemoveOp : Py_Op<"list.remove"> {
  let summary = "Remove the first equal element from a typed Python list";
  let description = [{
    Removes the first element equal to the provided value. Equality is resolved
    statically at lowering time from the list element type.
  }];

  let arguments = (ins
    Py_ListType:$list,
    AnyType:$value
  );

  let assemblyFormat = "$list `:` type($list) `,` $value `:` type($value) attr-dict";
  let hasVerifier = 1;
}

def Py_ListGetOp : Py_Op<"list.get", []> {
  let summary = "Get an element from a typed Python list";
  let description = [{
    Loads the element at the specified index from a typed list.
  }];

  let arguments = (ins
    Py_ListType:$list,
    AnyType:$index
  );

  let results = (outs AnyType:$result);

  let assemblyFormat = "$list `:` type($list) `[` $index `:` type($index) `]` attr-dict `->` type($result)";
  let hasVerifier = 1;
}

def Py_NoneOp : Py_Op<"none", [Pure]> {
  let summary = "Produce the Python None singleton";
  let description = [{
    Materializes the canonical `None` object in the Python dialect. This is
    typically used when supplying the absence of keyword arguments to
    `py.call` or representing literal `None` in the IR.

    Example:
      %none = py.none : !py.none
  }];

  let results = (outs Py_NoneType:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
}

def Py_StrConstantOp : Py_Op<"str.constant", []> {
  let summary = "Create a Python string literal";
  let description = [{
    Produces a `!py.str` SSA value from a compile-time string attribute.
  }];

  let arguments = (ins StrAttr:$value);
  let results = (outs Py_StrType:$result);

  let assemblyFormat = "$value attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def Py_IntConstantOp : Py_Op<"int.constant", []> {
  let summary = "Create a Python integer literal";
  let description = [{
    Produces a `!py.int` SSA value from a compile-time integer attribute.
    The value is stored as a decimal string to support arbitrary precision.

    Example:
      %0 = py.int.constant "42" : !py.int
      %1 = py.int.constant "-12345678901234567890" : !py.int
  }];

  let arguments = (ins StrAttr:$value);
  let results = (outs Py_IntType:$result);

  let assemblyFormat = "$value attr-dict `:` type($result)";
}

def Py_FloatConstantOp : Py_Op<"float.constant", []> {
  let summary = "Create a Python float literal";
  let description = [{
    Produces a `!py.float` SSA value from a compile-time floating-point attribute.
  }];

  let arguments = (ins F64Attr:$value);
  let results = (outs Py_FloatType:$result);

  let assemblyFormat = "$value attr-dict `:` type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Type casting operations
//===----------------------------------------------------------------------===//

def Py_CastFromPrimOp : Py_Op<"cast.from_prim", []> {
  let summary = "Explicitly box a primitive value to a Python object";
  let description = [{
    Converts a primitive MLIR type (e.g., i32, f64, i1) to the corresponding
    Python object type (e.g., !py.int, !py.float, !py.bool).
    This is an explicit boxing operation with no implicit conversions.

    Example:
      %py_int = py.cast.from_prim %i32_val : i32 -> !py.int
      %py_float = py.cast.from_prim %f64_val : f64 -> !py.float
      %py_bool = py.cast.from_prim %i1_val : i1 -> !py.bool
  }];

  let arguments = (ins
    AnyType:$input
  );

  let results = (outs AnyType:$result);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";

  let hasVerifier = 1;
}

def Py_CastToPrimOp : Py_Op<"cast.to_prim", [Pure]> {
  let summary = "Explicitly unbox a Python object to a primitive value";
  let description = [{
    Converts a Python object type to the corresponding primitive MLIR type.
    This is an explicit unboxing operation with no implicit conversions.
    Runtime checks may be inserted at lowering to ensure type safety. The
    mandatory `mode` attribute selects the conversion policy and must be
    one of "exact", "truncate", or "saturate" (see verifier).

    Example:
      %i32_val = py.cast.to_prim %py_int : !py.int -> i32
      %f64_val = py.cast.to_prim %py_float : !py.float -> f64
      %i1_val = py.cast.to_prim %py_bool : !py.bool -> i1
  }];

  let arguments = (ins
    AnyType:$input,
    StrAttr:$mode
  );

  let results = (outs AnyType:$result);

  let assemblyFormat = "$input `mode` `=` $mode attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 0;

}

def Py_CastIdentityOp : Py_Op<"cast.identity", [Pure]> {
  let summary = "Explicit identity cast used during lowering";
  let description = [{
    Serves as an explicit annotation that two SSA values share the same
    in-memory representation even when their types differ. This is mainly
    produced by the lowering pipeline to bridge !py.* types and their lowered
    pointer representation. At least one of the operand or result types must be
    a !py.* type.

    Example:
      %llvm = py.cast.identity %py_val : !py.int -> !llvm.ptr
      %py = py.cast.identity %llvm_val : !llvm.ptr -> !py.int
  }];

  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

def Py_UpcastOp : Py_Op<"upcast", [Pure]> {
  let summary = "Explicitly widen a Python value to !py.object";
  let description = [{
    Performs an explicit upcast from any !py.* value to the canonical
    !py.object supertype. This is required instead of relying on implicit
    conversions when a generic object is needed.
  }];

  let arguments = (ins
    AnyType:$input
  );

  let results = (outs Py_ObjectType:$result);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Arithmetic operations
//===----------------------------------------------------------------------===//

def Py_NumAddOp : Py_Op<"num.add", [SameOperandsAndResultType]> {
  let summary = "Python numeric addition";
  let description = [{
    Performs addition on Python numeric types (!py.int, !py.float).
    Both operands must have the same type, and the result has the same type.

    Example:
      %sum = py.num.add %a, %b : !py.int
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs
  );

  let results = (outs AnyType:$result);

  let assemblyFormat = "$lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict `->` type($result)";
  let hasVerifier = 1;
}

def Py_NumSubOp : Py_Op<"num.sub", [SameOperandsAndResultType]> {
  let summary = "Python numeric subtraction";
  let description = [{
    Performs subtraction on Python numeric types (!py.int, !py.float).
    Both operands must have the same type, and the result has the same type.
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs
  );

  let results = (outs AnyType:$result);

  let assemblyFormat = "$lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict `->` type($result)";
}

def Py_NumLeOp : Py_Op<"num.le", [Pure]> {
  let summary = "Python numeric <= comparison";
  let description = [{
    Compares two numeric values and returns a `!py.bool` object.
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs
  );

  let results = (outs Py_BoolType:$result);

  let assemblyFormat = "$lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict `->` type($result)";
  let hasVerifier = 1;
}

def Py_NumLtOp : Py_Op<"num.lt", [Pure]> {
  let summary = "Python numeric < comparison";
  let description = [{
    Compares two numeric values and returns a `!py.bool` object.
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs
  );

  let results = (outs Py_BoolType:$result);

  let assemblyFormat = "$lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict `->` type($result)";
  let hasVerifier = 1;
}

def Py_NumGtOp : Py_Op<"num.gt", [Pure]> {
  let summary = "Python numeric > comparison";
  let description = [{
    Compares two numeric values and returns a `!py.bool` object.
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs
  );

  let results = (outs Py_BoolType:$result);

  let assemblyFormat = "$lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict `->` type($result)";
  let hasVerifier = 1;
}

def Py_NumGeOp : Py_Op<"num.ge", [Pure]> {
  let summary = "Python numeric >= comparison";
  let description = [{
    Compares two numeric values and returns a `!py.bool` object.
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs
  );

  let results = (outs Py_BoolType:$result);

  let assemblyFormat = "$lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict `->` type($result)";
  let hasVerifier = 1;
}

def Py_NumEqOp : Py_Op<"num.eq", [Pure]> {
  let summary = "Python numeric == comparison";
  let description = [{
    Compares two numeric values and returns a `!py.bool` object.
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs
  );

  let results = (outs Py_BoolType:$result);

  let assemblyFormat = "$lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict `->` type($result)";
  let hasVerifier = 1;
}

def Py_NumNeOp : Py_Op<"num.ne", [Pure]> {
  let summary = "Python numeric != comparison";
  let description = [{
    Compares two numeric values and returns a `!py.bool` object.
  }];

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs
  );

  let results = (outs Py_BoolType:$result);

  let assemblyFormat = "$lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict `->` type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Class instantiation
//===----------------------------------------------------------------------===//

def Py_ClassNewOp : Py_Op<"class.new", []> {
  let summary = "Create a new instance of a class";
  let description = [{
    Creates a fresh instance storage for the specified statically resolved
    class. This does NOT call __init__ - that must be done separately.

    Example:
      %instance = py.class.new @MyClass : !py.class<"MyClass">
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$class_name
  );

  let results = (outs Py_ClassType:$result);

  let assemblyFormat = "$class_name attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def Py_ClassPromoteOp : Py_Op<"class.promote", []> {
  let summary = "Promote a static class value to managed storage";
  let description = [{
    Ensures that a statically typed class instance has managed lifetime and can
    safely escape into containers. Local stack-backed instances are copied into
    managed storage, while already-managed instances are retained and forwarded.
  }];

  let arguments = (ins
    Py_ClassType:$input
  );

  let results = (outs Py_ClassType:$result);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

def Py_PublishOp : Py_Op<"publish", []> {
  let summary = "Publish a local value into shared ownership world";
  let description = [{
    Internal-only ownership boundary marker. The input and result keep the same
    logical type, but lowering may materialize runtime promotion for values that
    need a shared representation.
  }];

  let arguments = (ins
    AnyType:$input
  );

  let results = (outs AnyType:$result);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Attribute access operations
//===----------------------------------------------------------------------===//

def Py_AttrGetOp : Py_Op<"attr.get", []> {
  let summary = "Get a statically declared field from a class instance";
  let description = [{
    Retrieves a statically declared field by name from a class instance.

    Example:
      %val = py.attr.get %obj["value"] : !py.class<"Foo"> -> !py.int
  }];

  let arguments = (ins
    AnyType:$object,
    StrAttr:$name
  );

  let results = (outs AnyType:$result);

  let assemblyFormat = "$object `[` $name `]` attr-dict `:` type($object) `->` type($result)";
  let hasVerifier = 1;
}

def Py_AttrSetOp : Py_Op<"attr.set", []> {
  let summary = "Set a statically declared field on a class instance";
  let description = [{
    Sets a statically declared field by name on a class instance.

    Example:
      py.attr.set %obj["value"] = %val : !py.class<"Foo">, !py.int
  }];

  let arguments = (ins
    AnyType:$object,
    StrAttr:$name,
    AnyType:$value
  );

  let assemblyFormat = "$object `[` $name `]` `=` $value attr-dict `:` type($object) `,` type($value)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Reference counting operations
//===----------------------------------------------------------------------===//

def Py_IncRefOp : Py_Op<"incref", []> {
  let summary = "Increment reference count of a Python object";
  let description = [{
    Increments the reference count of a Python object. This operation has
    side effects and cannot be eliminated by dead code elimination.

    The operation accepts NULL pointers safely - NULL inputs are ignored
    at runtime.

    Example:
      py.incref %obj : !py.object
  }];

  let arguments = (ins AnyType:$object);
  let assemblyFormat = "$object `:` type($object) attr-dict";
}

def Py_DecRefOp : Py_Op<"decref", []> {
  let summary = "Decrement reference count of a Python object";
  let description = [{
    Decrements the reference count of a Python object. When the reference
    count reaches zero, the object is deallocated.

    The operation accepts NULL pointers safely - NULL inputs are ignored
    at runtime.

    Example:
      py.decref %obj : !py.object
  }];

  let arguments = (ins AnyType:$object);
  let assemblyFormat = "$object `:` type($object) attr-dict";
}
