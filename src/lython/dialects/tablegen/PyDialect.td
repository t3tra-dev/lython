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
def Py_IntType : Type<CPred<"::py::isPyIntType($_self)">, "py.int">;
def Py_FloatType : Type<CPred<"::py::isPyFloatType($_self)">, "py.float">;
def Py_BoolType : Type<CPred<"::py::isPyBoolType($_self)">, "py.bool">;
def Py_StrType : Type<CPred<"::py::isPyStrType($_self)">, "py.str">;
def Py_NoneType : Type<CPred<"::py::isPyNoneType($_self)">, "py.none">;
def Py_ObjectType : Type<CPred<"::py::isPyObjectType($_self)">, "py.object">;
def Py_TupleType : Type<CPred<"::py::isPyTupleType($_self)">, "py.tuple">;
def Py_DictType : Type<CPred<"::py::isPyDictType($_self)">, "py.dict">;
def Py_ListType : Type<CPred<"::py::isPyListType($_self)">, "py.list">;
def Py_ClassType : Type<CPred<"::py::isPyClassType($_self)">, "py.class">;
def Py_TypeType : Type<CPred<"::py::isPyTypeType($_self)">, "py.type">;
def Py_ProtocolType
    : Type<CPred<"::py::isPyProtocolType($_self)">, "py.protocol">;
def Py_SelfType : Type<CPred<"::py::isPySelfType($_self)">, "py.self">;
def Py_ExceptionType
    : Type<CPred<"::py::isPyExceptionType($_self)">, "py.exception">;
def Py_ExceptionCellType
    : Type<CPred<"::py::isPyExceptionCellType($_self)">, "py.exception_cell">;
def Py_TracebackType
    : Type<CPred<"::py::isPyTracebackType($_self)">, "py.traceback">;
def Py_LocationType
    : Type<CPred<"::py::isPyLocationType($_self)">, "py.location">;
def Py_CallableType
    : Type<CPred<"::py::isCallableType($_self)">, "Callable contract">;
def Py_UnionType : Type<CPred<"::py::isPyUnionType($_self)">, "py.union">;

class Py_Op<string mnemonic, list<Trait> traits = []>
    : Op<Py_Dialect, mnemonic, traits>;

def Py_ReturnOp : Py_Op<"return", [Terminator]> {
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";
}

def Py_CallableFuncOp : Py_Op<"callable.func", [Symbol, IsolatedFromAbove]> {
  let summary = "Python level function definition";
  let description = [{
    Defines a Python-visible Callable implementation. The `function_type`
    attribute carries the complete Callable contract used by the body region.
    When `has_vararg`/`has_kwargs` are true, the final one or two inputs are
    expected to be `!py.tuple` / `!py.dict` respectively.

    v3: The `specialize` attribute enables polymorphic specialization.
    When set, multiple specialized versions of this function can be generated
    for different *args element types, enabling type-specific optimizations
    while controlling code bloat.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
      TypeAttrOf<Py_CallableType>:$function_type,
      OptionalAttr<UnitAttr>:$has_vararg, OptionalAttr<UnitAttr>:$has_kwargs,
      OptionalAttr<ArrayAttr>:$arg_names, OptionalAttr<ArrayAttr>:$kwonly_names,
      OptionalAttr<ArrayAttr>:$closure_types,
      OptionalAttr<UnitAttr>:$specialize, OptionalAttr<UnitAttr>:$maythrow,
      OptionalAttr<UnitAttr>:$nothrow);

  let regions = (region AnyRegion:$body);

  let hasVerifier = 1;
  let assemblyFormat =
      "$sym_name `:` $function_type attr-dict-with-keyword $body";
}

def Py_ClassOp
    : Py_Op<"class", [Symbol, SymbolTable, IsolatedFromAbove, NoTerminator]> {
  let summary = "Python class definition";
  let description = [{
    Defines a statically resolved class. The class symbol owns the nominal
    class type, immutable PyTypeObject-style base metadata, and optional fixed
    field schema metadata.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
      OptionalAttr<StrArrayAttr>:$base_names,
      OptionalAttr<StrArrayAttr>:$field_names,
      OptionalAttr<TypeArrayAttr>:$field_types,
      OptionalAttr<StrArrayAttr>:$method_names,
      OptionalAttr<TypeArrayAttr>:$method_types,
      OptionalAttr<StrArrayAttr>:$method_kinds);

  // A class with an empty block and no methods is a declaration: the typing
  // manifest (runtime/typing.mlir) declares the abstract container tower
  // this way. The custom parser materializes the mandatory block for `{}`.
  let regions = (region SizedRegion<1>:$body);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def Py_CallOp : Py_Op<"call"> {
  let summary = "Invoke a Python callable";
  let description = [{
    Supplies positional arguments along with parallel tuples of keyword names
    and values. This is the canonical Python call operation: keyword
    dictionaries are a lowering/runtime concern and are not represented as a
    separate dialect call shape.
  }];

  let arguments = (ins Py_CallableType:$callable, Py_TupleType:$posargs,
      Py_TupleType:$kwnames, Py_TupleType:$kwvalues);

  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat =
      "$callable `:` type($callable) `(` $posargs `:` type($posargs) `,` "
      "$kwnames `:` type($kwnames) `,` $kwvalues `:` type($kwvalues) `)` "
      "attr-dict `:` type($results)";

  let hasVerifier = 1;
}

def Py_InvokeOp : Py_Op<"invoke", [Terminator, AttrSizedOperandSegments]> {
  let summary = "Invoke a Python callable with EH successors";
  let description = [{
    EH-aware call equivalent to `py.call`. On success, control
    transfers to the normal destination. On exception, control transfers
    to the unwind destination.
  }];

  let arguments = (ins Py_CallableType:$callable, Py_TupleType:$posargs,
      Py_TupleType:$kwnames, Py_TupleType:$kwvalues,
      Variadic<AnyType>:$normalDestOperands,
      Variadic<AnyType>:$unwindDestOperands);

  let successors = (successor AnySuccessor:$normalDest,
      AnySuccessor:$unwindDest);

  let hasVerifier = 1;
}

def Py_TryOp : Py_Op<"try"> {
  let summary = "Try/Except/Finally structured control flow";

  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$try_region, AnyRegion:$except_region,
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
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";
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
  let summary = "Create a py.exception value";

  let arguments = (ins Variadic<Py_StrType>:$args);
  let results = (outs Py_ExceptionType:$result);
  let assemblyFormat = "($args^ `:` type($args))? attr-dict `:` type($result)";
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
  let arguments = (ins Py_ExceptionType:$exception,
      TypeAttrOf<Py_ClassType>:$handler);
  let results = (outs I1:$result);
  let assemblyFormat =
      "$exception `,` $handler `:` type($exception) attr-dict";
  let hasVerifier = 1;
}

def Py_MakeFunctionOp : Py_Op<"make_function", [Pure, AttrSizedOperandSegments]> {
  let summary = "Materialize a Callable value from a Python function symbol";
  let description = [{
    Produces a Callable SSA value corresponding to the referenced Python
    function symbol. The result type must match the Callable contract recorded
    on that symbol.

    v2: Optionally accepts runtime metadata:
    - defaults: Tuple of default values for positional parameters
    - kwdefaults: Dict of default values for keyword-only parameters
    - closure: Tuple of captured closure variables
    - annotations: Dict of type annotations
    - module: Module name where the function is defined
  }];

  let arguments = (ins FlatSymbolRefAttr:$target,
      Optional<Py_TupleType>:$defaults, Optional<Py_DictType>:$kwdefaults,
      Optional<Py_TupleType>:$closure, Optional<Py_DictType>:$annotations,
      Optional<Py_StrType>:$module);

  let results = (outs Py_CallableType:$result);

  let assemblyFormat =
      "$target (`defaults` `(` $defaults^ `:` type($defaults) `)`)? "
      "(`kwdefaults` `(` $kwdefaults^ `:` type($kwdefaults) `)`)? "
      "(`closure` `(` $closure^ `:` type($closure) `)`)? "
      "(`annotations` `(` $annotations^ `:` type($annotations) `)`)? "
      "(`module` `(` $module^ `:` type($module) `)`)? "
      "attr-dict `:` type($result)";

  let hasVerifier = 1;
}

def Py_CallableObjectOp : Py_Op<"callable.object", [Pure]> {
  let summary = "Create a Callable reference to an existing function symbol";
  let description = [{
    Produces a lightweight callable handle without attaching defaults or
    closure state. This is ideal for simply referencing an existing Python
    function definition.
  }];

  let arguments = (ins FlatSymbolRefAttr:$target);

  let results = (outs Py_CallableType:$result);

  let assemblyFormat = "$target attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def Py_AwaitOp : Py_Op<"await"> {
  let summary = "Await a statically typed awaitable value";
  let arguments = (ins AnyType:$awaitable);
  let results = (outs AnyType:$result);
  let assemblyFormat =
      "$awaitable attr-dict `:` type($awaitable) `->` type($result)";
  let hasVerifier = 1;
}

def Py_AsyncNextOp : Py_Op<"async.next"> {
  let summary = "Advance an async iterator one step";
  let description = [{
    Represents one step of the Python async iteration protocol after
    `__anext__` has been resolved statically. The operand is the concrete
    awaitable returned by `__anext__`. The operation awaits it and maps
    `StopAsyncIteration` to a false valid bit, matching `py.next`'s
    element/valid split for synchronous iteration.
  }];

  let arguments = (ins AnyType:$awaitable);

  let results = (outs AnyType:$element, I1:$valid);

  let assemblyFormat =
      "$awaitable attr-dict `:` type($awaitable) `->` type($element)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Resolved special-method protocol operations
//===----------------------------------------------------------------------===//

def Py_EnterOp : Py_Op<"enter"> {
  let summary = "Invoke a statically resolved __enter__ contract";
  let description = [{
    Represents `manager.__enter__()` after the frontend has resolved the
    matching ContextManager contract from runtime/typing.mlir and selected the
    concrete method implementation. Lowering treats this as a static call while
    keeping the Python object-family special method visible in PyDialect.
  }];

  let arguments = (ins FlatSymbolRefAttr:$target,
      TypeAttrOf<Py_CallableType>:$callee_type, AnyType:$manager,
      OptionalAttr<UnitAttr>:$async_method);

  let results = (outs AnyType:$result);

  let assemblyFormat =
      "$target `:` $callee_type `(` $manager `:` type($manager) `)` "
      "attr-dict `->` type($result)";
  let hasVerifier = 1;
}

def Py_ExitOp : Py_Op<"exit"> {
  let summary = "Invoke a statically resolved __exit__ contract";
  let description = [{
    Represents `manager.__exit__(exc_type, exc_value, traceback)` after the
    frontend has resolved the matching ContextManager contract from
    runtime/typing.mlir and selected the concrete method implementation.
  }];

  let arguments = (ins FlatSymbolRefAttr:$target,
      TypeAttrOf<Py_CallableType>:$callee_type, AnyType:$manager,
      AnyType:$exc_type, AnyType:$exc_value, AnyType:$traceback,
      OptionalAttr<UnitAttr>:$async_method);

  let results = (outs AnyType:$result);

  let assemblyFormat =
      "$target `:` $callee_type `(` $manager `:` type($manager) `,` "
      "$exc_type `:` type($exc_type) `,` $exc_value `:` type($exc_value) `,` "
      "$traceback `:` type($traceback) `)` attr-dict `->` type($result)";
  let hasVerifier = 1;
}

def Py_AEnterOp : Py_Op<"aenter"> {
  let summary = "Invoke a statically resolved __aenter__ contract";
  let description = [{
    Represents `manager.__aenter__()` after the frontend has resolved the
    matching AsyncContextManager contract from runtime/typing.mlir and selected
    the concrete method implementation. The result is the awaitable produced by
    the special method.
  }];

  let arguments = (ins FlatSymbolRefAttr:$target,
      TypeAttrOf<Py_CallableType>:$callee_type, AnyType:$manager,
      OptionalAttr<UnitAttr>:$async_method);

  let results = (outs AnyType:$result);

  let assemblyFormat =
      "$target `:` $callee_type `(` $manager `:` type($manager) `)` "
      "attr-dict `->` type($result)";
  let hasVerifier = 1;
}

def Py_AExitOp : Py_Op<"aexit"> {
  let summary = "Invoke a statically resolved __aexit__ contract";
  let description = [{
    Represents `manager.__aexit__(exc_type, exc_value, traceback)` after the
    frontend has resolved the matching AsyncContextManager contract from
    runtime/typing.mlir and selected the concrete method implementation. The
    result is the awaitable produced by the special method.
  }];

  let arguments = (ins FlatSymbolRefAttr:$target,
      TypeAttrOf<Py_CallableType>:$callee_type, AnyType:$manager,
      AnyType:$exc_type, AnyType:$exc_value, AnyType:$traceback,
      OptionalAttr<UnitAttr>:$async_method);

  let results = (outs AnyType:$result);

  let assemblyFormat =
      "$target `:` $callee_type `(` $manager `:` type($manager) `,` "
      "$exc_type `:` type($exc_type) `,` $exc_value `:` type($exc_value) `,` "
      "$traceback `:` type($traceback) `)` attr-dict `->` type($result)";
  let hasVerifier = 1;
}

def Py_SendOp : Py_Op<"send"> {
  let summary = "Invoke a statically resolved Generator/Coroutine send contract";
  let description = [{
    Represents `receiver.send(value)` after the frontend has resolved the
    matching Generator or Coroutine contract from runtime/typing.mlir and
    selected the concrete method implementation.
  }];

  let arguments = (ins FlatSymbolRefAttr:$target,
      TypeAttrOf<Py_CallableType>:$callee_type, AnyType:$receiver,
      AnyType:$value);

  let results = (outs AnyType:$result);

  let assemblyFormat =
      "$target `:` $callee_type `(` $receiver `:` type($receiver) `,` "
      "$value `:` type($value) `)` attr-dict `->` type($result)";
  let hasVerifier = 1;
}

def Py_ThrowOp : Py_Op<"throw"> {
  let summary = "Invoke a statically resolved Generator/Coroutine throw contract";
  let description = [{
    Represents `receiver.throw(...)` after overload resolution against the
    Generator or Coroutine contract in runtime/typing.mlir.
  }];

  let arguments = (ins FlatSymbolRefAttr:$target,
      TypeAttrOf<Py_CallableType>:$callee_type, AnyType:$receiver,
      Py_TupleType:$args);

  let results = (outs AnyType:$result);

  let assemblyFormat =
      "$target `:` $callee_type `(` $receiver `:` type($receiver) `,` "
      "$args `:` type($args) `)` attr-dict `->` type($result)";
  let hasVerifier = 1;
}

def Py_CloseOp : Py_Op<"close"> {
  let summary = "Invoke a statically resolved Generator/Coroutine close contract";

  let arguments = (ins FlatSymbolRefAttr:$target,
      TypeAttrOf<Py_CallableType>:$callee_type, AnyType:$receiver);

  let results = (outs AnyType:$result);

  let assemblyFormat =
      "$target `:` $callee_type `(` $receiver `:` type($receiver) `)` "
      "attr-dict `->` type($result)";
  let hasVerifier = 1;
}

def Py_ASendOp : Py_Op<"asend"> {
  let summary = "Invoke a statically resolved AsyncGenerator asend contract";

  let arguments = (ins FlatSymbolRefAttr:$target,
      TypeAttrOf<Py_CallableType>:$callee_type, AnyType:$receiver,
      AnyType:$value, OptionalAttr<UnitAttr>:$async_method);

  let results = (outs AnyType:$result);

  let assemblyFormat =
      "$target `:` $callee_type `(` $receiver `:` type($receiver) `,` "
      "$value `:` type($value) `)` attr-dict `->` type($result)";
  let hasVerifier = 1;
}

def Py_AThrowOp : Py_Op<"athrow"> {
  let summary = "Invoke a statically resolved AsyncGenerator athrow contract";

  let arguments = (ins FlatSymbolRefAttr:$target,
      TypeAttrOf<Py_CallableType>:$callee_type, AnyType:$receiver,
      Py_TupleType:$args, OptionalAttr<UnitAttr>:$async_method);

  let results = (outs AnyType:$result);

  let assemblyFormat =
      "$target `:` $callee_type `(` $receiver `:` type($receiver) `,` "
      "$args `:` type($args) `)` attr-dict `->` type($result)";
  let hasVerifier = 1;
}

def Py_ACloseOp : Py_Op<"aclose"> {
  let summary = "Invoke a statically resolved AsyncGenerator aclose contract";

  let arguments = (ins FlatSymbolRefAttr:$target,
      TypeAttrOf<Py_CallableType>:$callee_type, AnyType:$receiver,
      OptionalAttr<UnitAttr>:$async_method);

  let results = (outs AnyType:$result);

  let assemblyFormat =
      "$target `:` $callee_type `(` $receiver `:` type($receiver) `)` "
      "attr-dict `->` type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Tuple / Dict / List construction
//===----------------------------------------------------------------------===//

def Py_TupleCreateOp : Py_Op<"tuple.create", []> {
  let summary = "Create a Python tuple from elements";
  let description = [{
    Constructs a `!py.tuple` value from a variadic list of elements.
    All elements must be represented in the tuple type. Heterogeneous tuples
    keep their static element types instead of being packed through a generic
    object top type.

    Example:
      %t = py.tuple.create %a, %b : (!py.int, !py.int) -> !py.tuple<!py.int>
      %t2 = py.tuple.create %x, %y : (!py.int, !py.str) -> !py.tuple<!py.int, !py.str>
  }];

  let arguments = (ins Variadic<AnyType>:$elements);

  let results = (outs Py_TupleType:$result);

  let assemblyFormat =
      "$elements attr-dict `:` functional-type($elements, $result)";
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

def Py_GetItemOp : Py_Op<"getitem", []> {
  let summary = "Python item access operation";
  let description = [{
    Represents `x[index]` / `x.__getitem__(index)` after the frontend has
    resolved a matching `__getitem__` contract from runtime/typing.mlir. The
    operation is intentionally container-generic; lowering selects the concrete
    tuple/list/dict ABI from the statically known receiver type.
  }];

  let arguments = (ins TypeAttrOf<Py_CallableType>:$callee_type,
      AnyType:$container, AnyType:$index);

  let results = (outs AnyType:$result);

  let assemblyFormat =
      "$callee_type `(` $container `[` $index `:` type($index) `]` `:` "
      "type($container) `)` attr-dict `->` type($result)";
  let hasVerifier = 1;
}

def Py_DictEmptyOp : Py_Op<"dict.empty", []> {
  let summary = "Create an empty Python dictionary";
  let description = [{
    Creates an empty `!py.dict<K, V>` value with the specified key and value types.

    Example:
      %d = py.dict.empty : !py.dict<!py.str, !py.int>
  }];

  let results = (outs Py_DictType:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
}

def Py_DictInsertOp : Py_Op<"dict.insert"> {
  let summary = "Insert a key-value pair into a dictionary";
  let description = [{
    Inserts or updates a key-value pair in a Python dictionary. This is a
    mutating operation: the dictionary operand keeps its identity and no new
    dictionary ownership token is produced.

    Example:
      py.dict.insert %d[%key] = %value : !py.dict<!py.str, !py.int>
  }];

  let arguments = (ins Py_DictType:$dict, AnyType:$key, AnyType:$value);

  let assemblyFormat = "$dict `:` type($dict) `[` $key `:` type($key) `]` `=` "
                       "$value `:` type($value) attr-dict";
  let hasVerifier = 1;
}

def Py_ContainsOp : Py_Op<"contains", []> {
  let summary = "Python containment operation";
  let description = [{
    Represents `item in container` after the frontend has resolved a matching
    `__contains__` contract from runtime/typing.mlir. The operation is
    container-generic; lowering selects the concrete ABI from the statically
    known receiver type.
  }];

  let arguments = (ins TypeAttrOf<Py_CallableType>:$callee_type,
      AnyType:$container, AnyType:$item);

  let results = (outs I1:$result);

  let assemblyFormat =
      "$callee_type `(` $item `in` $container `)` attr-dict `:` "
      "type($item) `,` type($container)";
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

  let arguments = (ins Py_ListType:$list, AnyType:$value);

  let assemblyFormat =
      "$list `:` type($list) `,` $value `:` type($value) attr-dict";
  let hasVerifier = 1;
}

def Py_ListRemoveOp : Py_Op<"list.remove"> {
  let summary = "Remove the first equal element from a typed Python list";
  let description = [{
    Removes the first element equal to the provided value. Equality is resolved
    statically at lowering time from the list element type.
  }];

  let arguments = (ins Py_ListType:$list, AnyType:$value);

  let assemblyFormat =
      "$list `:` type($list) `,` $value `:` type($value) attr-dict";
  let hasVerifier = 1;
}

def Py_NoneOp : Py_Op<"none", [Pure]> {
  let summary = "Produce the Python None singleton";
  let description = [{
    Materializes the canonical `None` object in the Python dialect. This is
    typically used when representing literal `None` in the IR.

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

  let arguments = (ins AnyType:$input);

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

  let arguments = (ins AnyType:$input, StrAttr:$mode);

  let results = (outs AnyType:$result);

  let assemblyFormat =
      "$input `mode` `=` $mode attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 0;
}

//===----------------------------------------------------------------------===//
// Python operation semantics
//===----------------------------------------------------------------------===//

def Py_AddOp : Py_Op<"add", []> {
  let summary = "Python addition operation";
  let description = [{
    Represents the Python `+` language operation. Method resolution (`__add__`,
    `__radd__`) is a lowering concern; this op carries only the language
    semantics and the statically declared result type.

    Example:
      %sum = py.add %a : !py.int, %b : !py.int -> !py.int
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);

  let results = (outs AnyType:$result);

  let assemblyFormat = "$lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict "
                       "`->` type($result)";
  let hasVerifier = 1;
}

def Py_StrConcat3Op : Py_Op<"str.concat3", []> {
  let summary = "Optimized three-way Python string concatenation";
  let description = [{
    Represents a proven-safe reassociation of `(a + b) + c` for statically
    typed `!py.str` values. This op is introduced only after ownership has
    been made explicit, so operand drops can be moved after the fused use
    without changing Python-level evaluation or reference-count semantics.
  }];

  let arguments = (ins Py_StrType:$lhs, Py_StrType:$middle, Py_StrType:$rhs);
  let results = (outs Py_StrType:$result);

  let assemblyFormat = "$lhs `:` type($lhs) `,` $middle `:` type($middle) "
                       "`,` $rhs `:` type($rhs) attr-dict `->` "
                       "type($result)";
  let hasVerifier = 1;
}

def Py_SubOp : Py_Op<"sub", []> {
  let summary = "Python subtraction operation";
  let description = [{
    Represents the Python `-` language operation. Method resolution (`__sub__`,
    `__rsub__`) is a lowering concern.
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);

  let results = (outs AnyType:$result);

  let assemblyFormat = "$lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict "
                       "`->` type($result)";
  let hasVerifier = 1;
}

def Py_MulOp : Py_Op<"mul", []> {
  let summary = "Python multiplication operation";
  let description = [{
    Represents the Python `*` language operation. Method resolution (`__mul__`,
    `__rmul__`) is a lowering concern.
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);

  let results = (outs AnyType:$result);

  let assemblyFormat = "$lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict "
                       "`->` type($result)";
  let hasVerifier = 1;
}

def Py_DivOp : Py_Op<"div", []> {
  let summary = "Python true division operation";
  let description = [{
    Represents the Python `/` language operation. Method resolution
    (`__truediv__`, `__rtruediv__`) is a lowering concern.
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);

  let results = (outs AnyType:$result);

  let assemblyFormat = "$lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict "
                       "`->` type($result)";
  let hasVerifier = 1;
}

def Py_FloorDivOp : Py_Op<"floordiv", []> {
  let summary = "Python floor division operation";
  let description = [{
    Represents the Python `//` language operation. Method resolution
    (`__floordiv__`, `__rfloordiv__`) is a lowering concern.
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);

  let results = (outs AnyType:$result);

  let assemblyFormat = "$lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict "
                       "`->` type($result)";
  let hasVerifier = 1;
}

def Py_ModOp : Py_Op<"mod", []> {
  let summary = "Python modulo operation";
  let description = [{
    Represents the Python `%` language operation. Method resolution (`__mod__`,
    `__rmod__`) is a lowering concern.
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);

  let results = (outs AnyType:$result);

  let assemblyFormat = "$lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict "
                       "`->` type($result)";
  let hasVerifier = 1;
}

def Py_LShiftOp : Py_Op<"lshift", []> {
  let summary = "Python left shift operation";
  let description = [{
    Represents the Python `<<` language operation. Method resolution
    (`__lshift__`, `__rlshift__`) is a lowering concern.
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);

  let results = (outs AnyType:$result);

  let assemblyFormat = "$lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict "
                       "`->` type($result)";
  let hasVerifier = 1;
}

def Py_RShiftOp : Py_Op<"rshift", []> {
  let summary = "Python right shift operation";
  let description = [{
    Represents the Python `>>` language operation. Method resolution
    (`__rshift__`, `__rrshift__`) is a lowering concern.
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);

  let results = (outs AnyType:$result);

  let assemblyFormat = "$lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict "
                       "`->` type($result)";
  let hasVerifier = 1;
}

def Py_BitAndOp : Py_Op<"bitand", []> {
  let summary = "Python bitwise and operation";
  let description = [{
    Represents the Python `&` language operation. Method resolution
    (`__and__`, `__rand__`) is a lowering concern.
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);

  let results = (outs AnyType:$result);

  let assemblyFormat = "$lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict "
                       "`->` type($result)";
  let hasVerifier = 1;
}

def Py_BitOrOp : Py_Op<"bitor", []> {
  let summary = "Python bitwise or operation";
  let description = [{
    Represents the Python `|` language operation. Method resolution (`__or__`,
    `__ror__`) is a lowering concern.
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);

  let results = (outs AnyType:$result);

  let assemblyFormat = "$lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict "
                       "`->` type($result)";
  let hasVerifier = 1;
}

def Py_BitXorOp : Py_Op<"bitxor", []> {
  let summary = "Python bitwise xor operation";
  let description = [{
    Represents the Python `^` language operation. Method resolution
    (`__xor__`, `__rxor__`) is a lowering concern.
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);

  let results = (outs AnyType:$result);

  let assemblyFormat = "$lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict "
                       "`->` type($result)";
  let hasVerifier = 1;
}

def Py_LeOp : Py_Op<"le", [Pure]> {
  let summary = "Python <= comparison";
  let description = [{
    Represents Python `<=` and returns a `!py.bool` object.
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);

  let results = (outs Py_BoolType:$result);

  let assemblyFormat = "$lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict "
                       "`->` type($result)";
  let hasVerifier = 1;
}

def Py_LtOp : Py_Op<"lt", [Pure]> {
  let summary = "Python < comparison";
  let description = [{
    Represents Python `<` and returns a `!py.bool` object.
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);

  let results = (outs Py_BoolType:$result);

  let assemblyFormat = "$lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict "
                       "`->` type($result)";
  let hasVerifier = 1;
}

def Py_GtOp : Py_Op<"gt", [Pure]> {
  let summary = "Python > comparison";
  let description = [{
    Represents Python `>` and returns a `!py.bool` object.
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);

  let results = (outs Py_BoolType:$result);

  let assemblyFormat = "$lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict "
                       "`->` type($result)";
  let hasVerifier = 1;
}

def Py_GeOp : Py_Op<"ge", [Pure]> {
  let summary = "Python >= comparison";
  let description = [{
    Represents Python `>=` and returns a `!py.bool` object.
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);

  let results = (outs Py_BoolType:$result);

  let assemblyFormat = "$lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict "
                       "`->` type($result)";
  let hasVerifier = 1;
}

def Py_EqOp : Py_Op<"eq", [Pure]> {
  let summary = "Python == comparison";
  let description = [{
    Represents Python `==` and returns a `!py.bool` object.
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);

  let results = (outs Py_BoolType:$result);

  let assemblyFormat = "$lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict "
                       "`->` type($result)";
  let hasVerifier = 1;
}

def Py_NeOp : Py_Op<"ne", [Pure]> {
  let summary = "Python != comparison";
  let description = [{
    Represents Python `!=` and returns a `!py.bool` object.
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);

  let results = (outs Py_BoolType:$result);

  let assemblyFormat = "$lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict "
                       "`->` type($result)";
  let hasVerifier = 1;
}

def Py_ReprOp : Py_Op<"repr", []> {
  let summary = "Python representation operation";
  let description = [{
    Represents Python `repr(value)` semantics. Static class/container/primitive
    resolution and unsupported-operation LyException emission are lowering
    concerns. The result is always an owned `!py.str`.
  }];

  let arguments = (ins AnyType:$input);

  let results = (outs Py_StrType:$result);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
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

  let arguments = (ins FlatSymbolRefAttr:$class_name);

  let results = (outs Py_ClassType:$result);

  let assemblyFormat = "$class_name attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def Py_ClassObjectOp : Py_Op<"class.object", [Pure]> {
  let summary = "Materialize a statically resolved class object";
  let description = [{
    Creates the class object value for a statically resolved class name. The
    result carries `!py.type<T>` where `T` is the corresponding instance type.
  }];

  let arguments = (ins FlatSymbolRefAttr:$class_name);

  let results = (outs Py_TypeType:$result);

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

  let arguments = (ins Py_ClassType:$input);

  let results = (outs Py_ClassType:$result);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

def Py_ClassUpcastOp : Py_Op<"class.upcast", []> {
  let summary = "View a class value as one of its nominal base classes";
  let description = [{
    Alias-preserving nominal upcast. The input and result are the same runtime
    object resource; only the static class view changes. The verifier requires
    the input class to be a subclass of the result class according to the
    module's `py.class` hierarchy.
  }];

  let arguments = (ins Py_ClassType:$input);

  let results = (outs Py_ClassType:$result);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

def Py_ClassRefineOp : Py_Op<"class.refine", []> {
  let summary = "View a class value as a proven runtime subclass";
  let description = [{
    Alias-preserving checked-view refinement. The operation performs no runtime
    test by itself; it records a fact already proven by frontend flow analysis
    or by an exact allocation fact. The verifier requires the result class to
    be a nominal subclass of the input class.
  }];

  let arguments = (ins Py_ClassType:$input);

  let results = (outs Py_ClassType:$result);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

def Py_ProtocolViewOp : Py_Op<"protocol.view", []> {
  let summary = "View a Python object as a statically proven protocol";
  let description = [{
    Alias-preserving structural protocol view. The input and result are the
    same runtime object resource; only the static contract changes after the
    frontend has proven conformance from the typing manifest.
  }];

  let arguments = (ins AnyType:$input);

  let results = (outs Py_ProtocolType:$result);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

def Py_ClassTestOp : Py_Op<"class.test", [Pure]> {
  let summary = "Test whether a class value has a compatible runtime class";
  let description = [{
    Runtime nominal class test over a statically class-typed value. The target
    is a class type attribute; lowering compares the object's class layout id
    against the target class and its known subclasses.
  }];

  let arguments = (ins Py_ClassType:$input, TypeAttr:$target);

  let results = (outs I1:$result);

  let assemblyFormat = "$input `,` $target `:` type($input) attr-dict";
  let hasVerifier = 1;
}

def Py_PublishOp : Py_Op<"publish", []> {
  let summary = "Publish a local value into shared ownership world";
  let description = [{
    Internal-only ownership boundary marker. The input and result keep the same
    logical type, but lowering may materialize runtime promotion for values that
    need a shared representation.
  }];

  let arguments = (ins AnyType:$input);

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

  let arguments = (ins AnyType:$object, StrAttr:$name);

  let results = (outs AnyType:$result);

  let assemblyFormat =
      "$object `[` $name `]` attr-dict `:` type($object) `->` type($result)";
  let hasVerifier = 1;
}

def Py_AttrSetOp : Py_Op<"attr.set", []> {
  let summary = "Set a statically declared field on a class instance";
  let description = [{
    Sets a statically declared field by name on a class instance.

    Example:
      py.attr.set %obj["value"] = %val : !py.class<"Foo">, !py.int
  }];

  let arguments = (ins AnyType:$object, StrAttr:$name, AnyType:$value);

  let assemblyFormat = "$object `[` $name `]` `=` $value attr-dict `:` "
                       "type($object) `,` type($value)";
  let hasVerifier = 1;
}

def Py_AttrGetLocalOp : Py_Op<"attr.get_local", []> {
  let summary = "Get a field through a proven-local class instance";
  let description = [{
    Internal zero-cost form of `py.attr.get`. This operation is only introduced
    by an optimizer pass after proving that the class instance is local and not
    shared. It preserves the source-level field access semantics but permits
    direct storage access during lowering.
  }];

  let arguments = (ins AnyType:$object, StrAttr:$name);

  let results = (outs AnyType:$result);

  let assemblyFormat =
      "$object `[` $name `]` attr-dict `:` type($object) `->` type($result)";
  let hasVerifier = 1;
}

def Py_AttrSetLocalOp : Py_Op<"attr.set_local", []> {
  let summary = "Set a field through a proven-local class instance";
  let description = [{
    Internal zero-cost form of `py.attr.set`. This operation is only introduced
    by an optimizer pass after proving that the class instance is local and not
    shared. It preserves the source-level field mutation semantics but permits
    direct storage access during lowering.
  }];

  let arguments = (ins AnyType:$object, StrAttr:$name, AnyType:$value);

  let assemblyFormat = "$object `[` $name `]` `=` $value attr-dict `:` "
                       "type($object) `,` type($value)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Iteration protocol operations
//===----------------------------------------------------------------------===//

def Py_IterOp : Py_Op<"iter", []> {
  let summary = "Produce an iterator from an iterable value";
  let description = [{
    Represents the Python `iter(x)` language operation after the frontend has
    resolved a matching `__iter__` contract from runtime/typing.mlir and
    selected the concrete method implementation. The iterator borrows its
    source: it aliases the iterable for liveness, and the concrete iteration
    state shape is recovered from this producer during iterator expansion.
  }];

  let arguments = (ins FlatSymbolRefAttr:$target,
      TypeAttrOf<Py_CallableType>:$callee_type, AnyType:$iterable,
      OptionalAttr<UnitAttr>:$returned_self);

  let results = (outs AnyType:$result);

  let assemblyFormat =
      "$target `:` $callee_type `(` $iterable `:` type($iterable) `)` "
      "attr-dict `->` type($result)";
  let hasVerifier = 1;
}

def Py_NextOp : Py_Op<"next", [AllTypesMatch<["iterator", "next"]>]> {
  let summary = "Advance an iterator one step";
  let description = [{
    Represents one step of the Python iteration protocol with linear state
    threading: the operands and results form a pure SSA chain so iteration
    composes with scf.while directly. Returns the element, an is-valid bit,
    and the advanced iterator. Exhaustion is the false bit, not an
    exception; the element lanes are null descriptors on the exhausted path
    and carry no ownership token there. A valid element result is owned.
  }];

  let arguments = (ins FlatSymbolRefAttr:$target,
      TypeAttrOf<Py_CallableType>:$callee_type, AnyType:$iterator);

  let results = (outs AnyType:$element, I1:$valid, AnyType:$next);

  let assemblyFormat = "$target `:` $callee_type `(` $iterator `:` "
                       "type($iterator) `)` attr-dict `->` type($element)";
  let hasVerifier = 1;
}

def Py_LenOp : Py_Op<"len", []> {
  let summary = "Python length operation";
  let description = [{
    Represents the Python `len(x)` language operation. `__len__` resolution
    is a typing-manifest concern: the frontend may emit this operation only
    after resolving a `__len__ -> int` contract from runtime/typing.mlir.
    Lowering then implements the operation from the receiver's concrete ABI.
    The result is an owned `!py.int`.
  }];

  let arguments = (ins TypeAttrOf<Py_CallableType>:$callee_type,
      AnyType:$input);

  let results = (outs Py_IntType:$result);

  let assemblyFormat =
      "$callee_type `(` $input `:` type($input) `)` attr-dict `->` "
      "type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Union (sum type) operations
//===----------------------------------------------------------------------===//

def Py_UnionWrapOp : Py_Op<"union.wrap", [Pure]> {
  let summary = "Inject a member value into a union";
  let description = [{
    Injects a value whose type is a member (or a subset union) of the result
    union. This is an ownership identity transform: the result is the same
    resource as the operand and no ownership token is produced or consumed.

    Example:
      %u = py.union.wrap %x : !py.str -> !py.union<!py.none, !py.str>
  }];

  let arguments = (ins AnyType:$input);

  let results = (outs Py_UnionType:$result);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

def Py_UnionTestOp : Py_Op<"union.test", [Pure]> {
  let summary = "Test whether the active union member has the given type";
  let description = [{
    Discriminates the active member of a union value. The tested member type
    is carried as a type attribute and must be a member of the operand union.
    The result is a primitive i1 intended to feed structured control flow;
    it never escapes as a Python value.

    Example:
      %is_none = py.union.test %u, !py.none : !py.union<!py.none, !py.str>
  }];

  let arguments = (ins Py_UnionType:$input, TypeAttr:$member);

  let results = (outs I1:$result);

  let assemblyFormat = "$input `,` $member `:` type($input) attr-dict";
  let hasVerifier = 1;
}

def Py_UnionUnwrapOp : Py_Op<"union.unwrap", [Pure]> {
  let summary = "Project the proven active member out of a union";
  let description = [{
    Projects the active member of a union value as the statically declared
    result type. The result type must be a member (or a subset union, the
    proven complement of a member test) of the operand union. The
    frontend may only emit this operation where the active member has been
    established (a dominating py.union.test branch); the operation performs
    no runtime re-check. Like py.union.wrap this is an ownership identity
    transform over the same resource.

    Example:
      %s = py.union.unwrap %u : !py.union<!py.none, !py.str> -> !py.str
  }];

  let arguments = (ins Py_UnionType:$input);

  let results = (outs AnyType:$result);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
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
      py.incref %obj : !py.int
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
      py.decref %obj : !py.int
  }];

  let arguments = (ins AnyType:$object);
  let assemblyFormat = "$object `:` type($object) attr-dict";
}
