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

//===----------------------------------------------------------------------===//
// Type predicates
//===----------------------------------------------------------------------===//
//
// PyDialect deliberately keeps only two kinds of types as dialect-level
// primitives:
//
//   1. contract algebra terms that cannot be recovered from structural class
//      contracts alone (`!py.protocol`, `!py.union`, `!py.literal`,
//      `!py.type`, `!py.self`, `!py.typevar`, `!py.paramspec`,
//      `!py.typevartuple`, `!py.unpack`);
//   2. non-Python compiler payloads that have no typeshed surface contract
//      (`!py.location` today).
//
// Python-visible classes are not dialect primitives when typeshed can describe
// them. `object`, `type`, `int`, `float`, `complex`, `str`, `bytes`,
// `bytearray`, `memoryview`, `bool`, `slice`, `tuple`, `list`, `dict`, `set`,
// `frozenset`, `range`, `property`, `BaseException`, and `TracebackType` all
// have contracts in typeshed (`stdlib/builtins.pyi` and `stdlib/types.pyi`).
// Dialect values therefore carry manifest contract terms for those classes
// instead of builtin-specific dialect types.
//
// Likewise, `NoneType`, `Callable`, `Awaitable`, `Generator`,
// `ContextManager`, `Sequence`, `Mapping`, and similar Python-level concepts
// are manifest contracts. A function object is just an object whose contract
// contains `__call__`; `NoneType` is a final manifest class whose `__bool__`
// returns `Literal[False]`. The textual `!py.callable<..., returns = ...>`
// spelling is the canonical parse/print form of the Callable protocol
// contract; it exists so nested Callable and variadic argument packs can round
// trip through MLIR assembly without introducing a second callable primitive.
def Py_ContractType
    : Type<CPred<"::py::isPyContractType($_self)">, "Python contract term">;

// Contract algebra primitives. These are type-level terms, not Python class
// objects. In particular, `!py.literal` represents `typing.Literal[...]`; the
// singleton `False` or `None` is still a value, while its exact static type is a
// Literal term that can be widened to a manifest class contract.
def Py_TypeType : Type<CPred<"::py::isPyTypeType($_self)">, "py.type">;
def Py_ProtocolType
    : Type<CPred<"::py::isPyProtocolType($_self)">, "py.protocol">;
def Py_LiteralType
    : Type<CPred<"::py::isPyLiteralType($_self)">, "py.literal">;
def Py_SelfType : Type<CPred<"::py::isPySelfType($_self)">, "py.self">;
def Py_TypeVarType
    : Type<CPred<"::py::isPyTypeVarType($_self)">, "py.typevar">;
def Py_ParamSpecType
    : Type<CPred<"::py::isPyParamSpecType($_self)">, "py.paramspec">;
def Py_TypeVarTupleType
    : Type<CPred<"::py::isPyTypeVarTupleType($_self)">,
           "py.typevartuple">;
def Py_UnpackType
    : Type<CPred<"::py::isPyUnpackType($_self)">, "py.unpack">;
def Py_UnionType : Type<CPred<"::py::isPyUnionType($_self)">, "py.union">;

// Non-Python compiler payloads. These are not user-visible Python classes and
// are not expected to appear in typeshed.
def Py_LocationType
    : Type<CPred<"::py::isPyLocationType($_self)">, "py.location">;

class Py_Op<string mnemonic, list<Trait> traits = []>
    : Op<Py_Dialect, mnemonic, traits>;

def Py_ReturnOp : Py_Op<"return", [Terminator]> {
  let arguments = (ins Variadic<Py_ContractType>:$operands);
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";
}

def Py_ClassOp
    : Py_Op<"class", [Symbol, SymbolTable, IsolatedFromAbove, NoTerminator]> {
  let summary = "Python class definition";
  let description = [{
    Defines a nominal Python class contract and the storage schema known to the
    compiler. Methods are recorded as manifest protocol contracts, not as a
    dialect Callable type. Builtin and typeshed-derived classes may therefore
    share the same representation as user classes: a nominal identity plus
    structural method contracts.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
      OptionalAttr<StrArrayAttr>:$base_names,
      OptionalAttr<StrArrayAttr>:$field_names,
      OptionalAttr<TypeArrayAttr>:$field_types,
      OptionalAttr<TypeArrayAttr>:$field_contract_types,
      OptionalAttr<StrArrayAttr>:$method_names,
      OptionalAttr<TypeArrayAttr>:$method_contracts,
      OptionalAttr<StrArrayAttr>:$method_kinds,
      OptionalAttr<StrArrayAttr>:$method_symbols);

  // A class with an empty block and no methods is a declaration: the typing
  // manifest (runtime/typing.mlir) declares the abstract container tower
  // this way. The custom parser materializes the mandatory block for `{}`.
  let regions = (region SizedRegion<1>:$body);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def Py_CallOp : Py_Op<"call"> {
  let summary = "Invoke an object with a resolved __call__ contract";
  let description = [{
    Canonical Python call operation. The callee is an arbitrary Python object
    value, but this op is only well-formed after the frontend/solver has proven
    that the value's contract contains `__call__` and has attached that
    protocol contract as `call_contract`.

    This keeps Callable as a structural manifest contract rather than a dialect
    type. Positional arguments and keyword names/values are ordinary Python
    values whose contracts must instantiate `builtins.tuple`; TableGen does not
    model them with a dedicated tuple primitive.
  }];

  let arguments = (ins TypeAttrOf<Py_ProtocolType>:$call_contract,
      Py_ContractType:$callable, Py_ContractType:$posargs,
      Py_ContractType:$kwnames, Py_ContractType:$kwvalues);

  let results = (outs Variadic<Py_ContractType>:$results);

  let assemblyFormat =
      "$call_contract `(` $callable `:` type($callable) `,` "
      "$posargs `:` type($posargs) `,` $kwnames `:` type($kwnames) `,` "
      "$kwvalues `:` type($kwvalues) `)` attr-dict `:` type($results)";

  let hasVerifier = 1;
}

def Py_InvokeOp : Py_Op<"invoke", [Terminator]> {
  let summary = "Invoke an object with a resolved __call__ contract and EH";
  let description = [{
    EH-aware equivalent of `py.call`. The callee must have a manifest-resolved
    `__call__` protocol contract; on success control transfers to the normal
    destination, and on exception control transfers to the unwind destination.
  }];

  let arguments = (ins TypeAttrOf<Py_ProtocolType>:$call_contract,
      Py_ContractType:$callable, Py_ContractType:$posargs,
      Py_ContractType:$kwnames, Py_ContractType:$kwvalues,
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

def Py_YieldValueOp : Py_Op<"yield_value"> {
  let summary = "Suspend a source generator and publish one yielded value";
  let description = [{
    Represents the Python `yield value` expression inside a generator body
    after TypeSystem has classified the enclosing callable as
    `Generator[Y, S, R]`. The operand is the yielded `Y` value. The result is
    the value received when the generator is resumed through `send`, whose
    contract is `S`.

    Lowering owns the frame state machine: this op is deliberately independent
    of the concrete `types.GeneratorType` storage layout so the frontend does
    not duplicate runtime frame semantics.
  }];

  let arguments = (ins Py_ContractType:$value);
  let results = (outs Py_ContractType:$sent);
  let assemblyFormat =
      "$value `:` type($value) attr-dict `->` type($sent)";
}

def Py_YieldFromOp : Py_Op<"yield_from"> {
  let summary = "Delegate a source generator yield stream to a proven iterable";
  let description = [{
    Represents the Python `yield from value` expression after TypeSystem has
    proven the yielded element type from a Generator, Iterator, or Iterable
    contract. The selected protocol evidence is carried as
    `yield_from_contract`; the operand is the delegated source. The result is
    the delegated iterator's completion value; the currently accepted source
    lowering slice only materializes `None` for statement-position delegation
    and keeps externally observable `StopIteration.value` support out of scope.
  }];

  let arguments = (ins TypeAttrOf<Py_ProtocolType>:$yield_from_contract,
      Py_ContractType:$source);
  let results = (outs Py_ContractType:$result);
  let assemblyFormat =
      "$yield_from_contract `(` $source `:` type($source) `)` "
      "attr-dict `->` type($result)";
  let hasVerifier = 1;
}

def Py_RaiseOp : Py_Op<"raise", [Terminator]> {
  let description = [{
    Raises a Python value proven by the manifest to be compatible with
    `builtins.BaseException`. `BaseException` is a typeshed contract, not a
    dialect exception type.

    `cause` carries the `raise ... from <expr>` operand (the raised value's
    `__cause__`); the `from_none` attribute records `raise ... from None`,
    which suppresses implicit `__context__` display without a cause object.
  }];

  let arguments = (ins Py_ContractType:$exception,
                       Optional<Py_ContractType>:$cause,
                       UnitAttr:$from_none);
  let assemblyFormat = "$exception `:` type($exception) "
                       "(`from` $cause^ `:` type($cause))? attr-dict";
  let hasVerifier = 1;
}

def Py_RaiseCurrentOp : Py_Op<"raise.current", [Terminator]> {
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def Py_LocationCurrentOp : Py_Op<"location.current", [Pure]> {
  let summary = "Materialize the current source location payload";
  let results = (outs Py_LocationType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Resolved Python value materialization
//===----------------------------------------------------------------------===//
//
// These ops are intentionally more general than legacy builtin-specific
// constructors. The dialect should not grow `py.func`, `py.list`, `py.dict`,
// or similar surface-class carriers when the manifest can describe those
// contracts. Instead, the frontend/solver first resolves the binding or
// aggregate contract, then uses the minimal value materializer below.

def Py_BindingRefOp : Py_Op<"binding.ref"> {
  let summary = "Materialize a solver-resolved Python object binding";
  let description = [{
    Represents a Python object value whose scope and alias resolution has
    already happened before PyDialect emission. This is not a dynamic name
    lookup primitive and it is not a Callable/function dialect type: if the
    result can be called, that fact must be expressed by the result contract and
    consumed through `py.call`.

    Nested functions and lambdas use the same operation with capture operands:
    the binding names the lowered callable target, while captures are ordinary
    Python values that initialize the callable object's closure evidence. The
    dialect still has no dedicated function/coroutine carrier type.
  }];

  let arguments = (ins StrAttr:$binding,
      Variadic<Py_ContractType>:$captures);
  let results = (outs Py_ContractType:$result);
  let assemblyFormat =
      "$binding (`(` $captures^ `:` type($captures) `)`)? attr-dict `:` "
      "type($result)";
  let hasVerifier = 1;
}

def Py_PackOp : Py_Op<"pack"> {
  let summary = "Materialize a resolved aggregate Python value";
  let description = [{
    Constructs a Python value whose shape is fully determined by its result
    contract, for example a tuple of call arguments, a list literal, a tuple
    literal, or a dict literal. Dict payloads are encoded as alternating
    key/value operands. Lowering must select the concrete ABI path from the
    manifest contract rather than from a family of container-specific dialect
    ops.
  }];

  let arguments = (ins Variadic<Py_ContractType>:$values);
  let results = (outs Py_ContractType:$result);
  let assemblyFormat = "($values^ `:` type($values))? attr-dict `->` type($result)";
  let hasVerifier = 1;
}

def Py_ExceptMatchOp : Py_Op<"except.match", [Pure]> {
  let summary = "Match an EH payload against a Python exception type object";
  let description = [{
    Tests whether a Python exception object matches a handler declared as a
    Python type object contract, e.g. `type[BaseException]`. The handler is not
    a dialect class carrier; class identity and type-object typing come from
    manifest contracts.
  }];

  let arguments = (ins Py_ContractType:$exception,
      TypeAttrOf<Py_TypeType>:$handler);
  let results = (outs I1:$result);
  let assemblyFormat = "$exception `,` $handler `:` type($exception) attr-dict";
  let hasVerifier = 1;
}

def Py_ExceptCurrentMatchOp : Py_Op<"except.current_match", []> {
  let summary = "Match and consume the current Python EH payload";
  let description = [{
    Tests the active runtime exception against a handler declared as a Python
    type object contract. This op is intentionally separate from
    `py.except.match`: landingpad-based EH keeps the current exception in the
    runtime TLS until a handler claims it. A successful match consumes that
    current exception; a failed match leaves it available for rethrow or for a
    later handler.
  }];

  let arguments = (ins TypeAttrOf<Py_TypeType>:$handler);
  let results = (outs I1:$result);
  let assemblyFormat = "$handler attr-dict";
  let hasVerifier = 1;
}

def Py_ExceptCurrentValueOp : Py_Op<"except.current_value", []> {
  let summary = "Borrow the current Python EH payload as a handler value";
  let description = [{
    Materializes the active runtime exception as a statically typed handler
    binding for `except T as e`. This op does not consume the current
    exception; handler completion still owns the final discard/rethrow decision.
    The result type must be the instance contract named by the handler type
    object.
  }];

  let arguments = (ins TypeAttrOf<Py_TypeType>:$handler);
  let results = (outs Py_ContractType:$result);
  let assemblyFormat = "$handler attr-dict `->` type($result)";
  let hasVerifier = 1;
}

def Py_AwaitOp : Py_Op<"await"> {
  let summary = "Await an object with a resolved __await__ contract";
  let description = [{
    Represents Python `await x` after resolving `x.__await__` from the manifest.
    Awaitable is therefore a structural contract, not a coro/task dialect type.
    The result type is the return value obtained by driving the iterator
    produced by `__await__`.
  }];

  let arguments = (ins TypeAttrOf<Py_ProtocolType>:$await_contract,
      Py_ContractType:$awaitable);
  let results = (outs Py_ContractType:$result);
  let assemblyFormat =
      "$await_contract `(` $awaitable `:` type($awaitable) `)` "
      "attr-dict `->` type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Resolved special-method protocol operations
//===----------------------------------------------------------------------===//

def Py_NewOp : Py_Op<"new"> {
  let summary = "Invoke a statically resolved __new__ contract";
  let description = [{
    Represents the allocation phase of Python class instantiation after
    resolving `cls.__new__` from runtime/typing.mlir. This is not the legacy
    `ClassNewOp`: the class object is a `type[T]` contract value, the selected
    `__new__` signature is carried as `new_contract`, and the result is the
    object returned by `__new__`.

    Instance construction is modeled as `py.new` followed by `py.init` when
    the returned object is compatible with the requested class, matching
    Python's `type.__call__` semantics without reintroducing a builtin class
    carrier type.
  }];

  let arguments = (ins OptionalAttr<FlatSymbolRefAttr>:$target,
      TypeAttrOf<Py_ProtocolType>:$new_contract, Py_TypeType:$class_object,
      Py_ContractType:$posargs, Py_ContractType:$kwnames,
      Py_ContractType:$kwvalues);

  let results = (outs Py_ContractType:$instance);

  let assemblyFormat =
      "$new_contract `(` $class_object `:` type($class_object) `,` "
      "$posargs `:` type($posargs) `,` $kwnames `:` type($kwnames) `,` "
      "$kwvalues `:` type($kwvalues) `)` attr-dict `->` type($instance)";
  let hasVerifier = 1;
}

def Py_InitOp : Py_Op<"init"> {
  let summary = "Invoke a statically resolved __init__ contract";
  let description = [{
    Represents the initialization phase of Python class instantiation after
    resolving `instance.__init__` from runtime/typing.mlir. The result is the
    Python-level return value of `__init__`; the verifier should require it to
    be `Literal[None]`. The constructed expression result remains the instance
    produced by `py.new`.

    Keeping `__init__` as a dedicated special-method op lets lowering select
    the concrete initializer from manifest evidence instead of treating
    construction as an opaque `py.call` of `type.__call__`.
  }];

  let arguments = (ins OptionalAttr<FlatSymbolRefAttr>:$target,
      TypeAttrOf<Py_ProtocolType>:$init_contract, Py_ContractType:$instance,
      Py_ContractType:$posargs, Py_ContractType:$kwnames,
      Py_ContractType:$kwvalues);

  let results = (outs Py_LiteralType:$result);

  let assemblyFormat =
      "$init_contract `(` $instance `:` type($instance) `,` "
      "$posargs `:` type($posargs) `,` $kwnames `:` type($kwnames) `,` "
      "$kwvalues `:` type($kwvalues) `)` attr-dict `->` type($result)";
  let hasVerifier = 1;
}

def Py_EnterOp : Py_Op<"enter"> {
  let summary = "Invoke a statically resolved __enter__ contract";
  let description = [{
    Represents `manager.__enter__()` after the frontend has resolved the
    matching ContextManager contract from runtime/typing.mlir and selected the
    concrete method implementation. Lowering treats this as a static call while
    keeping the Python object-family special method visible in PyDialect.
  }];

  let arguments = (ins FlatSymbolRefAttr:$target,
      TypeAttrOf<Py_ProtocolType>:$callee_contract, Py_ContractType:$manager,
      OptionalAttr<UnitAttr>:$async_method);

  let results = (outs Py_ContractType:$result);

  let assemblyFormat =
      "$target `:` $callee_contract `(` $manager `:` type($manager) `)` "
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
      TypeAttrOf<Py_ProtocolType>:$callee_contract, Py_ContractType:$manager,
      Py_ContractType:$exc_type, Py_ContractType:$exc_value,
      Py_ContractType:$traceback, OptionalAttr<UnitAttr>:$async_method);

  let results = (outs Py_ContractType:$result);

  let assemblyFormat =
      "$target `:` $callee_contract `(` $manager `:` type($manager) `,` "
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
      TypeAttrOf<Py_ProtocolType>:$callee_contract, Py_ContractType:$manager,
      OptionalAttr<UnitAttr>:$async_method);

  let results = (outs Py_ContractType:$result);

  let assemblyFormat =
      "$target `:` $callee_contract `(` $manager `:` type($manager) `)` "
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
      TypeAttrOf<Py_ProtocolType>:$callee_contract, Py_ContractType:$manager,
      Py_ContractType:$exc_type, Py_ContractType:$exc_value,
      Py_ContractType:$traceback, OptionalAttr<UnitAttr>:$async_method);

  let results = (outs Py_ContractType:$result);

  let assemblyFormat =
      "$target `:` $callee_contract `(` $manager `:` type($manager) `,` "
      "$exc_type `:` type($exc_type) `,` $exc_value `:` type($exc_value) `,` "
      "$traceback `:` type($traceback) `)` attr-dict `->` type($result)";
  let hasVerifier = 1;
}

def Py_RoundOp : Py_Op<"round", []> {
  let summary = "Invoke a statically resolved __round__ contract";
  let description = [{
    Represents `round(x)` / `round(x, ndigits)` after the frontend has resolved
    a matching `__round__` contract from runtime/typing.mlir and selected the
    concrete method implementation. The operand list contains the receiver and,
    when present, the ndigits argument.
  }];

  let arguments = (ins FlatSymbolRefAttr:$target,
      TypeAttrOf<Py_ProtocolType>:$callee_contract,
      Variadic<Py_ContractType>:$inputs);

  let results = (outs Py_ContractType:$result);

  let assemblyFormat =
      "$target `:` $callee_contract `(` $inputs `:` type($inputs) `)` "
      "attr-dict `->` type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Resolved item/container protocol operations
//===----------------------------------------------------------------------===//

def Py_GetItemOp : Py_Op<"getitem", []> {
  let summary = "Python item access operation";
  let description = [{
    Represents `x[index]` / `x.__getitem__(index)` after the frontend has
    resolved a matching `__getitem__` contract from runtime/typing.mlir. The
    operation is intentionally container-generic; lowering selects the concrete
    tuple/list/dict ABI from the statically known receiver type.
  }];

  let arguments = (ins OptionalAttr<FlatSymbolRefAttr>:$target,
      TypeAttrOf<Py_ProtocolType>:$callee_contract,
      Py_ContractType:$container, Py_ContractType:$index);

  let results = (outs Py_ContractType:$result);

  let assemblyFormat =
      "$callee_contract `(` $container `[` $index `:` type($index) `]` `:` "
      "type($container) `)` attr-dict `->` type($result)";
  let hasVerifier = 1;
}

def Py_SetItemOp : Py_Op<"setitem", []> {
  let summary = "Python item assignment operation";
  let description = [{
    Represents `x[index] = value` / `x.__setitem__(index, value)` after the
    frontend has resolved a matching `__setitem__` contract from
    runtime/typing.mlir. The operation is container-generic; lowering selects
    the concrete ABI from the statically known receiver type.
  }];

  let arguments = (ins OptionalAttr<FlatSymbolRefAttr>:$target,
      TypeAttrOf<Py_ProtocolType>:$callee_contract,
      Py_ContractType:$container, Py_ContractType:$index,
      Py_ContractType:$value);

  // Structural mutation (`ly.structural_mutation`): one container-typed
  // result rebinds the receiver local so the (possibly reallocated)
  // representation threads through SSA like any reassignment.
  let results = (outs Variadic<Py_ContractType>:$results);

  let assemblyFormat =
      "$callee_contract `(` $container `[` $index `:` type($index) `]` `=` "
      "$value `:` type($value) `:` type($container) `)` attr-dict "
      "(`->` type($results)^)?";
  let hasVerifier = 1;
}

def Py_DelItemOp : Py_Op<"delitem", []> {
  let summary = "Python item deletion operation";
  let description = [{
    Represents `del x[index]` / `x.__delitem__(index)` after the frontend has
    resolved a matching `__delitem__` contract from runtime/typing.mlir. The
    operation is container-generic; lowering selects the concrete ABI from the
    statically known receiver type.
  }];

  let arguments = (ins OptionalAttr<FlatSymbolRefAttr>:$target,
      TypeAttrOf<Py_ProtocolType>:$callee_contract,
      Py_ContractType:$container, Py_ContractType:$index);

  let assemblyFormat =
      "$callee_contract `(` `del` $container `[` $index `:` type($index) `]` "
      "`:` type($container) `)` attr-dict";
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

  let arguments = (ins OptionalAttr<FlatSymbolRefAttr>:$target,
      TypeAttrOf<Py_ProtocolType>:$callee_contract,
      Py_ContractType:$container, Py_ContractType:$item);

  let results = (outs I1:$result);

  let assemblyFormat =
      "$callee_contract `(` $item `in` $container `)` attr-dict `:` "
      "type($item) `,` type($container)";
  let hasVerifier = 1;
}

def Py_BoolOp : Py_Op<"bool", []> {
  let summary = "Python truth-value operation";
  let description = [{
    Represents a statically resolved `__bool__` truth-value test. The operation
    returns the primitive condition bit used by control-flow; the verifier still
    checks that the resolved special method has the Python-level
    `__bool__() -> bool` contract.
  }];

  let arguments = (ins OptionalAttr<FlatSymbolRefAttr>:$target,
      TypeAttrOf<Py_ProtocolType>:$callee_contract, Py_ContractType:$input);

  let results = (outs I1:$result);

  let assemblyFormat =
      "$callee_contract `(` $input `:` type($input) `)` attr-dict";
  let hasVerifier = 1;
}

def Py_NoneOp : Py_Op<"none", [Pure]> {
  let summary = "Produce the Python None singleton";
  let description = [{
    Materializes the canonical `None` singleton. `None` is a value, not a
    dialect type; the exact static type is `Literal[None]` and can be widened
    to the manifest-defined final `NoneType` class contract when needed.

    Example:
      %none = py.none : !py.literal<None>
  }];

  let results = (outs Py_LiteralType:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def Py_BoolConstantOp : Py_Op<"bool.constant", [Pure]> {
  let summary = "Produce a Python bool singleton";
  let description = [{
    Materializes `True` or `False` as an exact Literal singleton. The Python
    `bool` class itself is the typeshed `builtins.bool` contract.
  }];

  let arguments = (ins BoolAttr:$value);
  let results = (outs Py_LiteralType:$result);

  let assemblyFormat = "$value attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def Py_StrConstantOp : Py_Op<"str.constant", []> {
  let summary = "Create a Python string literal";
  let description = [{
    Produces an exact `Literal["..."]` value from a compile-time string
    attribute. It may be widened to the typeshed `builtins.str` contract.
  }];

  let arguments = (ins StrAttr:$value);
  let results = (outs Py_LiteralType:$result);

  let assemblyFormat = "$value attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def Py_IntConstantOp : Py_Op<"int.constant", []> {
  let summary = "Create a Python integer literal";
  let description = [{
    Produces an exact integer Literal from a compile-time integer attribute.
    The value is stored as a decimal string to support arbitrary precision and
    may be widened to the typeshed `builtins.int` contract.

    Example:
      %0 = py.int.constant "42" : !py.literal<42>
      %1 = py.int.constant "-12345678901234567890" : !py.literal<-12345678901234567890>
  }];

  let arguments = (ins StrAttr:$value);
  let results = (outs Py_LiteralType:$result);

  let assemblyFormat = "$value attr-dict `:` type($result)";
}

def Py_FloatConstantOp : Py_Op<"float.constant", []> {
  let summary = "Create a Python float literal";
  let description = [{
    Produces a Python float value. Python `typing.Literal` does not support
    float literals, so the result is a manifest `builtins.float` contract value,
    not a dedicated float primitive.
  }];

  let arguments = (ins F64Attr:$value);
  let results = (outs Py_ContractType:$result);

  let assemblyFormat = "$value attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def Py_BytesConstantOp : Py_Op<"bytes.constant", []> {
  let summary = "Create a Python bytes literal";
  let description = [{
    Produces a `builtins.bytes` contract value from a compile-time byte
    string attribute (the StrAttr holds the raw bytes, NULs included).
    `typing.Literal` does support bytes, but the static surface tracks
    bytes non-literally (no bytes-driven branch folding), so the result is
    the manifest contract like the float literal.
  }];

  let arguments = (ins StrAttr:$value);
  let results = (outs Py_ContractType:$result);

  let assemblyFormat = "$value attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Type casting operations
//===----------------------------------------------------------------------===//

def Py_CastFromPrimOp : Py_Op<"cast.from_prim", []> {
  let summary = "Explicitly box a primitive value to a Python contract value";
  let description = [{
    Converts a primitive MLIR type (e.g., i32, f64, i1) to a Python value whose
    result type is a manifest contract such as `builtins.int`,
    `builtins.float`, or `builtins.bool`. The dialect does not expose
    builtin-specific carrier types.

    Example:
      %py_int = py.cast.from_prim %i32_val : i32 -> !py.protocol
      %py_float = py.cast.from_prim %f64_val : f64 -> !py.protocol
      %py_bool = py.cast.from_prim %i1_val : i1 -> !py.protocol
  }];

  let arguments = (ins AnyType:$input);

  let results = (outs Py_ContractType:$result);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";

  let hasVerifier = 1;
}

def Py_CastToPrimOp : Py_Op<"cast.to_prim", [Pure]> {
  let summary = "Explicitly unbox a Python object to a primitive value";
  let description = [{
    Converts a Python value with a manifest contract to the corresponding
    primitive MLIR type. Runtime checks may be inserted at lowering to ensure
    the contract is compatible. The mandatory `mode` attribute selects the
    conversion policy and must be one of "exact", "truncate", or "saturate".

    Example:
      %i32_val = py.cast.to_prim %py_int : !py.protocol -> i32
      %f64_val = py.cast.to_prim %py_float : !py.protocol -> f64
      %i1_val = py.cast.to_prim %py_bool : !py.protocol -> i1
  }];

  let arguments = (ins Py_ContractType:$input, StrAttr:$mode);

  let results = (outs AnyType:$result);

  let assemblyFormat =
      "$input `mode` `=` $mode attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 0;
}

//===----------------------------------------------------------------------===//
// Resolved Python operator semantics
//===----------------------------------------------------------------------===//
//
// These operations preserve Python source-level operator spelling while making
// method resolution an earlier, manifest-driven responsibility. The selected
// dunder method is carried by `method_name` and `callee_contract`; lowering must
// not rediscover `__add__`/`__radd__`/`__lt__`/... from ad hoc type tests.
//
// `target` is optional because builtin class contracts can lower directly from
// their resolved contract, while user-defined methods can name the concrete
// symbol.
class Py_UnaryMethodOp<string mnemonic, Type resultType = Py_ContractType,
                       list<Trait> traits = []>
    : Py_Op<mnemonic, traits> {
  let summary = "Resolved Python unary special-method operation";
  let description = [{
    Represents a Python unary operator after the frontend/solver has selected
    the exact special method contract from runtime/typing.mlir. This keeps
    `__neg__`, `__pos__`, and `__invert__` on the same manifest-driven path as
    binary operators instead of encoding them as ad hoc rewrites.
  }];

  let arguments = (ins OptionalAttr<FlatSymbolRefAttr>:$target,
      StrAttr:$method_name, TypeAttrOf<Py_ProtocolType>:$callee_contract,
      Py_ContractType:$input);

  let results = (outs resultType:$result);

  let assemblyFormat =
      "$method_name `:` $callee_contract `(` $input `:` type($input) `)` "
      "attr-dict `->` type($result)";
  let hasVerifier = 1;
}

class Py_BinaryMethodOp<string mnemonic, Type resultType = Py_ContractType,
                        list<Trait> traits = []>
    : Py_Op<mnemonic, traits> {
  let summary = "Resolved Python binary special-method operation";
  let description = [{
    Represents a Python binary operator after the frontend/solver has selected
    the exact special method contract from runtime/typing.mlir. The result type
    is the manifest-instantiated return contract. Reflected methods are not a
    second lowering path: they are represented by a different `method_name` and
    `callee_contract` on the same op shape.
  }];

  let arguments = (ins OptionalAttr<FlatSymbolRefAttr>:$target,
      StrAttr:$method_name, TypeAttrOf<Py_ProtocolType>:$callee_contract,
      Py_ContractType:$lhs, Py_ContractType:$rhs);

  let results = (outs resultType:$result);

  let assemblyFormat =
      "$method_name `:` $callee_contract `(` $lhs `:` type($lhs) `,` "
      "$rhs `:` type($rhs) `)` attr-dict `->` type($result)";
  let hasVerifier = 1;
}

def Py_NegOp : Py_UnaryMethodOp<"neg">;
def Py_PosOp : Py_UnaryMethodOp<"pos">;
def Py_InvertOp : Py_UnaryMethodOp<"invert">;

def Py_AddOp : Py_BinaryMethodOp<"add">;
def Py_SubOp : Py_BinaryMethodOp<"sub">;
def Py_MulOp : Py_BinaryMethodOp<"mul">;
def Py_DivOp : Py_BinaryMethodOp<"div">;
def Py_FloorDivOp : Py_BinaryMethodOp<"floordiv">;
def Py_ModOp : Py_BinaryMethodOp<"mod">;
def Py_LShiftOp : Py_BinaryMethodOp<"lshift">;
def Py_RShiftOp : Py_BinaryMethodOp<"rshift">;
def Py_BitAndOp : Py_BinaryMethodOp<"bitand">;
def Py_BitOrOp : Py_BinaryMethodOp<"bitor">;
def Py_BitXorOp : Py_BinaryMethodOp<"bitxor">;
def Py_PowOp : Py_BinaryMethodOp<"pow">;

def Py_LeOp : Py_BinaryMethodOp<"le", Py_ContractType, [Pure]>;
def Py_LtOp : Py_BinaryMethodOp<"lt", Py_ContractType, [Pure]>;
def Py_GtOp : Py_BinaryMethodOp<"gt", Py_ContractType, [Pure]>;
def Py_GeOp : Py_BinaryMethodOp<"ge", Py_ContractType, [Pure]>;
def Py_EqOp : Py_BinaryMethodOp<"eq", Py_ContractType, [Pure]>;
def Py_NeOp : Py_BinaryMethodOp<"ne", Py_ContractType, [Pure]>;

def Py_ReprOp : Py_Op<"repr", []> {
  let summary = "Python representation operation";
  let description = [{
    Represents Python `repr(value)` semantics after the frontend has selected
    the `__repr__` contract. When `target` is present it names the statically
    resolved method implementation. The result contract must instantiate
    `builtins.str`.
  }];

  let arguments = (ins OptionalAttr<FlatSymbolRefAttr>:$target,
      TypeAttrOf<Py_ProtocolType>:$callee_contract, Py_ContractType:$input);

  let results = (outs Py_ContractType:$result);

  let assemblyFormat =
      "$callee_contract `(` $input `:` type($input) `)` attr-dict `->` "
      "type($result)";
  let hasVerifier = 1;
}

def Py_IntOp : Py_Op<"int", []> {
  let summary = "Python integer conversion operation";
  let description = [{
    Represents Python `int(value)` semantics after the frontend has selected
    the `__int__` contract (str parsing and float truncation are the manifest
    `__int__` runtime methods of those contracts). When `target` is present it
    names the statically resolved method implementation. The result contract
    must instantiate `builtins.int`.
  }];

  let arguments = (ins OptionalAttr<FlatSymbolRefAttr>:$target,
      TypeAttrOf<Py_ProtocolType>:$callee_contract, Py_ContractType:$input);

  let results = (outs Py_ContractType:$result);

  let assemblyFormat =
      "$callee_contract `(` $input `:` type($input) `)` attr-dict `->` "
      "type($result)";
  let hasVerifier = 1;
}

def Py_FloatOp : Py_Op<"float", []> {
  let summary = "Python float conversion operation";
  let description = [{
    Represents Python `float(value)` semantics after the frontend has selected
    the `__float__` contract (the correctly rounded int conversion is the
    manifest `__float__` runtime method of `builtins.int`). When `target` is
    present it names the statically resolved method implementation. The result
    contract must instantiate `builtins.float`.
  }];

  let arguments = (ins OptionalAttr<FlatSymbolRefAttr>:$target,
      TypeAttrOf<Py_ProtocolType>:$callee_contract, Py_ContractType:$input);

  let results = (outs Py_ContractType:$result);

  let assemblyFormat =
      "$callee_contract `(` $input `:` type($input) `)` attr-dict `->` "
      "type($result)";
  let hasVerifier = 1;
}

def Py_StrOp : Py_Op<"str", []> {
  let summary = "Python string conversion operation";
  let description = [{
    Represents Python `str(value)` semantics after the frontend has selected the
    `__str__` contract. When `target` is present it names the statically
    resolved method implementation. The result contract must instantiate
    `builtins.str`.
  }];

  let arguments = (ins OptionalAttr<FlatSymbolRefAttr>:$target,
      TypeAttrOf<Py_ProtocolType>:$callee_contract, Py_ContractType:$input);

  let results = (outs Py_ContractType:$result);

  let assemblyFormat =
      "$callee_contract `(` $input `:` type($input) `)` attr-dict `->` "
      "type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Contract evidence and type objects
//===----------------------------------------------------------------------===//

def Py_TypeObjectOp : Py_Op<"type.object", [Pure]> {
  let summary = "Materialize a statically resolved Python type object";
  let description = [{
    Creates the runtime type object for a statically resolved Python contract.
    The result carries `type[T]` as `!py.type<T>`; `T` may be a manifest class
    contract, protocol contract, union, or literal term.
  }];

  let arguments = (ins TypeAttrOf<Py_ContractType>:$instance_contract);

  let results = (outs Py_TypeType:$result);

  let assemblyFormat = "$instance_contract attr-dict `:` type($result)";
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

  let arguments = (ins Py_ContractType:$input);

  let results = (outs Py_ContractType:$result);

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

  let arguments = (ins Py_ContractType:$input);

  let results = (outs Py_ContractType:$result);

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

  let arguments = (ins Py_ContractType:$input);

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

  let arguments = (ins Py_ContractType:$input, TypeAttrOf<Py_ContractType>:$target);

  let results = (outs I1:$result);

  let assemblyFormat = "$input `,` $target `:` type($input) attr-dict";
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
      %val = py.attr.get %obj["value"] : !py.protocol -> !py.protocol
  }];

  let arguments = (ins Py_ContractType:$object, StrAttr:$name);

  let results = (outs Py_ContractType:$result);

  let assemblyFormat =
      "$object `[` $name `]` attr-dict `:` type($object) `->` type($result)";
  let hasVerifier = 1;
}

def Py_AttrSetOp : Py_Op<"attr.set", []> {
  let summary = "Set a statically declared field on a class instance";
  let description = [{
    Sets a statically declared field by name on a class instance.

    Example:
      py.attr.set %obj["value"] = %val : !py.protocol, !py.protocol
  }];

  let arguments = (ins Py_ContractType:$object, StrAttr:$name,
      Py_ContractType:$value);

  let assemblyFormat = "$object `[` $name `]` `=` $value attr-dict `:` "
                       "type($object) `,` type($value)";
  let hasVerifier = 1;
}

def Py_GlobalGetOp : Py_Op<"global.get", []> {
  let summary = "Read a module-level global variable";
  let description = [{
    Reads the current value of a module-level global variable named `name`.
    Module globals are backed by process-lifetime storage (see the runtime
    lowering), so the read is a plain load; for an unboxed integer global the
    load is async-signal-safe (no allocation). The declared type of the global
    is the result contract.

    Example:
      %v = py.global.get "LIMIT" : !py.contract<"builtins.int">
  }];

  let arguments = (ins StrAttr:$name);
  let results = (outs Py_ContractType:$result);
  let assemblyFormat = "$name attr-dict `:` type($result)";
}

def Py_GlobalSetOp : Py_Op<"global.set", []> {
  let summary = "Write a module-level global variable";
  let description = [{
    Stores `value` into the module-level global named `name`. For an unboxed
    integer global this is an unbox-and-store; the store is async-signal-safe.

    Example:
      py.global.set "LIMIT" = %v : !py.contract<"builtins.int">
  }];

  let arguments = (ins StrAttr:$name, Py_ContractType:$value);
  let assemblyFormat = "$name `=` $value attr-dict `:` type($value)";
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
      TypeAttrOf<Py_ProtocolType>:$callee_contract, Py_ContractType:$iterable,
      OptionalAttr<UnitAttr>:$returned_self);

  let results = (outs Py_ContractType:$result);

  let assemblyFormat =
      "$target `:` $callee_contract `(` $iterable `:` type($iterable) `)` "
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
      TypeAttrOf<Py_ProtocolType>:$callee_contract, Py_ContractType:$iterator);

  let results = (outs Py_ContractType:$element, I1:$valid,
      Py_ContractType:$next);

  let assemblyFormat = "$target `:` $callee_contract `(` $iterator `:` "
                       "type($iterator) `)` attr-dict `->` type($element)";
  let hasVerifier = 1;
}

def Py_AIterOp : Py_Op<"aiter", []> {
  let summary = "Invoke a statically resolved __aiter__ contract";
  let description = [{
    Represents `async_iterable.__aiter__()` after manifest contract resolution.
    `AsyncIterable` and `AsyncIterator` are typing.mlir contracts; this op only
    records the selected special method and its instantiated protocol contract.
  }];

  let arguments = (ins FlatSymbolRefAttr:$target,
      TypeAttrOf<Py_ProtocolType>:$callee_contract,
      Py_ContractType:$async_iterable, OptionalAttr<UnitAttr>:$returned_self);

  let results = (outs Py_ContractType:$result);

  let assemblyFormat =
      "$target `:` $callee_contract `(` $async_iterable `:` "
      "type($async_iterable) `)` attr-dict `->` type($result)";
  let hasVerifier = 1;
}

def Py_ANextOp : Py_Op<"anext"> {
  let summary = "Invoke a statically resolved __anext__ contract";
  let description = [{
    Represents `async_iterator.__anext__()` after manifest contract resolution.
    The result is the awaitable returned by `__anext__`; awaiting it is expressed
    by `py.await`, and `StopAsyncIteration` handling remains ordinary exception
    control flow rather than a separate async-next dialect primitive.
  }];

  let arguments = (ins FlatSymbolRefAttr:$target,
      TypeAttrOf<Py_ProtocolType>:$callee_contract,
      Py_ContractType:$async_iterator);

  let results = (outs Py_ContractType:$awaitable);

  let assemblyFormat =
      "$target `:` $callee_contract `(` $async_iterator `:` "
      "type($async_iterator) `)` attr-dict `->` type($awaitable)";
  let hasVerifier = 1;
}

def Py_LenOp : Py_Op<"len", []> {
  let summary = "Python length operation";
  let description = [{
    Represents the Python `len(x)` language operation. `__len__` resolution
    is a typing-manifest concern: the frontend may emit this operation only
    after resolving a `__len__ -> int` contract from runtime/typing.mlir.
    Lowering then implements the operation from the receiver's resolved
    contract evidence. The result contract must instantiate `builtins.int`.
  }];

  let arguments = (ins OptionalAttr<FlatSymbolRefAttr>:$target,
      TypeAttrOf<Py_ProtocolType>:$callee_contract, Py_ContractType:$input);

  let results = (outs Py_ContractType:$result);

  let assemblyFormat =
      "$callee_contract `(` $input `:` type($input) `)` attr-dict `->` "
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
      %u = py.union.wrap %x
          : !py.protocol -> !py.union<!py.literal<None>, !py.protocol>
  }];

  let arguments = (ins Py_ContractType:$input);

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
      %is_none = py.union.test %u, !py.literal<None>
          : !py.union<!py.literal<None>, !py.protocol>
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
      %s = py.union.unwrap %u
          : !py.union<!py.literal<None>, !py.protocol> -> !py.protocol
  }];

  let arguments = (ins Py_UnionType:$input);

  let results = (outs Py_ContractType:$result);

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
      py.incref %obj : !py.protocol
  }];

  let arguments = (ins Py_ContractType:$object);
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
      py.decref %obj : !py.protocol
  }];

  let arguments = (ins Py_ContractType:$object);
  let assemblyFormat = "$object `:` type($object) attr-dict";
}
