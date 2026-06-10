# Lython Emitter

This module is the C++ Python frontend emitter. The previous Python
`frontend`/`visitors` implementation has been removed; source input now flows
through the C++ parser and this emitter before entering the lowering pipeline.

Input:

- `lython::parser::Node`, a Python `ast`-shaped tree produced by the C++
  parser.

Output:

- top-level `py` dialect `mlir::ModuleOp` suitable for the existing lowering
  pipeline.
- Text output, when needed for diagnostics or snapshots, is produced by MLIR's
  printer API rather than by manually concatenating MLIR assembly strings.

Current status:

- `lyc jit <file.py>` always uses this emitter for Python input.
- The legacy Python `ast.parse`/visitor CLI fallback has been removed. Missing
  language support must be implemented in the C++ parser/emitter path instead
  of adding a Python fallback.
- The implemented slice covers imports as no-op static declarations, `pass`,
  expression statements, annotated assignment for names/class fields/typed dict
  subscripts, augmented assignment for names/class fields/typed dict
  subscripts, literal tuple/list unpack assignment and statically finite tuple
  RHS unpack assignment including starred rest lists, `if`, nothrow
  context-manager `with` lowering for static
  class managers, `return`, `raise` of statically typed `!py.exception`
  expressions or statically known builtin exception classes including
  `from None` context suppression,
  active-exception validation for bare `raise`, multi-handler `try`/`except`
  lowering with statically known builtin exception classes, including CPython
  3.14 parenthesized and unparenthesized tuple exception handlers plus bare
  default handlers, and builtin
  exception constructors such as `RuntimeError(...)`/`TypeError(...)`
  represented as verified `BaseException` subclasses, statically
  annotated functions,
  native functions, direct static calls, class definitions with statically
  resolved C3 inheritance, class constructors, statically resolved methods,
  read-only `global` declarations for static primitive integer/float module
  constants,
  `assert` lowering to statically typed `AssertionError` construction with an
  optional string message,
  `if` expressions/statements with primitive loop-carried values and direct
  same-type primitive local introduction across both branches,
  `del` for primitive/static names, list literals including starred display
  expansion from statically finite tuple/list sources, static finite list
  slices, selected list methods, static finite tuple/list concatenation and
  repetition,
  primitive `for` loops over `range(...)` including static non-zero positive
  and negative steps, static integer literal bounds, loop `else`,
  `break`/`continue`, and reuse of existing primitive loop targets,
  single `range(...)` list comprehensions, static finite-list copies of the
  form `[xs[i] for i in range(len(xs))]`, and
  `list(<generator expression>)` eager materialization with static integer
  literal or primitive integer bounds and filters, single `range(...)`
  primitive typed
  dict comprehensions with static integer literal or primitive integer bounds
  and filters, primitive typed dict
  literals and subscript access/update, expected-type lowering for annotated
  empty `list[...]` and `dict[K, V]` literals and no-argument
  `list()`/`dict()`/homogeneous `tuple()` constructors in expected-type
  positions, finite-sequence `list(...)`/`tuple(...)` constructors including
  expected-type empty sequence list construction, static `list(range(...))`
  and `tuple(range(...))` materialization, static `len(range(...))`, builtin
  `bool()`/`bool(x)` via the shared truthiness lowering including `None` and
  statically finite list truthiness, limited `int(...)`/`float(...)`,
  `str(...)`, and `repr(...)` lowering for statically supported values, static
  keyword arguments for direct
  functions, class constructors, and class methods, static `*args` expansion
  from tuple/list literals and static `**kwargs` expansion from dict literals
  for calls, statically bounded finite tuple slices, direct `print(...)` with
  default space-separated varargs and statically known string `sep`/`end`,
  static `len(...)` on finite tuple/list/string values, use of those static
  lengths as primitive `range(...)` bounds, primitive
  integer/float constructors, explicit `from_prim(...)`/`to_prim(...)`
  primitive casts, tensor/matrix constructors, basic constants, boolean
  short-circuiting, ownership-neutral named expressions, arithmetic including
  primitive integer division/bitwise invert and statically known non-negative
  primitive integer powers, primitive float remainder, tensor arithmetic/matmul,
  singleton `is`/`is not`
  comparisons, basic/debug f-strings with `!r`/`!s` conversion and empty format
  specs, pattern matching with primitive carried values and direct same-type
  primitive local introduction across exhaustive cases, `while` loops with
  loop-carried primitive and statically typed Python object locals,
  primitive chained
  comparisons, annotation aliases
  for `List`/`Tuple`/`Dict`, homogeneous
  `tuple[T, ...]`, async annotations for `Coroutine`/`Task`/`Future` including
  `asyncio.Task[T]`, type-only imports from `typing` and `collections.abc`,
  the static `lython` facade imports that map `Int`/`Float`/`Tensor` style
  primitive names to `lyrt.prim` and `native`/`from_prim`/`to_prim` to `lyrt`,
  CPython 3.14 `type` statements, `TypeAlias` annotated aliases, and
  TypeVar-only generic type aliases with static annotation-time
  specialization, CPython function and parameter `type_comment` parsing for
  unannotated functions/methods, nested functions
  with closure capture for statically materializable primitive/object values,
  expected-type lambda lowering for statically typed `Callable[[...], Ret]`
  values,
  definition-time callable defaults/kwdefaults via `py.make_function` metadata,
  `Callable[[...], Ret]` value calls when the callable target metadata is
  statically recoverable from a local `py.func.object` or `py.make_function`,
  static specialization for functions that accept known `Callable` arguments,
  return-summary lowering for functions that return a known callable or one of
  their callable parameters, and direct materialization of simple nested
  closure callables returned from functions or methods.
- Unsupported AST forms fail in the emitter with source diagnostics instead of
  falling back to CPython.
- Nested closure rebinding after the nested function definition and shaped
  primitive captures are still rejected rather than falling back to the Python
  visitor. Module-scope lambda captures are also rejected because Python globals
  have late-binding semantics that should not be silently value-captured.
  Callable parameters still require statically known callable arguments or
  defaults so the emitter can specialize the callee; it must not generate
  dynamic vectorcall IR that the lowering pipeline cannot prove. Returned
  method closure summaries currently reject `self`/class-object captures until
  class object closure ownership and promotion are proven in the same model.
  Function decorators other than `@native(gc="none")`, async function
  decorators, method decorators, and class decorators are parsed as CPython
  3.14 AST but rejected until decorator application has a static lowering
  model. Class keyword arguments such as `metaclass=...` are also rejected
  rather than silently ignored.
  `with` currently requires pre-bound static class context-manager values,
  nothrow `__enter__`/`__exit__`, and a body without exception or unstructured
  control flow; exception suppression needs a richer `__exit__` exception
  payload path before it can be lowered soundly. Global rebinding and global
  deletion are rejected until module storage has a proven write-back ABI.
  `print(file=...)` is accepted only for static `file=None`, and
  `print(flush=...)` is accepted only for static `False`/`None`; other values
  are still rejected because the current host I/O boundary only models stdout
  writes without an explicit flush operation. Generic aliases can use `TypeVar`
  defaults, a final `TypeVarTuple` pack, and fixed-position `ParamSpec` packs
  used in `Callable[P, R]`; all are resolved during static specialization.
  Generic function, async function, method, and class `type_params` are parsed
  from CPython 3.14 syntax but rejected until the emitter has a proven static
  specialization model for those definitions.

Frontend rule:

- New frontend behavior should be added here first.
- Unsupported Python syntax should fail with parser or emitter diagnostics
  rather than falling back to CPython or a Python visitor layer.
