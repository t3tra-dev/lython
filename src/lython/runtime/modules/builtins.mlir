// Contract manifest AND runtime implementation for the builtin functions --
// CPython's Python/bltinmodule.c (print, len, abs, divmod, pow, ord, chr, ...).
//
// Signature source (1:1 correspondence target):
//   https://github.com/python/typeshed/blob/main/stdlib/builtins.pyi
//
// The builtin TYPES that builtins.pyi also declares live where CPython keeps
// them: runtime/objects/<name>.mlir is Objects/<name>object.c (`int` is
// objects/long.mlir, `str` objects/unicode.mlir, the exception hierarchy
// objects/exceptions.mlir), and what they share with the interpreter is
// runtime/python/<name>.mlir, Python/<name>.c (errors, pyhash, dtoa,
// formatter_unicode). Each file states its own deviations from CPython.
//
// Conventions (shared by all contract manifests):
//   - `!py.contract<"builtins.int">` is a nominal manifest contract.
//   - `!py.contract<"$T">` names a generic parameter from `ly.typing.params`.
//   - Method contracts are Callable terms in `method_contracts`; hand-written
//     manifests may use the typeshed-shaped `!py.protocol<"Callable", ...>`
//     spelling with `!py.callable<...>` for nested/variadic terms.
//
// The remaining declared-but-unimplemented names on the builtin contracts are
// inventoried in rfc/contract-audit.md, which also records what the
// differential audit found to agree; do not add a name to a contract without
// an implementation behind it.

module attributes {
  ly.typing.manifest,
  // Manifest Callable contracts for builtin free functions. These are the
  // single trusted source for these signatures; the emitter's seedBuiltins
  // reads them here instead of constructing the contracts in C++.
  // "builtins.abs" appears once per overload: duplicate names merge into an
  // overload set on the table side (CPython abs is int->int / float->float /
  // complex->float, which one generic T->T contract cannot express -- the
  // complex result is a float).
  ly.typing.function_names = ["builtins.print", "builtins.len", "builtins.hash", "builtins.sorted", "builtins.abs", "builtins.abs", "builtins.abs", "builtins.divmod", "builtins.pow", "builtins.ord", "builtins.chr", "builtins.hex", "builtins.oct", "builtins.bin", "builtins.input", "builtins.list", "builtins.tuple", "builtins.id"],
  ly.typing.function_contracts = [
    !py.callable<[], vararg = !py.contract<"builtins.tuple", [!py.contract<"builtins.object">]>, returns = [!py.literal<None>]>,
    !py.callable<[!py.contract<"builtins.object">], returns = [!py.contract<"builtins.int">]>,
    !py.callable<[!py.contract<"builtins.object">], returns = [!py.contract<"builtins.int">]>,
    !py.callable<[!py.protocol<"Iterable", [!py.typevar<"T">]>], returns = [!py.contract<"builtins.list", [!py.typevar<"T">]>]>,
    !py.callable<[!py.contract<"builtins.int">], returns = [!py.contract<"builtins.int">]>,
    !py.callable<[!py.contract<"builtins.float">], returns = [!py.contract<"builtins.float">]>,
    !py.callable<[!py.contract<"builtins.complex">], returns = [!py.contract<"builtins.float">]>,
    !py.callable<[!py.contract<"builtins.int">, !py.contract<"builtins.int">], returns = [!py.contract<"builtins.tuple", [!py.contract<"builtins.int">, !py.contract<"builtins.int">]>]>,
    !py.callable<[!py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">], returns = [!py.contract<"builtins.int">]>,
    !py.callable<[!py.contract<"builtins.str">], returns = [!py.contract<"builtins.int">]>,
    !py.callable<[!py.contract<"builtins.int">], returns = [!py.contract<"builtins.str">]>,
    !py.callable<[!py.contract<"builtins.int">], returns = [!py.contract<"builtins.str">]>,
    !py.callable<[!py.contract<"builtins.int">], returns = [!py.contract<"builtins.str">]>,
    !py.callable<[!py.contract<"builtins.int">], returns = [!py.contract<"builtins.str">]>,
    !py.callable<[!py.contract<"builtins.str">], returns = [!py.contract<"builtins.str">]>,
    !py.callable<[!py.contract<"builtins.list", [!py.typevar<"T">]>], returns = [!py.contract<"builtins.list", [!py.typevar<"T">]>]>,
    !py.callable<[!py.contract<"builtins.list", [!py.typevar<"T">]>], returns = [!py.contract<"builtins.tuple", [!py.typevar<"T">]>]>,
    !py.callable<[!py.contract<"builtins.object">], returns = [!py.contract<"builtins.int">]>
  ]
} {
  func.func private @__ly_pending_push(%mark: i64, %kind: i64, %value: i64) -> i64
  func.func private @__ly_pending_pop(%index: i64)
  // ===== declared here, defined in another runtime file or built by the lowering =====
  func.func private @LyObject_IdentityKey(%word: i64) -> i64 attributes {ly.runtime.contract = "builtins.object", ly.runtime.primitive = "identity_key"}
  func.func private @__ly_unicode_from_valid_utf8(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]}
  func.func private @LyBaseException_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.BaseException", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"}
  func.func private @LyBaseException_New(%class_word: i64 {ly.runtime.class_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class, ly.runtime.contract = "builtins.BaseException", ly.runtime.initializer = "__new__"}
  func.func private @LyEH_ThrowException(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "raise"}
  func.func private @LyList_Copy(%self: memref<5xi64> {ly.ownership.object_header}) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.list", ly.runtime.method = "copy", ly.runtime.result_contract = "builtins.list"}
  func.func private @LyLong_DecRef(%header: memref<2xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.int", ly.runtime.deallocator}
  func.func private @LyLong_FromI64(%value: i64 {ly.runtime.default_i64 = 0 : i64}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class, ly.runtime.contract = "builtins.int", ly.runtime.initializer = "__new__"}
  func.func private @LyLong_Mod(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_header: memref<2xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__mod__"}
  func.func private @LyLong_Mul(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_header: memref<2xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__mul__"}
  func.func private @LyUnicode_DecRef(%header: memref<2xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.str", ly.runtime.deallocator}
  func.func private @LyUnicode_FromBytes(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class, ly.runtime.contract = "builtins.str", ly.runtime.initializer = "__new__"}
  func.func private @LyUnicode_FromI64(%value: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]}
  func.func private @LyUnicode_Print(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) attributes {ly.runtime.contract = "builtins.str", ly.runtime.primitive = "print"}
  func.func private @__ly_fmt_copy_bytes(%dst: memref<?xi8>, %dpos: i64, %src: memref<?xi8>, %len: i64) -> i64
  func.func private @__ly_list_copy_alloc(%src_len: i64, %src_items: memref<?xi64>) -> memref<5xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.list"], ly.ownership.owned_results = [0]}
  func.func private @__ly_list_items(%self: memref<5xi64>) -> memref<?xi64> attributes {ly.runtime.contract = "builtins.list", ly.runtime.interior_word, ly.runtime.primitive = "items_view"}
  func.func private @__ly_long_floor_divmod(%lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>) attributes {ly.ownership.owned_result_contracts = ["builtins.int", "builtins.int"], ly.ownership.owned_results = [0, 1]}
  func.func private @__ly_long_format_pow2(%meta: memref<2xi64>, %digits: memref<?xi32>, %bits: i64, %letter: i8) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]}
  func.func private @__ly_long_operand_view(%meta: memref<2xi64>, %digits: memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
  func.func private @__ly_long_parts(%header: memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
  func.func private @__ly_long_raise_division_by_zero()
  func.func private @__ly_long_view_as_i64(%meta: memref<2xi64>, %digits: memref<?xi32>) -> i64
  func.func private @__ly_long_view_fits_i64(%meta: memref<2xi64>, %digits: memref<?xi32>) -> i1
  func.func private @__ly_raise_static_message(%class_word: i64, %message: memref<?xi8>, %length: i64)
  func.func private @__ly_sort_slots(%items: memref<?xi64>, %len: i64)
  func.func private @__ly_tuple_alloc(%length: i64) -> memref<5xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.tuple"], ly.ownership.owned_results = [0]}
  func.func private @__ly_tuple_copy_alloc(%src_len: i64, %src_items: memref<?xi64>) -> memref<5xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.tuple"], ly.ownership.owned_results = [0]}
  func.func private @__ly_tuple_items(%self: memref<5xi64>) -> memref<?xi64> attributes {ly.runtime.contract = "builtins.tuple", ly.runtime.interior_word, ly.runtime.primitive = "items_view"}
  func.func private @__ly_tuple_store_long(%items: memref<?xi64>, %slot: index, %h: memref<2xi64>)
  func.func private @__ly_unicode_count(%header: memref<2xi64>, %bytes: memref<?xi8>) -> i64
  func.func private @__ly_unicode_get(%bytes: memref<?xi8>, %width: i64, %i: index) -> i64
  func.func private @__ly_unicode_single(%cp: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]}
  func.func private @__ly_unicode_utf8_length(%header: memref<2xi64>, %bytes: memref<?xi8>) -> i64
  func.func private @__ly_unicode_width(%header: memref<2xi64>) -> i64

  // ===========================================================
  // Lowering strategies (transform dialect carrier) -- layer 4 of the
  // transformation stack: target-selected,
  // schedule-shaped transformations that ship WITH the module instead of in
  // C++. Per-op lowerings stay in RuntimeBundleLowerer (layers 1-2); stage
  // ordering stays in LoweringPipeline.cpp (layer 3).
  //
  // A modules/<name>.mlir may declare lowering strategies as a nested
  // strategy-library module marked `transform.with_named_sequence`. The
  // runtime import skips it (it never merges into the user module); the
  // lowering pipeline's strategy interpreter collects these libraries from
  // the embedded manifests and applies matched sequences to the user module
  // via transform::applyTransforms -- matchers select payload
  // ops (transform.foreach_match), actions rewrite them
  // (transform.apply_patterns / transform.include).
  // ===========================================================
  module @__lython_lowering_strategies attributes {transform.with_named_sequence} {
    // Post-import cleanup: the user module was canonicalized before the
    // runtime implementations were imported (pipeline phase 6 runs before
    // phase 8), so the freshly imported runtime function bodies reach the
    // runtime lowering uncanonicalized. Re-run canonicalization + CSE over
    // the whole module once the implementations are in.
    transform.named_sequence @__lython_strategy_post_import_cleanup(%root: !transform.any_op) {
      %canonicalized = transform.apply_registered_pass "canonicalize" to %root : (!transform.any_op) -> !transform.any_op
      %cleaned = transform.apply_registered_pass "cse" to %canonicalized : (!transform.any_op) -> !transform.any_op
      transform.yield
    }
  }

  func.func private @LyBuiltin_Len() -> memref<2xi64> attributes {ly.runtime.builtin = "len", ly.runtime.builtin_lowering = "method", ly.runtime.builtin_method = "__len__", ly.runtime.contract = "builtins.object", ly.runtime.primitive = "builtin_len", ly.runtime.result_contract = "builtins.int"}

  // id(x) (builtin_id): the word `is` compares -- the box's entity through
  // identity_key -- so `id(a) == id(b)` answers what `a is b` answers. The
  // value types `is` refuses are refused before a call is built.
  func.func @LyBuiltin_Id(%box: memref<5xi64>) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.builtin = "id", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.object", ly.runtime.primitive = "builtin_id", ly.runtime.result_contract = "builtins.int"} {
    %entity_slot = arith.constant 2 : index
    %entity = memref.load %box[%entity_slot] : memref<5xi64>
    %key = func.call @LyObject_IdentityKey(%entity) : (i64) -> i64
    %result = func.call @LyLong_FromI64(%key) : (i64) -> memref<2xi64>
    func.return %result : memref<2xi64>
  }

  func.func private @LyBuiltin_Hash() -> memref<2xi64> attributes {ly.runtime.builtin = "hash", ly.runtime.builtin_lowering = "method", ly.runtime.builtin_method = "__hash__", ly.runtime.contract = "builtins.object", ly.runtime.primitive = "builtin_hash", ly.runtime.result_contract = "builtins.int"}

  // Raw fd write boundary (built by RuntimeSupportBuilder): extracting the
  // payload pointer from the descriptor is the irreducibly-llvm part, so it
  // is the ONLY piece of the print path that stays out of this manifest.
  func.func private @LyHost_WriteBytes(i32, memref<?xi8>, i64)

  memref.global "private" constant @__ly_print_end : memref<1xi8> = dense<[10]>

  // print(object) after the emitter's sep-join desugar: CPython
  // builtin_print_impl's tail with objects_length == 1, file = sys.stdout
  // (fd 1) and the default end = "\n" (Python/bltinmodule.c). The str
  // rendering of each argument stays ahead of this sink (method_sink
  // dispatch / emitter desugar) because it needs per-value evidence.
  func.func @LyUnicode_PrintLine(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) attributes {ly.runtime.builtin = "print", ly.runtime.builtin_lowering = "method_sink", ly.runtime.builtin_method = "__repr__", ly.runtime.builtin_sink_contract = "builtins.str", ly.runtime.contract = "builtins.str", ly.runtime.primitive = "print_line", ly.runtime.result_contract = "types.NoneType"} {
    %stdout = arith.constant 1 : i32
    func.call @LyUnicode_Print(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> ()
    %end_static = memref.get_global @__ly_print_end : memref<1xi8>
    %end = memref.cast %end_static : memref<1xi8> to memref<?xi8>
    %end_length = arith.constant 1 : i64
    func.call @LyHost_WriteBytes(%stdout, %end, %end_length) : (i32, memref<?xi8>, i64) -> ()
    func.return
  }

  // list(xs): a fresh shallow copy of a list argument (the emitter routes
  // other iterables through the comprehension desugar instead).
  func.func @LyBuiltin_List(%self: memref<5xi64> {ly.ownership.object_header}) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "list", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.list", ly.runtime.primitive = "builtin_list", ly.runtime.result_contract = "builtins.list"} {
    %copy = func.call @LyList_Copy(%self) : (memref<5xi64>) -> memref<5xi64>
    func.return %copy : memref<5xi64>
  }

  // tuple(xs): freeze a list's items into a fresh tuple. Both contracts are now
  // one-lane handles of different widths, and the SLOT layout of the items
  // array is the same for both, so the fill loop is shared and only the
  // destination differs.
  func.func @LyBuiltin_Tuple(%self: memref<5xi64> {ly.ownership.object_header}) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "tuple", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.list", ly.runtime.primitive = "builtin_tuple", ly.runtime.result_contract = "builtins.tuple"} {
    %length_slot = arith.constant 2 : index
    %len = memref.load %self[%length_slot] : memref<5xi64>
    %items = func.call @__ly_list_items(%self) : (memref<5xi64>) -> memref<?xi64>
    %result = func.call @__ly_tuple_copy_alloc(%len, %items) : (i64, memref<?xi64>) -> memref<5xi64>
    func.return %result : memref<5xi64>
  }

  // sorted(xs): a fresh sorted list (the argument is untouched).
  // ⭐ The copy is registered while the sort runs: a comparison can raise
  // (`sorted([1, "a"])`, a user `__lt__`), and CPython's sorted releases its
  // new list before returning the error (errors.mlir, "what a native body
  // owes").
  func.func @LyBuiltin_Sorted(%self: memref<5xi64> {ly.ownership.object_header}) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "sorted", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.list", ly.runtime.primitive = "builtin_sorted", ly.runtime.result_contract = "builtins.list"} {
    %length_slot = arith.constant 2 : index
    %len = memref.load %self[%length_slot] : memref<5xi64>
    %items = func.call @__ly_list_items(%self) : (memref<5xi64>) -> memref<?xi64>
    %copy = func.call @__ly_list_copy_alloc(%len, %items) : (i64, memref<?xi64>) -> memref<5xi64>
    %copy_items = func.call @__ly_list_items(%copy) : (memref<5xi64>) -> memref<?xi64>
    %mark_slot = memref.alloca() : memref<1xi64>
    %mark_index = memref.extract_aligned_pointer_as_index %mark_slot : memref<1xi64> -> index
    %mark = arith.index_cast %mark_index : index to i64
    %copy_index = memref.extract_aligned_pointer_as_index %copy : memref<5xi64> -> index
    %copy_word = arith.index_cast %copy_index : index to i64
    %list_kind = arith.constant 2 : i64
    %pending = func.call @__ly_pending_push(%mark, %list_kind, %copy_word) : (i64, i64, i64) -> i64
    func.call @__ly_sort_slots(%copy_items, %len) : (memref<?xi64>, i64) -> ()
    func.call @__ly_pending_pop(%pending) : (i64) -> ()
    func.return %copy : memref<5xi64>
  }

  // ===== impls: numeric/string free builtins =====
  // abs(): per-class __abs__ methods plus the builtin method dispatcher.
  func.func private @LyBuiltin_Abs() -> memref<2xi64> attributes {ly.runtime.builtin = "abs", ly.runtime.builtin_lowering = "method", ly.runtime.builtin_method = "__abs__", ly.runtime.contract = "builtins.object", ly.runtime.primitive = "builtin_abs", ly.runtime.result_contract = "builtins.int"}

  // divmod(a, b) for ints: one floor-division pass, packed as (q, r).
  func.func @LyBuiltin_DivMod(%ah: memref<2xi64> {ly.ownership.object_header}, %bh: memref<2xi64> {ly.ownership.object_header}) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "divmod", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.tuple", ly.runtime.primitive = "builtin_divmod", ly.runtime.result_contract = "builtins.tuple"} {
    %am, %ad = func.call @__ly_long_parts(%ah) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %bm, %bd = func.call @__ly_long_parts(%bh) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %a_meta, %a_digits = func.call @__ly_long_operand_view(%am, %ad) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %b_meta, %b_digits = func.call @__ly_long_operand_view(%bm, %bd) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0 : i64
    %b_sign = memref.load %b_meta[%c0] : memref<2xi64>
    %b_zero = arith.cmpi eq, %b_sign, %zero : i64
    scf.if %b_zero {
      func.call @__ly_long_raise_division_by_zero() : () -> ()
    }
    %qr:2 = func.call @__ly_long_floor_divmod(%a_meta, %a_digits, %b_meta, %b_digits) : (memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>)
    %two = arith.constant 2 : i64
    %self = func.call @__ly_tuple_alloc(%two) : (i64) -> memref<5xi64>
    %items = func.call @__ly_tuple_items(%self) : (memref<5xi64>) -> memref<?xi64>
    %slot0 = arith.constant 0 : index
    %slot1 = arith.constant 1 : index
    func.call @__ly_tuple_store_long(%items, %slot0, %qr#0) : (memref<?xi64>, index, memref<2xi64>) -> ()
    func.call @__ly_tuple_store_long(%items, %slot1, %qr#1) : (memref<?xi64>, index, memref<2xi64>) -> ()
    func.return %self : memref<5xi64>
  }

  // "pow() 2nd argument cannot be negative when 3rd argument is specified"
  memref.global "private" constant @__ly_pow_msg_negative : memref<68xi8> = dense<[112, 111, 119, 40, 41, 32, 50, 110, 100, 32, 97, 114, 103, 117, 109, 101, 110, 116, 32, 99, 97, 110, 110, 111, 116, 32, 98, 101, 32, 110, 101, 103, 97, 116, 105, 118, 101, 32, 119, 104, 101, 110, 32, 51, 114, 100, 32, 97, 114, 103, 117, 109, 101, 110, 116, 32, 105, 115, 32, 115, 112, 101, 99, 105, 102, 105, 101, 100]>
  // "pow() 3rd argument cannot be 0"
  memref.global "private" constant @__ly_pow_msg_zero_mod : memref<30xi8> = dense<[112, 111, 119, 40, 41, 32, 51, 114, 100, 32, 97, 114, 103, 117, 109, 101, 110, 116, 32, 99, 97, 110, 110, 111, 116, 32, 98, 101, 32, 48]>

  // pow(base, exp, mod) for ints: square-and-multiply over the exponent's
  // 30-bit limbs, reducing modulo `mod` at every step.
  func.func @LyBuiltin_PowMod(%ah: memref<2xi64> {ly.ownership.object_header}, %eh: memref<2xi64> {ly.ownership.object_header}, %mh: memref<2xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "pow", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.int", ly.runtime.primitive = "builtin_powmod", ly.runtime.result_contract = "builtins.int"} {
    %em, %ed = func.call @__ly_long_parts(%eh) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %mm, %md = func.call @__ly_long_parts(%mh) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %exp_meta, %exp_digits = func.call @__ly_long_operand_view(%em, %ed) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %mod_meta, %mod_digits = func.call @__ly_long_operand_view(%mm, %md) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %value_error = arith.constant {ly.class_of = "builtins.ValueError"} 53 : i64
    %exp_sign = memref.load %exp_meta[%c0] : memref<2xi64>
    %exp_negative = arith.cmpi slt, %exp_sign, %zero : i64
    scf.if %exp_negative {
      %msg_static = memref.get_global @__ly_pow_msg_negative : memref<68xi8>
      %msg = memref.cast %msg_static : memref<68xi8> to memref<?xi8>
      %len = arith.constant 68 : i64
      func.call @__ly_raise_static_message(%value_error, %msg, %len) : (i64, memref<?xi8>, i64) -> ()
    }
    %mod_sign = memref.load %mod_meta[%c0] : memref<2xi64>
    %mod_zero = arith.cmpi eq, %mod_sign, %zero : i64
    scf.if %mod_zero {
      %msg_static = memref.get_global @__ly_pow_msg_zero_mod : memref<30xi8>
      %msg = memref.cast %msg_static : memref<30xi8> to memref<?xi8>
      %len = arith.constant 30 : i64
      func.call @__ly_raise_static_message(%value_error, %msg, %len) : (i64, memref<?xi8>, i64) -> ()
    }
    // result = 1 % mod; acc = base % mod.
    %one_h = func.call @LyLong_FromI64(%zero) : (i64) -> memref<2xi64>
    func.call @LyLong_DecRef(%one_h) : (memref<2xi64>) -> ()
    %c1_i64 = arith.constant 1 : i64
    %init_h = func.call @LyLong_FromI64(%c1_i64) : (i64) -> memref<2xi64>
    %result0 = func.call @LyLong_Mod(%init_h, %mh) : (memref<2xi64>, memref<2xi64>) -> memref<2xi64>
    %result0_p0_meta, %result0_p0_digits = func.call @__ly_long_parts(%result0) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    func.call @LyLong_DecRef(%init_h) : (memref<2xi64>) -> ()
    %acc0 = func.call @LyLong_Mod(%ah, %mh) : (memref<2xi64>, memref<2xi64>) -> memref<2xi64>
    %acc0_p0_meta, %acc0_p0_digits = func.call @__ly_long_parts(%acc0) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %exp_count = memref.load %exp_meta[%c1] : memref<2xi64>
    %total_bits_step = arith.constant 30 : i64
    %total_bits = arith.muli %exp_count, %total_bits_step : i64
    %total_index = arith.index_cast %total_bits : i64 to index
    %final:6 = scf.for %bit = %c0 to %total_index step %c1 iter_args(%rh = %result0, %rm = %result0_p0_meta, %rd = %result0_p0_digits, %bh2 = %acc0, %bm2 = %acc0_p0_meta, %bd2 = %acc0_p0_digits) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<2xi64>, memref<?xi32>) {
      %bit_i64 = arith.index_cast %bit : index to i64
      %limb_index_i64 = arith.divui %bit_i64, %total_bits_step : i64
      %bit_in_limb = arith.remui %bit_i64, %total_bits_step : i64
      %limb_index = arith.index_cast %limb_index_i64 : i64 to index
      %limb_i32 = memref.load %exp_digits[%limb_index] : memref<?xi32>
      %limb = arith.extui %limb_i32 : i32 to i64
      %shifted = arith.shrui %limb, %bit_in_limb : i64
      %c1_bit = arith.constant 1 : i64
      %bit_set0 = arith.andi %shifted, %c1_bit : i64
      %bit_set = arith.cmpi ne, %bit_set0, %zero : i64
      %next_r:3 = scf.if %bit_set -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) {
        %prod = func.call @LyLong_Mul(%rh, %bh2) : (memref<2xi64>, memref<2xi64>) -> memref<2xi64>
        %red = func.call @LyLong_Mod(%prod, %mh) : (memref<2xi64>, memref<2xi64>) -> memref<2xi64>
        %red_p0_meta, %red_p0_digits = func.call @__ly_long_parts(%red) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
        func.call @LyLong_DecRef(%prod) : (memref<2xi64>) -> ()
        func.call @LyLong_DecRef(%rh) : (memref<2xi64>) -> ()
        scf.yield %red, %red_p0_meta, %red_p0_digits : memref<2xi64>, memref<2xi64>, memref<?xi32>
      } else {
        scf.yield %rh, %rm, %rd : memref<2xi64>, memref<2xi64>, memref<?xi32>
      }
      %sq = func.call @LyLong_Mul(%bh2, %bh2) : (memref<2xi64>, memref<2xi64>) -> memref<2xi64>
      %sqr = func.call @LyLong_Mod(%sq, %mh) : (memref<2xi64>, memref<2xi64>) -> memref<2xi64>
      %sqr_p0_meta, %sqr_p0_digits = func.call @__ly_long_parts(%sqr) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
      func.call @LyLong_DecRef(%sq) : (memref<2xi64>) -> ()
      func.call @LyLong_DecRef(%bh2) : (memref<2xi64>) -> ()
      scf.yield %next_r#0, %next_r#1, %next_r#2, %sqr, %sqr_p0_meta, %sqr_p0_digits : memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<2xi64>, memref<?xi32>
    }
    func.call @LyLong_DecRef(%final#3) : (memref<2xi64>) -> ()
    func.return %final#0 : memref<2xi64>
  }

  // ord(): the single code point of a one-character str.
  memref.global "private" constant @__ly_ord_msg : memref<49xi8> = dense<[111, 114, 100, 40, 41, 32, 101, 120, 112, 101, 99, 116, 101, 100, 32, 97, 32, 99, 104, 97, 114, 97, 99, 116, 101, 114, 44, 32, 98, 117, 116, 32, 115, 116, 114, 105, 110, 103, 32, 111, 102, 32, 108, 101, 110, 103, 116, 104, 32]>
  memref.global "private" constant @__ly_ord_found_msg : memref<6xi8> = dense<[32, 102, 111, 117, 110, 100]>

  func.func @LyBuiltin_Ord(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "ord", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.str", ly.runtime.primitive = "builtin_ord", ly.runtime.result_contract = "builtins.int"} {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1 : i64
    %count = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %not_single = arith.cmpi ne, %count, %one : i64
    scf.if %not_single {
      // CPython names the offending length: "...but string of length 3
      // found" (Python/bltinmodule.c, builtin_ord). Stopping at "expected a
      // character" hides whether the argument was empty or too long, which
      // is the only thing the message is for.
      %type_error = arith.constant {ly.class_of = "builtins.TypeError"} 52 : i64
      %buf_s = memref.alloca() : memref<96xi8>
      %buf = memref.cast %buf_s : memref<96xi8> to memref<?xi8>
      %zero = arith.constant 0 : i64
      %prefix_s = memref.get_global @__ly_ord_msg : memref<49xi8>
      %prefix = memref.cast %prefix_s : memref<49xi8> to memref<?xi8>
      %l49 = arith.constant 49 : i64
      %a = func.call @__ly_fmt_copy_bytes(%buf, %zero, %prefix, %l49) : (memref<?xi8>, i64, memref<?xi8>, i64) -> i64
      %nh, %nb = func.call @LyUnicode_FromI64(%count) : (i64) -> (memref<2xi64>, memref<?xi8>)
      %nlen = func.call @__ly_unicode_utf8_length(%nh, %nb) : (memref<2xi64>, memref<?xi8>) -> i64
      %b = func.call @__ly_fmt_copy_bytes(%buf, %a, %nb, %nlen) : (memref<?xi8>, i64, memref<?xi8>, i64) -> i64
      func.call @LyUnicode_DecRef(%nh) : (memref<2xi64>) -> ()
      %found_s = memref.get_global @__ly_ord_found_msg : memref<6xi8>
      %found = memref.cast %found_s : memref<6xi8> to memref<?xi8>
      %l6 = arith.constant 6 : i64
      %c = func.call @__ly_fmt_copy_bytes(%buf, %b, %found, %l6) : (memref<?xi8>, i64, memref<?xi8>, i64) -> i64
      func.call @__ly_raise_static_message(%type_error, %buf, %c) : (i64, memref<?xi8>, i64) -> ()
    }
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %cp = func.call @__ly_unicode_get(%bytes, %width, %c0) : (memref<?xi8>, i64, index) -> i64
    %h = func.call @LyLong_FromI64(%cp) : (i64) -> memref<2xi64>
    func.return %h : memref<2xi64>
  }

  // "EOF when reading a line"
  memref.global "private" constant @__ly_input_eof_msg : memref<23xi8> = dense<[69, 79, 70, 32, 119, 104, 101, 110, 32, 114, 101, 97, 100, 105, 110, 103, 32, 97, 32, 108, 105, 110, 101]>

  func.func private @LyHost_GetcStdin() -> i32
  func.func private @LyHost_FFlush(i64) -> i32

  // input(prompt): write the prompt without a newline, read one line from
  // stdin (the trailing newline is stripped), EOFError when the stream ends
  // before any character.
  func.func @LyBuiltin_Input(%prompt_header: memref<2xi64> {ly.ownership.object_header}, %prompt_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "input", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.str", ly.runtime.primitive = "builtin_input", ly.runtime.result_contract = "builtins.str"} {
    %zero64 = arith.constant 0 : i64
    %minus_one = arith.constant -1 : i32
    %newline = arith.constant 10 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %initial_cap = arith.constant 128 : index
    func.call @LyUnicode_Print(%prompt_header, %prompt_bytes) : (memref<2xi64>, memref<?xi8>) -> ()
    // fflush(NULL): flush the prompt before blocking on stdin.
    %flush_all = func.call @LyHost_FFlush(%zero64) : (i64) -> i32
    %buf0 = memref.alloc(%initial_cap) : memref<?xi8>
    cf.br ^loop(%buf0, %initial_cap, %c0 : memref<?xi8>, index, index)

  ^loop(%buf: memref<?xi8>, %cap: index, %len: index):
    %ch = func.call @LyHost_GetcStdin() : () -> i32
    %is_eof = arith.cmpi eq, %ch, %minus_one : i32
    cf.cond_br %is_eof, ^eof(%buf, %len : memref<?xi8>, index), ^got(%buf, %cap, %len, %ch : memref<?xi8>, index, index, i32)

  ^got(%gbuf: memref<?xi8>, %gcap: index, %glen: index, %gch: i32):
    %is_nl = arith.cmpi eq, %gch, %newline : i32
    cf.cond_br %is_nl, ^done(%gbuf, %glen : memref<?xi8>, index), ^store(%gbuf, %gcap, %glen, %gch : memref<?xi8>, index, index, i32)

  ^store(%sbuf: memref<?xi8>, %scap: index, %slen: index, %sch: i32):
    %full = arith.cmpi uge, %slen, %scap : index
    %grown:2 = scf.if %full -> (memref<?xi8>, index) {
      %two = arith.constant 2 : index
      %newcap = arith.muli %scap, %two : index
      %newbuf = memref.alloc(%newcap) : memref<?xi8>
      scf.for %i = %c0 to %slen step %c1 {
        %b = memref.load %sbuf[%i] : memref<?xi8>
        memref.store %b, %newbuf[%i] : memref<?xi8>
      }
      memref.dealloc %sbuf : memref<?xi8>
      scf.yield %newbuf, %newcap : memref<?xi8>, index
    } else {
      scf.yield %sbuf, %scap : memref<?xi8>, index
    }
    %byte = arith.trunci %sch : i32 to i8
    memref.store %byte, %grown#0[%slen] : memref<?xi8>
    %next_len = arith.addi %slen, %c1 : index
    cf.br ^loop(%grown#0, %grown#1, %next_len : memref<?xi8>, index, index)

  ^eof(%ebuf: memref<?xi8>, %elen: index): // EOF before any character: EOFError
    %saw_any = arith.cmpi ne, %elen, %c0 : index
    cf.cond_br %saw_any, ^done(%ebuf, %elen : memref<?xi8>, index), ^raise(%ebuf : memref<?xi8>)

  ^raise(%rbuf: memref<?xi8>):
    memref.dealloc %rbuf : memref<?xi8>
    %eof_class = arith.constant {ly.class_of = "builtins.EOFError"} 106 : i64
    %msg_static = memref.get_global @__ly_input_eof_msg : memref<23xi8>
    %msg = memref.cast %msg_static : memref<23xi8> to memref<?xi8>
    %msg_len = arith.constant 23 : i64
    %exception:3 = func.call @LyBaseException_New(%eof_class) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    %message_header, %message_bytes = func.call @__ly_unicode_from_valid_utf8(%msg, %c0, %msg_len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %initialized:3 = func.call @LyBaseException_Init(%exception#0, %exception#1, %exception#2, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.call @LyEH_ThrowException(%initialized#0, %initialized#1, %initialized#2) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    // The throw does not return; satisfy the CFG with an empty line.
    %dead = memref.alloc(%c0) : memref<?xi8>
    cf.br ^done(%dead, %c0 : memref<?xi8>, index)

  ^done(%dbuf: memref<?xi8>, %dlen: index):
    %dlen64 = arith.index_cast %dlen : index to i64
    %result_header, %result_bytes = func.call @LyUnicode_FromBytes(%dbuf, %c0, %dlen64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    memref.dealloc %dbuf : memref<?xi8>
    func.return %result_header, %result_bytes : memref<2xi64>, memref<?xi8>
  }

  // "chr() arg not in range(0x110000)"
  memref.global "private" constant @__ly_chr_msg : memref<32xi8> = dense<[99, 104, 114, 40, 41, 32, 97, 114, 103, 32, 110, 111, 116, 32, 105, 110, 32, 114, 97, 110, 103, 101, 40, 48, 120, 49, 49, 48, 48, 48, 48, 41]>

  func.func @LyBuiltin_Chr(%header: memref<2xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "chr", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.int", ly.runtime.primitive = "builtin_chr", ly.runtime.result_contract = "builtins.str"} {
    %meta_raw, %digits_raw = func.call @__ly_long_parts(%header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %meta, %digits = func.call @__ly_long_operand_view(%meta_raw, %digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0 : i64
    %max = arith.constant 1114112 : i64
    %fits = func.call @__ly_long_view_fits_i64(%meta, %digits) : (memref<2xi64>, memref<?xi32>) -> i1
    %cp_raw = func.call @__ly_long_view_as_i64(%meta, %digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %negative = arith.cmpi slt, %cp_raw, %zero : i64
    %too_big = arith.cmpi sge, %cp_raw, %max : i64
    %true = arith.constant true
    %not_fits = arith.xori %fits, %true : i1
    %bad0 = arith.ori %negative, %too_big : i1
    %bad = arith.ori %bad0, %not_fits : i1
    scf.if %bad {
      %value_error = arith.constant {ly.class_of = "builtins.ValueError"} 53 : i64
      %msg_static = memref.get_global @__ly_chr_msg : memref<32xi8>
      %msg = memref.cast %msg_static : memref<32xi8> to memref<?xi8>
      %len = arith.constant 32 : i64
      func.call @__ly_raise_static_message(%value_error, %msg, %len) : (i64, memref<?xi8>, i64) -> ()
    }
    %h, %b = func.call @__ly_unicode_single(%cp_raw) : (i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %h, %b : memref<2xi64>, memref<?xi8>
  }

  func.func @LyBuiltin_Hex(%header: memref<2xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "hex", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.int", ly.runtime.primitive = "builtin_hex", ly.runtime.result_contract = "builtins.str"} {
    %meta_raw, %digits_raw = func.call @__ly_long_parts(%header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %meta, %digits = func.call @__ly_long_operand_view(%meta_raw, %digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %bits = arith.constant 4 : i64
    %letter = arith.constant 120 : i8
    %h, %b = func.call @__ly_long_format_pow2(%meta, %digits, %bits, %letter) : (memref<2xi64>, memref<?xi32>, i64, i8) -> (memref<2xi64>, memref<?xi8>)
    func.return %h, %b : memref<2xi64>, memref<?xi8>
  }

  func.func @LyBuiltin_Oct(%header: memref<2xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "oct", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.int", ly.runtime.primitive = "builtin_oct", ly.runtime.result_contract = "builtins.str"} {
    %meta_raw, %digits_raw = func.call @__ly_long_parts(%header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %meta, %digits = func.call @__ly_long_operand_view(%meta_raw, %digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %bits = arith.constant 3 : i64
    %letter = arith.constant 111 : i8
    %h, %b = func.call @__ly_long_format_pow2(%meta, %digits, %bits, %letter) : (memref<2xi64>, memref<?xi32>, i64, i8) -> (memref<2xi64>, memref<?xi8>)
    func.return %h, %b : memref<2xi64>, memref<?xi8>
  }

  func.func @LyBuiltin_Bin(%header: memref<2xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "bin", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.int", ly.runtime.primitive = "builtin_bin", ly.runtime.result_contract = "builtins.str"} {
    %meta_raw, %digits_raw = func.call @__ly_long_parts(%header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %meta, %digits = func.call @__ly_long_operand_view(%meta_raw, %digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %bits = arith.constant 1 : i64
    %letter = arith.constant 98 : i8
    %h, %b = func.call @__ly_long_format_pow2(%meta, %digits, %bits, %letter) : (memref<2xi64>, memref<?xi32>, i64, i8) -> (memref<2xi64>, memref<?xi8>)
    func.return %h, %b : memref<2xi64>, memref<?xi8>
  }
}
