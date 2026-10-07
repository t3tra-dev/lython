// `int` -- CPython's Objects/longobject.c: magnitudes in 30-bit digits as
// PyLong keeps them, the small-int cache, and int(str) (PyLong_FromUnicodeObject).
//
// Signature source (1:1 correspondence target):
//   https://github.com/python/typeshed/blob/main/stdlib/builtins.pyi

module attributes {
  ly.typing.manifest,
  ly.runtime.contracts = ["builtins.int"]
} {
  // ===== declared here, defined in another runtime file or built by the lowering =====
  func.func private @__ly_unicode_from_valid_utf8(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]}
  func.func private @LyBaseException_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.BaseException", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"}
  func.func private @LyBaseException_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 5 : i64, ly.runtime.contract = "builtins.BaseException", ly.runtime.initializer = "__new__"}
  func.func private @LyEH_ThrowException(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "raise"}
  func.func private @LyFloat_AsF64(%header: memref<3xi64> {ly.ownership.object_header}) -> f64 attributes {ly.runtime.contract = "builtins.float", ly.runtime.method = "__float__", ly.runtime.primitive = "unbox.f64"}
  func.func private @LyFloat_DecRef(%header: memref<3xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.float", ly.runtime.deallocator}
  func.func private @LyFloat_FromF64(%value: f64 {ly.runtime.default_f64 = 0.0 : f64}) -> memref<3xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 2 : i64, ly.runtime.contract = "builtins.float", ly.runtime.initializer = "__new__"}
  func.func private @LyFloat_Int(%header: memref<3xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.float", ly.runtime.method = "__int__", ly.runtime.result_contract = "builtins.int"}
  func.func private @LyObject_ReleaseStorageToZero(%storage: memref<?xi64>) -> i1 attributes {ly.runtime.contract = "builtins.object", ly.runtime.primitive = "release_to_zero"}
  func.func private @LyUnicode_Concat(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_bytes: memref<?xi8>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__add__"}
  func.func private @LyUnicode_DecRef(%header: memref<2xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.str", ly.runtime.deallocator}
  func.func private @LyUnicode_FromBytes(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 4 : i64, ly.runtime.contract = "builtins.str", ly.runtime.initializer = "__new__"}
  func.func private @LyUnicode_FromI64(%value: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]}
  func.func private @LyUnicode_Repr(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"}
  func.func private @__ly_addresses_are_word_wide() -> i1
  func.func private @__ly_alloc_count(%count: i64, %element_bytes: i64, %extra: i64) -> index
  func.func private @__ly_check_alloc_count(%count: i64, %element_bytes: i64, %extra: i64)
  func.func private @__ly_float_format_core(%value: f64, %spec: memref<?xi64>, %tname: memref<?xi8>, %tname_len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]}
  func.func private @__ly_fmt_int_base2k(%meta: memref<2xi64>, %digits: memref<?xi32>, %shift: i64, %upper: i1, %out: memref<?xi32>) -> i64
  memref.global "private" constant @__ly_fmt_msg_name_int : memref<3xi8>
  func.func private @__ly_fmt_parse_spec(%spec_h: memref<2xi64>, %spec_b: memref<?xi8>, %out: memref<?xi64>) -> i1
  func.func private @__ly_fmt_raise_alt_c()
  func.func private @__ly_fmt_raise_c_range()
  func.func private @__ly_fmt_raise_cannot_group(%gcp: i64, %wcp: i64)
  func.func private @__ly_fmt_raise_int_precision()
  func.func private @__ly_fmt_raise_invalid_spec(%spec_h: memref<2xi64>, %spec_b: memref<?xi8>, %name: memref<?xi8>, %name_len: i64)
  func.func private @__ly_fmt_raise_sign_c()
  func.func private @__ly_fmt_raise_unknown_code(%code: i64, %name: memref<?xi8>, %name_len: i64)
  func.func private @__ly_fmt_raise_z_int()
  func.func private @__ly_fmt_render_number(%sign_cp: i64, %pre0: i64, %pre1: i64, %body: memref<?xi32>, %body_len: i64, %int_len: i64, %group_cp: i64, %group_size: i64, %fill_cp: i64, %align_cp: i64, %width_in: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]}
  func.func private @__ly_global_view_i32(%pointer: i64, %size: i64) -> memref<?xi32>
  func.func private @__ly_global_view_i64(%pointer: i64, %size: i64) -> memref<?xi64>
  func.func private @__ly_handle_retain_raw(%entity: i64)
  func.func private @__ly_raise_static_message(%class_id: i64, %message: memref<?xi8>, %length: i64)
  func.func private @__ly_slot_word_from_view_address(%address: i64) -> i64
  func.func private @__ly_slot_word_is_immediate(%word: i64) -> i1

  py.class @int attributes {
    base_names = ["object"], ly.typing.final,
    method_names = ["__new__", "__add__", "__sub__", "__mul__", "__floordiv__",
                    "__truediv__", "__mod__", "__and__", "__or__", "__xor__",
                    "__lshift__", "__rshift__", "__neg__", "__pos__",
                    "__invert__", "__round__", "__int__", "__float__",
                    "__bool__", "__index__", "__hash__", "__lt__", "__le__",
                    "__gt__", "__ge__", "__repr__", "__str__", "__eq__", "__ne__",
                    "__pow__", "__abs__", "__format__",
                    "__lt__", "__le__", "__gt__", "__ge__", "__eq__", "__ne__",
                    "__round__", "bit_length"],
    method_contracts = [
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.int">>, !py.union<!py.contract<"typing.SupportsInt">, !py.contract<"typing.SupportsIndex">, !py.contract<"builtins.str">, !py.contract<"builtins.bytes">, !py.contract<"builtins.bytearray">>] -> [!py.self]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.float">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">] -> [!py.contract<"builtins.float">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.float">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.float">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.float">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.float">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.float">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">, !py.contract<"builtins.float">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>
    ],
    method_kinds = ["classmethod", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance",
                    "instance", "instance"]
  } {}

  func.func @LyLong_DecRef(%header: memref<2xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.int", ly.runtime.deallocator} {
    %storage = memref.cast %header : memref<2xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    // int is one allocation; the header view carries it. meta/digits are
    // interior views of the same block and must not be freed.
    memref.dealloc %header : memref<2xi64>
    cf.br ^done

  ^done:
    func.return
  }

  // ===== impls: long =====
  func.func private @LyLong_Shape() -> memref<2xi64> attributes {ly.runtime.contract = "builtins.int", ly.runtime.shape}

  // Views of int operands used by the arithmetic entry points.
  func.func private @__ly_long_operand_view(%meta: memref<2xi64>, %digits: memref<?xi32>) -> (memref<2xi64>, memref<?xi32>) {
    func.return %meta, %digits : memref<2xi64>, memref<?xi32>
  }

  memref.global "private" constant @__ly_long_msg_division_by_zero : memref<16xi8> = dense<[100, 105, 118, 105, 115, 105, 111, 110, 32, 98, 121, 32, 122, 101, 114, 111]>
  memref.global "private" constant @__ly_long_msg_integer_division_or_modulo_by_zero : memref<34xi8> = dense<[105, 110, 116, 101, 103, 101, 114, 32, 100, 105, 118, 105, 115, 105, 111, 110, 32, 111, 114, 32, 109, 111, 100, 117, 108, 111, 32, 98, 121, 32, 122, 101, 114, 111]>
  memref.global "private" constant @__ly_long_msg_integer_modulo_by_zero : memref<22xi8> = dense<[105, 110, 116, 101, 103, 101, 114, 32, 109, 111, 100, 117, 108, 111, 32, 98, 121, 32, 122, 101, 114, 111]>
  memref.global "private" constant @__ly_long_msg_negative_shift_count : memref<20xi8> = dense<[110, 101, 103, 97, 116, 105, 118, 101, 32, 115, 104, 105, 102, 116, 32, 99, 111, 117, 110, 116]>
  // "int too large to convert to a native 64-bit integer"
  memref.global "private" constant @__ly_long_msg_int_too_large : memref<51xi8> = dense<[105, 110, 116, 32, 116, 111, 111, 32, 108, 97, 114, 103, 101, 32, 116, 111, 32, 99, 111, 110, 118, 101, 114, 116, 32, 116, 111, 32, 97, 32, 110, 97, 116, 105, 118, 101, 32, 54, 52, 45, 98, 105, 116, 32, 105, 110, 116, 101, 103, 101, 114]>
  // "invalid literal for int() with base 10" (CPython appends the repr of the
  // input; message concatenation needs the str-track's formatting work)
  memref.global "private" constant @__ly_long_msg_invalid_int_literal_prefix : memref<40xi8> = dense<[105, 110, 118, 97, 108, 105, 100, 32, 108, 105, 116, 101, 114, 97, 108, 32, 102, 111, 114, 32, 105, 110, 116, 40, 41, 32, 119, 105, 116, 104, 32, 98, 97, 115, 101, 32, 49, 48, 58, 32]>
  // "int ** negative int is rejected: the static result type is int; use float(base) ** exponent"
  memref.global "private" constant @__ly_long_msg_pow_negative_exponent : memref<91xi8> = dense<[105, 110, 116, 32, 42, 42, 32, 110, 101, 103, 97, 116, 105, 118, 101, 32, 105, 110, 116, 32, 105, 115, 32, 114, 101, 106, 101, 99, 116, 101, 100, 58, 32, 116, 104, 101, 32, 115, 116, 97, 116, 105, 99, 32, 114, 101, 115, 117, 108, 116, 32, 116, 121, 112, 101, 32, 105, 115, 32, 105, 110, 116, 59, 32, 117, 115, 101, 32, 102, 108, 111, 97, 116, 40, 98, 97, 115, 101, 41, 32, 42, 42, 32, 101, 120, 112, 111, 110, 101, 110, 116]>
  // "too many digits in integer"
  memref.global "private" constant @__ly_long_msg_too_many_digits : memref<26xi8> = dense<[116, 111, 111, 32, 109, 97, 110, 121, 32, 100, 105, 103, 105, 116, 115, 32, 105, 110, 32, 105, 110, 116, 101, 103, 101, 114]>
  // "int too large to convert to float"
  memref.global "private" constant @__ly_long_msg_int_too_large_float : memref<33xi8> = dense<[105, 110, 116, 32, 116, 111, 111, 32, 108, 97, 114, 103, 101, 32, 116, 111, 32, 99, 111, 110, 118, 101, 114, 116, 32, 116, 111, 32, 102, 108, 111, 97, 116]>
  // "integer division result too large for a float"
  memref.global "private" constant @__ly_long_msg_div_result_too_large : memref<45xi8> = dense<[105, 110, 116, 101, 103, 101, 114, 32, 100, 105, 118, 105, 115, 105, 111, 110, 32, 114, 101, 115, 117, 108, 116, 32, 116, 111, 111, 32, 108, 97, 114, 103, 101, 32, 102, 111, 114, 32, 97, 32, 102, 108, 111, 97, 116]>
  // "cannot convert float NaN to integer"
  memref.global "private" constant @__ly_long_msg_float_nan : memref<35xi8> = dense<[99, 97, 110, 110, 111, 116, 32, 99, 111, 110, 118, 101, 114, 116, 32, 102, 108, 111, 97, 116, 32, 78, 97, 78, 32, 116, 111, 32, 105, 110, 116, 101, 103, 101, 114]>
  // "cannot convert float infinity to integer"
  memref.global "private" constant @__ly_long_msg_float_infinity : memref<40xi8> = dense<[99, 97, 110, 110, 111, 116, 32, 99, 111, 110, 118, 101, 114, 116, 32, 102, 108, 111, 97, 116, 32, 105, 110, 102, 105, 110, 105, 116, 121, 32, 116, 111, 32, 105, 110, 116, 101, 103, 101, 114]>
  // "zero to a negative power"
  memref.global "private" constant @__ly_long_msg_zero_negative_power : memref<24xi8> = dense<[122, 101, 114, 111, 32, 116, 111, 32, 97, 32, 110, 101, 103, 97, 116, 105, 118, 101, 32, 112, 111, 119, 101, 114]>
  // "negative number cannot be raised to a fractional power".
  //
  // ⛔ CPython returns a COMPLEX here, and complex is implemented -- what
  // blocks it is the static result type, not the type's absence. `x ** y` over
  // two floats is a float for every pair but this one, and a return type of
  // `float | complex` is a union the py ABI cannot carry out of an operator.
  // typeshed says the same thing by giving `float.__pow__` a result of `Any`.
  // So the operator raises where the answer would leave the type: loud, and at
  // the value that caused it.
  memref.global "private" constant @__ly_long_msg_fractional_power_negative : memref<54xi8> = dense<[110, 101, 103, 97, 116, 105, 118, 101, 32, 110, 117, 109, 98, 101, 114, 32, 99, 97, 110, 110, 111, 116, 32, 98, 101, 32, 114, 97, 105, 115, 101, 100, 32, 116, 111, 32, 97, 32, 102, 114, 97, 99, 116, 105, 111, 110, 97, 108, 32, 112, 111, 119, 101, 114]>
  memref.global "private" constant @__ly_long_zero_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" constant @__ly_long_zero_meta : memref<2xi64> = dense<[0, 0]>
  memref.global "private" constant @__ly_long_zero_digits : memref<1xi32> = dense<[0]>
  memref.global "private" constant @__ly_long_one_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" constant @__ly_long_one_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" constant @__ly_long_one_digits : memref<1xi32> = dense<[1]>
  memref.global "private" constant @__ly_long_two_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" constant @__ly_long_two_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" constant @__ly_long_two_digits : memref<1xi32> = dense<[2]>

  // ⭐ CPython's `_PyLong_SMALL_INTS` (Objects/longobject.c). Every int in
  // [-_PY_NSMALLNEGINTS, _PY_NSMALLPOSINTS) -- -5 through 256 in 3.14 -- is a
  // shared immortal object there, and `PyLong_FromLong` returns it without
  // allocating (`IS_SMALL_INT` / `get_small_int`). This runtime cached three
  // of them, so `i + 1` in a loop allocated and freed on every step.
  //
  // ⛔ ONE HEADER PER VALUE IS NOT NEEDED, because the emitter refuses `is` on
  // int (identity of a value type is not observable in this language), so the
  // table is a plain byte block laid out exactly like a heap int:
  // [0,16) header, [16,32) meta, [32,36) digits, padded to 48 for alignment.
  //
  // ⛔ Filled at first use rather than written out as a literal: the digits are
  // i32 and the block is bytes, so a static initializer would have to commit to
  // an endianness, and this compiler cross-compiles. A store through the typed
  // view is correct on either.
  memref.global "private" @__ly_long_small_ints : memref<12576xi8> = dense<0> {alignment = 16 : i64}
  memref.global "private" @__ly_long_small_ready : memref<1xi64> = dense<0>

  func.func private @__ly_long_small_slot(%value: i64) -> memref<2xi64> {
    %table_static = memref.get_global @__ly_long_small_ints : memref<12576xi8>
    %table = memref.cast %table_static : memref<12576xi8> to memref<?xi8>
    %neg_span = arith.constant 5 : i64
    %slot_bytes = arith.constant 48 : i64
    %slot_index = arith.addi %value, %neg_span : i64
    %byte_offset_i64 = arith.muli %slot_index, %slot_bytes : i64
    %byte_offset = arith.index_cast %byte_offset_i64 : i64 to index
    %meta_delta = arith.constant 16 : index
    %digits_delta = arith.constant 32 : index
    %one_digit = arith.constant 1 : index
    %meta_offset = arith.addi %byte_offset, %meta_delta : index
    %digits_offset = arith.addi %byte_offset, %digits_delta : index
    %header = memref.view %table[%byte_offset][] : memref<?xi8> to memref<2xi64>
    %meta = memref.view %table[%meta_offset][] : memref<?xi8> to memref<2xi64>
    %digits = memref.view %table[%digits_offset][%one_digit] : memref<?xi8> to memref<?xi32>
    func.return %header : memref<2xi64>
  }

  func.func private @__ly_long_small_ensure() {
    %ready = memref.get_global @__ly_long_small_ready : memref<1xi64>
    %flag_slot = arith.constant 0 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %flag = memref.load %ready[%flag_slot] : memref<1xi64>
    %needs_init = arith.cmpi eq, %flag, %zero : i64
    scf.if %needs_init {
      %first = arith.constant 0 : index
      %past_last = arith.constant 262 : index
      %step = arith.constant 1 : index
      %lowest = arith.constant -5 : i64
      %immortal = arith.constant 9223372036854775807 : i64
      %layout_int = arith.constant 1 : i64
      %minus_one = arith.constant -1 : i64
      %refcount_slot = arith.constant 0 : index
      %layout_slot = arith.constant 1 : index
      %sign_slot = arith.constant 0 : index
      %digit_count_slot = arith.constant 1 : index
      %digit0_slot = arith.constant 0 : index
      scf.for %slot = %first to %past_last step %step {
        %offset = arith.index_cast %slot : index to i64
        %value = arith.addi %offset, %lowest : i64
        %header = func.call @__ly_long_small_slot(%value) : (i64) -> memref<2xi64>
        %meta, %digits = func.call @__ly_long_parts(%header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
        memref.store %immortal, %header[%refcount_slot] : memref<2xi64>
        memref.store %layout_int, %header[%layout_slot] : memref<2xi64>
        %negative = arith.cmpi slt, %value, %zero : i64
        %is_zero = arith.cmpi eq, %value, %zero : i64
        %signed_sign = arith.select %negative, %minus_one, %one : i1, i64
        %sign = arith.select %is_zero, %zero, %signed_sign : i1, i64
        %digit_count = arith.select %is_zero, %zero, %one : i1, i64
        memref.store %sign, %meta[%sign_slot] : memref<2xi64>
        memref.store %digit_count, %meta[%digit_count_slot] : memref<2xi64>
        %negated = arith.subi %zero, %value : i64
        %magnitude = arith.select %negative, %negated, %value : i1, i64
        %digit = arith.trunci %magnitude : i64 to i32
        memref.store %digit, %digits[%digit0_slot] : memref<?xi32>
      }
      memref.store %one, %ready[%flag_slot] : memref<1xi64>
    }
    func.return
  }

  // CPython 3.14 unified the /, //, and % zero-divisor message to
  // "division by zero" (gh-87999), so the historic per-operator texts are not
  // used -- and once they were gone the three raisers were the same function
  // under three names.
  func.func private @__ly_long_raise_division_by_zero() {
    %class_id = arith.constant 61 : i64
    %length = arith.constant 16 : i64
    %message_static = memref.get_global @__ly_long_msg_division_by_zero : memref<16xi8>
    %message = memref.cast %message_static : memref<16xi8> to memref<?xi8>
    func.call @__ly_raise_static_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @__ly_long_raise_negative_shift() {
    %class_id = arith.constant 53 : i64
    %length = arith.constant 20 : i64
    %message_static = memref.get_global @__ly_long_msg_negative_shift_count : memref<20xi8>
    %message = memref.cast %message_static : memref<20xi8> to memref<?xi8>
    func.call @__ly_raise_static_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  // The int handle is the header alone; meta and digits are recovered from it.
  // Every int object is one allocation with meta at +16 and digits at +32 --
  // `__ly_long_alloc_raw`, `LyLong_FromI64` and `__ly_long_small_slot` are the
  // only three sites that build one, and all three lay it out that way.
  //
  // Why not keep passing the three views: they are fifteen descriptor words
  // carrying five words of information, and every `LyLong_*` call paid to
  // shuffle them. The digits span is clamped to one because a zero has no
  // digits and `__ly_long_view_as_i64` still reads digit 0 -- the allocation
  // always has that slot.
  //
  // No function that receives an int PARAMETER may write its digits: the span
  // here is `ndigits`, not the capacity, so a write past the normalized end
  // would fall outside the view. Nothing does today (checked across all ten
  // manifests); an in-place mutator would have to take the block instead.
  //
  // Why NOT also narrow the `scf` regions inside the bignum bodies, which still
  // carry the three views as loop-carried and yielded values: the win is in the
  // CALL ABI, which every int operation pays, while those regions are the
  // multi-limb paths a small int never enters. Narrowing them is mechanical and
  // still open; leaving them is what kept this change to signatures, returns,
  // and call sites.
  func.func private @__ly_long_parts(%header: memref<2xi64>) -> (memref<2xi64>, memref<?xi32>) {
    %base_index = memref.extract_aligned_pointer_as_index %header : memref<2xi64> -> index
    %base = arith.index_cast %base_index : index to i64
    %meta_delta = arith.constant 16 : i64
    %digits_delta = arith.constant 32 : i64
    %meta_word = arith.addi %base, %meta_delta : i64
    %digits_word = arith.addi %base, %digits_delta : i64
    %two = arith.constant 2 : i64
    %meta_dyn = func.call @__ly_global_view_i64(%meta_word, %two) : (i64, i64) -> memref<?xi64>
    %meta = memref.cast %meta_dyn : memref<?xi64> to memref<2xi64>
    %digit_count_slot = arith.constant 1 : index
    %ndigits = memref.load %meta[%digit_count_slot] : memref<2xi64>
    %one = arith.constant 1 : i64
    %empty = arith.cmpi slt, %ndigits, %one : i64
    %span = arith.select %empty, %one, %ndigits : i1, i64
    %digits = func.call @__ly_global_view_i32(%digits_word, %span) : (i64, i64) -> memref<?xi32>
    func.return %meta, %digits : memref<2xi64>, memref<?xi32>
  }

  // Returns the header alone; the caller takes its meta and digits through
  // `__ly_long_parts`. The digit-count slot holds the CAPACITY on return, not
  // the final digit count, so that view spans the whole buffer -- which is why
  // a caller must take it BEFORE narrowing the count to the normalized length.
  // Every caller does, on the line after the allocation.
  func.func private @__ly_long_alloc_raw(%sign: i64, %capacity: i64) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0]} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %layout_int = arith.constant 1 : i64
    %needs_one = arith.cmpi sle, %capacity, %zero : i64
    %alloc_count_i64 = arith.select %needs_one, %one, %capacity : i1, i64
    // One entity, one allocation: [0,16) header, [16,32) meta, [32,..) digits.
    // The header view carries the allocation; meta/digits are interior views
    // and must never be freed separately.
    %four_bytes = arith.constant 4 : i64
    %block_prefix = arith.constant 32 : i64
    %alloc_count = func.call @__ly_alloc_count(%alloc_count_i64, %four_bytes, %block_prefix) : (i64, i64, i64) -> index
    %four_index = arith.constant 4 : index
    %prefix_index = arith.constant 32 : index
    %digit_bytes = arith.muli %alloc_count, %four_index : index
    %block_bytes = arith.addi %digit_bytes, %prefix_index : index
    %block = memref.alloc(%block_bytes) {alignment = 16 : i64} : memref<?xi8>
    %header_offset = arith.constant 0 : index
    %meta_offset = arith.constant 16 : index
    %digits_offset = arith.constant 32 : index
    %header = memref.view %block[%header_offset][] {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<?xi8> to memref<2xi64>
    %meta = memref.view %block[%meta_offset][] : memref<?xi8> to memref<2xi64>
    %digits = memref.view %block[%digits_offset][%alloc_count] : memref<?xi8> to memref<?xi32>
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %sign_slot = arith.constant 0 : index
    %digit_count_slot = arith.constant 1 : index
    memref.store %one, %header[%refcount_slot] : memref<2xi64>
    memref.store %layout_int, %header[%layout_slot] : memref<2xi64>
    memref.store %sign, %meta[%sign_slot] : memref<2xi64>
    memref.store %capacity, %meta[%digit_count_slot] : memref<2xi64>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero_i32 = arith.constant 0 : i32
    scf.for %iv = %c0 to %alloc_count step %c1 {
      memref.store %zero_i32, %digits[%iv] : memref<?xi32>
    }
    func.return %header : memref<2xi64>
  }

  func.func private @__ly_long_normalize(%meta: memref<2xi64>, %digits: memref<?xi32>, %capacity: i64) {
    %zero = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %capacity_index = arith.index_cast %capacity : i64 to index
    %last = scf.for %iv = %c0 to %capacity_index step %c1 iter_args(%last_iter = %zero) -> (i64) {
      %digit_i32 = memref.load %digits[%iv] : memref<?xi32>
      %digit = arith.extui %digit_i32 : i32 to i64
      %nonzero = arith.cmpi ne, %digit, %zero : i64
      %next_index = arith.addi %iv, %c1 : index
      %next_count = arith.index_cast %next_index : index to i64
      %next = arith.select %nonzero, %next_count, %last_iter : i1, i64
      scf.yield %next : i64
    }
    %sign_slot = arith.constant 0 : index
    %digit_count_slot = arith.constant 1 : index
    %raw_sign = memref.load %meta[%sign_slot] : memref<2xi64>
    %is_zero = arith.cmpi eq, %last, %zero : i64
    %sign = arith.select %is_zero, %zero, %raw_sign : i1, i64
    memref.store %sign, %meta[%sign_slot] : memref<2xi64>
    memref.store %last, %meta[%digit_count_slot] : memref<2xi64>
    func.return
  }

  func.func private @__ly_long_copy_with_sign(%sign: i64, %meta_in: memref<2xi64>, %digits_in: memref<?xi32>) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0]} {
    %count_slot = arith.constant 1 : index
    %count = memref.load %meta_in[%count_slot] : memref<2xi64>
    %header = func.call @__ly_long_alloc_raw(%sign, %count) : (i64, i64) -> memref<2xi64>
    %meta, %digits = func.call @__ly_long_parts(%header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %count_index = arith.index_cast %count : i64 to index
    scf.for %iv = %c0 to %count_index step %c1 {
      %digit = memref.load %digits_in[%iv] : memref<?xi32>
      memref.store %digit, %digits[%iv] : memref<?xi32>
    }
    func.call @__ly_long_normalize(%meta, %digits, %count) : (memref<2xi64>, memref<?xi32>, i64) -> ()
    func.return %header : memref<2xi64>
  }

  func.func private @__ly_long_copy(%meta_in: memref<2xi64>, %digits_in: memref<?xi32>) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0]} {
    %sign_slot = arith.constant 0 : index
    %sign = memref.load %meta_in[%sign_slot] : memref<2xi64>
    %header = func.call @__ly_long_copy_with_sign(%sign, %meta_in, %digits_in) : (i64, memref<2xi64>, memref<?xi32>) -> memref<2xi64>
    func.return %header : memref<2xi64>
  }

  func.func private @__ly_long_abs_compare(%lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> i64 {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %neg_one = arith.constant -1 : i64
    %count_slot = arith.constant 1 : index
    %lhs_count = memref.load %lhs_meta[%count_slot] : memref<2xi64>
    %rhs_count = memref.load %rhs_meta[%count_slot] : memref<2xi64>
    %lhs_longer = arith.cmpi sgt, %lhs_count, %rhs_count : i64
    %rhs_longer = arith.cmpi slt, %lhs_count, %rhs_count : i64
    %size_cmp = arith.select %lhs_longer, %one, %zero : i1, i64
    %size_cmp2 = arith.select %rhs_longer, %neg_one, %size_cmp : i1, i64
    %same_size = arith.cmpi eq, %size_cmp2, %zero : i64
    %result = scf.if %same_size -> (i64) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %count_index = arith.index_cast %lhs_count : i64 to index
      %cmp = scf.for %iv = %c0 to %count_index step %c1 iter_args(%state = %zero) -> (i64) {
        %still_equal = arith.cmpi eq, %state, %zero : i64
        %iv_next = arith.addi %iv, %c1 : index
        %rev = arith.subi %count_index, %iv_next : index
        %lhs_digit_i32 = memref.load %lhs_digits[%rev] : memref<?xi32>
        %rhs_digit_i32 = memref.load %rhs_digits[%rev] : memref<?xi32>
        %lhs_digit = arith.extui %lhs_digit_i32 : i32 to i64
        %rhs_digit = arith.extui %rhs_digit_i32 : i32 to i64
        %gt = arith.cmpi ugt, %lhs_digit, %rhs_digit : i64
        %lt = arith.cmpi ult, %lhs_digit, %rhs_digit : i64
        %digit_cmp = arith.select %gt, %one, %zero : i1, i64
        %digit_cmp2 = arith.select %lt, %neg_one, %digit_cmp : i1, i64
        %next = arith.select %still_equal, %digit_cmp2, %state : i1, i64
        scf.yield %next : i64
      }
      scf.yield %cmp : i64
    } else {
      scf.yield %size_cmp2 : i64
    }
    func.return %result : i64
  }

  func.func private @__ly_long_add_abs(%sign: i64, %lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0]} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %mask = arith.constant 1073741823 : i64
    %shift30 = arith.constant 30 : i64
    %count_slot = arith.constant 1 : index
    %lhs_count = memref.load %lhs_meta[%count_slot] : memref<2xi64>
    %rhs_count = memref.load %rhs_meta[%count_slot] : memref<2xi64>
    %lhs_ge = arith.cmpi sge, %lhs_count, %rhs_count : i64
    %max_count = arith.select %lhs_ge, %lhs_count, %rhs_count : i1, i64
    %capacity = arith.addi %max_count, %one : i64
    %header = func.call @__ly_long_alloc_raw(%sign, %capacity) : (i64, i64) -> memref<2xi64>
    %meta, %digits = func.call @__ly_long_parts(%header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %max_index = arith.index_cast %max_count : i64 to index
    %carry = scf.for %iv = %c0 to %max_index step %c1 iter_args(%carry_iter = %zero) -> (i64) {
      %iv_i64 = arith.index_cast %iv : index to i64
      %has_lhs = arith.cmpi slt, %iv_i64, %lhs_count : i64
      %lhs_digit = scf.if %has_lhs -> (i64) {
        %digit_i32 = memref.load %lhs_digits[%iv] : memref<?xi32>
        %digit = arith.extui %digit_i32 : i32 to i64
        scf.yield %digit : i64
      } else {
        scf.yield %zero : i64
      }
      %has_rhs = arith.cmpi slt, %iv_i64, %rhs_count : i64
      %rhs_digit = scf.if %has_rhs -> (i64) {
        %digit_i32 = memref.load %rhs_digits[%iv] : memref<?xi32>
        %digit = arith.extui %digit_i32 : i32 to i64
        scf.yield %digit : i64
      } else {
        scf.yield %zero : i64
      }
      %partial = arith.addi %lhs_digit, %rhs_digit : i64
      %sum = arith.addi %partial, %carry_iter : i64
      %out_i64 = arith.andi %sum, %mask : i64
      %out = arith.trunci %out_i64 : i64 to i32
      memref.store %out, %digits[%iv] : memref<?xi32>
      %next_carry = arith.shrui %sum, %shift30 : i64
      scf.yield %next_carry : i64
    }
    %carry_slot = arith.index_cast %max_count : i64 to index
    %carry_i32 = arith.trunci %carry : i64 to i32
    memref.store %carry_i32, %digits[%carry_slot] : memref<?xi32>
    func.call @__ly_long_normalize(%meta, %digits, %capacity) : (memref<2xi64>, memref<?xi32>, i64) -> ()
    func.return %header : memref<2xi64>
  }

  func.func private @__ly_long_sub_abs(%sign: i64, %lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0]} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %base = arith.constant 1073741824 : i64
    %count_slot = arith.constant 1 : index
    %lhs_count = memref.load %lhs_meta[%count_slot] : memref<2xi64>
    %rhs_count = memref.load %rhs_meta[%count_slot] : memref<2xi64>
    %header = func.call @__ly_long_alloc_raw(%sign, %lhs_count) : (i64, i64) -> memref<2xi64>
    %meta, %digits = func.call @__ly_long_parts(%header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %lhs_index = arith.index_cast %lhs_count : i64 to index
    %borrow = scf.for %iv = %c0 to %lhs_index step %c1 iter_args(%borrow_iter = %zero) -> (i64) {
      %iv_i64 = arith.index_cast %iv : index to i64
      %lhs_digit_i32 = memref.load %lhs_digits[%iv] : memref<?xi32>
      %lhs_digit = arith.extui %lhs_digit_i32 : i32 to i64
      %has_rhs = arith.cmpi slt, %iv_i64, %rhs_count : i64
      %rhs_digit = scf.if %has_rhs -> (i64) {
        %digit_i32 = memref.load %rhs_digits[%iv] : memref<?xi32>
        %digit = arith.extui %digit_i32 : i32 to i64
        scf.yield %digit : i64
      } else {
        scf.yield %zero : i64
      }
      %rhs_with_borrow = arith.addi %rhs_digit, %borrow_iter : i64
      %needs_borrow = arith.cmpi ult, %lhs_digit, %rhs_with_borrow : i64
      %raw_diff = arith.subi %lhs_digit, %rhs_with_borrow : i64
      %borrowed_diff = arith.addi %raw_diff, %base : i64
      %diff = arith.select %needs_borrow, %borrowed_diff, %raw_diff : i1, i64
      %out = arith.trunci %diff : i64 to i32
      memref.store %out, %digits[%iv] : memref<?xi32>
      %next_borrow = arith.select %needs_borrow, %one, %zero : i1, i64
      scf.yield %next_borrow : i64
    }
    func.call @__ly_long_normalize(%meta, %digits, %lhs_count) : (memref<2xi64>, memref<?xi32>, i64) -> ()
    func.return %header : memref<2xi64>
  }

  func.func private @__ly_long_add_signed_general(%lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_effective_sign: i64, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0]} {
    %zero = arith.constant 0 : i64
    %sign_slot = arith.constant 0 : index
    %lhs_sign = memref.load %lhs_meta[%sign_slot] : memref<2xi64>
    %lhs_zero = arith.cmpi eq, %lhs_sign, %zero : i64
    %result:3 = scf.if %lhs_zero -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) {
      %h = func.call @__ly_long_copy_with_sign(%rhs_effective_sign, %rhs_meta, %rhs_digits) : (i64, memref<2xi64>, memref<?xi32>) -> memref<2xi64>
      %m, %d = func.call @__ly_long_parts(%h) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
      scf.yield %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
    } else {
      %rhs_zero = arith.cmpi eq, %rhs_effective_sign, %zero : i64
      %inner:3 = scf.if %rhs_zero -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) {
        %h = func.call @__ly_long_copy(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> memref<2xi64>
        %m, %d = func.call @__ly_long_parts(%h) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
        scf.yield %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
      } else {
        %same_sign = arith.cmpi eq, %lhs_sign, %rhs_effective_sign : i64
        %combined:3 = scf.if %same_sign -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) {
          %h = func.call @__ly_long_add_abs(%lhs_sign, %lhs_meta, %lhs_digits, %rhs_meta, %rhs_digits) : (i64, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<?xi32>) -> memref<2xi64>
          %m, %d = func.call @__ly_long_parts(%h) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
          scf.yield %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
        } else {
          %cmp = func.call @__ly_long_abs_compare(%lhs_meta, %lhs_digits, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<?xi32>) -> i64
          %lhs_bigger = arith.cmpi sgt, %cmp, %zero : i64
          %abs_result:3 = scf.if %lhs_bigger -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) {
            %h = func.call @__ly_long_sub_abs(%lhs_sign, %lhs_meta, %lhs_digits, %rhs_meta, %rhs_digits) : (i64, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<?xi32>) -> memref<2xi64>
            %m, %d = func.call @__ly_long_parts(%h) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
            scf.yield %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
          } else {
            %rhs_bigger = arith.cmpi slt, %cmp, %zero : i64
            %rhs_result:3 = scf.if %rhs_bigger -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) {
              %h = func.call @__ly_long_sub_abs(%rhs_effective_sign, %rhs_meta, %rhs_digits, %lhs_meta, %lhs_digits) : (i64, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<?xi32>) -> memref<2xi64>
              %m, %d = func.call @__ly_long_parts(%h) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
              scf.yield %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
            } else {
              %h = func.call @LyLong_FromI64(%zero) : (i64) -> memref<2xi64>
              %m, %d = func.call @__ly_long_parts(%h) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
              scf.yield %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
            }
            scf.yield %rhs_result#0, %rhs_result#1, %rhs_result#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
          }
          scf.yield %abs_result#0, %abs_result#1, %abs_result#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
        }
        scf.yield %combined#0, %combined#1, %combined#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
      }
      scf.yield %inner#0, %inner#1, %inner#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
    }
    func.return %result#0 : memref<2xi64>
  }

  func.func @LyLong_FromI64(%value: i64 {ly.runtime.default_i64 = 0 : i64}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 1 : i64, ly.runtime.contract = "builtins.int", ly.runtime.initializer = "__new__"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    // CPython's IS_SMALL_INT: -_PY_NSMALLNEGINTS <= value < _PY_NSMALLPOSINTS.
    %small_lowest = arith.constant -5 : i64
    %small_past_last = arith.constant 257 : i64
    %at_or_above_lowest = arith.cmpi sge, %value, %small_lowest : i64
    %below_past_last = arith.cmpi slt, %value, %small_past_last : i64
    %is_small = arith.andi %at_or_above_lowest, %below_past_last : i1
    cf.cond_br %is_small, ^cached, ^heap

  ^cached:
    func.call @__ly_long_small_ensure() : () -> ()
    %cached_header = func.call @__ly_long_small_slot(%value) : (i64) -> memref<2xi64>
    func.return %cached_header : memref<2xi64>

  ^heap:
    %is_zero = arith.cmpi eq, %value, %zero : i64
    %is_negative = arith.cmpi slt, %value, %zero : i64
    %negative_one = arith.constant -1 : i64
    %positive_one = arith.constant 1 : i64
    %signed_sign = arith.select %is_negative, %negative_one, %positive_one : i1, i64
    %sign = arith.select %is_zero, %zero, %signed_sign : i1, i64
    %abs_value = scf.if %is_negative -> (i64) {
      %negated = arith.subi %zero, %value : i64
      scf.yield %negated : i64
    } else {
      scf.yield %value : i64
    }
    %mask = arith.constant 1073741823 : i64
    %shift30 = arith.constant 30 : i64
    %shift60 = arith.constant 60 : i64
    %digit0_i64 = arith.andi %abs_value, %mask : i64
    %digit1_shifted = arith.shrui %abs_value, %shift30 : i64
    %digit1_i64 = arith.andi %digit1_shifted, %mask : i64
    %digit2_i64 = arith.shrui %abs_value, %shift60 : i64
    %digit1_nonzero = arith.cmpi ne, %digit1_i64, %zero : i64
    %digit2_nonzero = arith.cmpi ne, %digit2_i64, %zero : i64
    %one_or_two = arith.select %digit1_nonzero, %two, %one : i1, i64
    %nonzero_digits = arith.select %digit2_nonzero, %three, %one_or_two : i1, i64
    %ndigits = arith.select %is_zero, %zero, %nonzero_digits : i1, i64
    %alloc_digits_i64 = arith.select %is_zero, %one, %ndigits : i1, i64
    %alloc_digits = arith.index_cast %alloc_digits_i64 : i64 to index
    // One entity, one allocation (see __ly_long_alloc_raw).
    %four_bytes = arith.constant 4 : i64
    %block_prefix = arith.constant 32 : i64
    %digit_bytes = arith.muli %alloc_digits_i64, %four_bytes : i64
    %block_bytes_i64 = arith.addi %digit_bytes, %block_prefix : i64
    %block_bytes = arith.index_cast %block_bytes_i64 : i64 to index
    %block = memref.alloc(%block_bytes) {alignment = 16 : i64} : memref<?xi8>
    %header_offset = arith.constant 0 : index
    %meta_offset = arith.constant 16 : index
    %digits_offset = arith.constant 32 : index
    %header = memref.view %block[%header_offset][] {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<?xi8> to memref<2xi64>
    %meta = memref.view %block[%meta_offset][] : memref<?xi8> to memref<2xi64>
    %digits = memref.view %block[%digits_offset][%alloc_digits] : memref<?xi8> to memref<?xi32>
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %sign_slot = arith.constant 0 : index
    %digit_count_slot = arith.constant 1 : index
    memref.store %one, %header[%refcount_slot] : memref<2xi64>
    memref.store %one, %header[%layout_slot] : memref<2xi64>
    memref.store %sign, %meta[%sign_slot] : memref<2xi64>
    memref.store %ndigits, %meta[%digit_count_slot] : memref<2xi64>
    %digit0_slot = arith.constant 0 : index
    %digit0 = arith.trunci %digit0_i64 : i64 to i32
    memref.store %digit0, %digits[%digit0_slot] : memref<?xi32>
    %has_digit1 = arith.cmpi sge, %ndigits, %two : i64
    scf.if %has_digit1 {
      %digit1_slot = arith.constant 1 : index
      %digit1 = arith.trunci %digit1_i64 : i64 to i32
      memref.store %digit1, %digits[%digit1_slot] : memref<?xi32>
    }
    %has_digit2 = arith.cmpi sge, %ndigits, %three : i64
    scf.if %has_digit2 {
      %digit2_slot = arith.constant 2 : index
      %digit2 = arith.trunci %digit2_i64 : i64 to i32
      memref.store %digit2, %digits[%digit2_slot] : memref<?xi32>
    }
    func.return %header : memref<2xi64>
  }

  // Constructor for compile-time digit spans (big int literals): the lowering
  // splits the literal into 30-bit limbs at compile time and this copies them
  // into a fresh heap object.
  func.func @LyLong_FromDigits(%sign: i64, %digits_in: memref<?xi32>) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.primitive = "from_digits"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %count_index = memref.dim %digits_in, %c0 : memref<?xi32>
    %count = arith.index_cast %count_index : index to i64
    %header = func.call @__ly_long_alloc_raw(%sign, %count) : (i64, i64) -> memref<2xi64>
    %meta, %digits = func.call @__ly_long_parts(%header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    scf.for %iv = %c0 to %count_index step %c1 {
      %digit = memref.load %digits_in[%iv] : memref<?xi32>
      memref.store %digit, %digits[%iv] : memref<?xi32>
    }
    func.call @__ly_long_normalize(%meta, %digits, %count) : (memref<2xi64>, memref<?xi32>, i64) -> ()
    func.return %header : memref<2xi64>
  }

  // ⭐ THE UNBOX THAT REPORTS INSTEAD OF RAISING. `unbox.i64` below is the
  // program-visible conversion and raises for a value wider than the window
  // ("never silently mis-execute"); this one answers the same question for a
  // reader that has a fallback -- the generator frame's int lane, which keeps
  // the BOX beside the word and only needs to know whether the word is usable.
  //
  // ⛔ The digit view is read unconditionally: `__ly_long_view_as_i64` reads
  // digit 0 whatever the magnitude, so the value is meaningless when `fits` is
  // false and never wrong when it is true. Branching around it would cost a
  // region for a load that cannot fault.
  func.func @LyLong_TryAsI64(%header: memref<2xi64> {ly.ownership.object_header}) -> (i64, i1) attributes {ly.runtime.contract = "builtins.int", ly.runtime.primitive = "try_unbox.i64"} {
    %meta, %digits = func.call @__ly_long_parts(%header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %fits = func.call @__ly_long_view_fits_i64(%meta, %digits) : (memref<2xi64>, memref<?xi32>) -> i1
    %value = func.call @__ly_long_view_as_i64(%meta, %digits) : (memref<2xi64>, memref<?xi32>) -> i64
    func.return %value, %fits : i64, i1
  }

  func.func private @__ly_int_immediate_fits(%value: i64) -> i1 {
    %wide = func.call @__ly_addresses_are_word_wide() : () -> i1
    %one = arith.constant 1 : i64
    %thirty_three = arith.constant 33 : i64
    %shift = arith.select %wide, %one, %thirty_three : i1, i64
    %shifted = arith.shli %value, %shift : i64
    %back = arith.shrsi %shifted, %shift : i64
    %fits = arith.cmpi eq, %back, %value : i64
    func.return %fits : i1
  }

  func.func private @__ly_int_to_immediate(%value: i64) -> i64 {
    %one = arith.constant 1 : i64
    %shifted = arith.shli %value, %one : i64
    %word = arith.ori %shifted, %one : i64
    func.return %word : i64
  }

  func.func private @__ly_int_from_immediate(%word: i64) -> i64 {
    %one = arith.constant 1 : i64
    %value = arith.shrsi %word, %one : i64
    func.return %value : i64
  }

  // The int a slot's entity word names, as an owned object: a fresh (or
  // small-table) int for an immediate, the object itself retained otherwise.
  // ⛔ It takes the word as the VIEW the lowering builds from it, not as an
  // i64: that view is what tells ownership the read is inside the container,
  // and a bare word let the container be released between the load and here.
  // An immediate's view is never dereferenced.
  func.func @LyLong_FromSlotWord(%slot_view: memref<2xi64>) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.primitive = "from_slot_word"} {
    %word_idx = memref.extract_aligned_pointer_as_index %slot_view : memref<2xi64> -> index
    %address = arith.index_cast %word_idx : index to i64
    %word = func.call @__ly_slot_word_from_view_address(%address) : (i64) -> i64
    %immediate = func.call @__ly_slot_word_is_immediate(%word) : (i64) -> i1
    %header = scf.if %immediate -> (memref<2xi64>) {
      %value = func.call @__ly_int_from_immediate(%word) : (i64) -> i64
      %fresh = func.call @LyLong_FromI64(%value) : (i64) -> memref<2xi64>
      scf.yield %fresh : memref<2xi64>
    } else {
      func.call @__ly_handle_retain_raw(%word) : (i64) -> ()
      %two = arith.constant 2 : i64
      %view = func.call @__ly_global_view_i64(%word, %two) : (i64, i64) -> memref<?xi64>
      %held = memref.cast %view : memref<?xi64> to memref<2xi64>
      scf.yield %held : memref<2xi64>
    }
    func.return %header : memref<2xi64>
  }

  // ⭐ A READ THAT MAKES NO OBJECT, for an int whose every use can take its
  // i64 (`deferredObject` in the lowering's bundle): the value, whether it is
  // the value (an immediate, or an object that fits 64 bits), and an owned
  // object to fall back on when it is not. For an immediate the fallback is
  // the immortal small int 0 -- never read, because `valid` is true -- so the
  // read allocates nothing; for an object it is the object, retained, since
  // the container may let it go before the fallback is taken.
  func.func @LyLong_ReadSlotWord(%slot_view: memref<2xi64>) -> (memref<2xi64>, i64, i1) attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.primitive = "read_slot_word"} {
    %word_idx = memref.extract_aligned_pointer_as_index %slot_view : memref<2xi64> -> index
    %address = arith.index_cast %word_idx : index to i64
    %word = func.call @__ly_slot_word_from_view_address(%address) : (i64) -> i64
    %immediate = func.call @__ly_slot_word_is_immediate(%word) : (i64) -> i1
    %true = arith.constant true
    %held, %value, %valid = scf.if %immediate -> (memref<2xi64>, i64, i1) {
      %v = func.call @__ly_int_from_immediate(%word) : (i64) -> i64
      %zero = arith.constant 0 : i64
      func.call @__ly_long_small_ensure() : () -> ()
      %stand_in = func.call @__ly_long_small_slot(%zero) : (i64) -> memref<2xi64>
      scf.yield %stand_in, %v, %true : memref<2xi64>, i64, i1
    } else {
      func.call @__ly_handle_retain_raw(%word) : (i64) -> ()
      %v, %ok = func.call @LyLong_TryAsI64(%slot_view) : (memref<2xi64>) -> (i64, i1)
      scf.yield %slot_view, %v, %ok : memref<2xi64>, i64, i1
    }
    func.return %held, %value, %valid : memref<2xi64>, i64, i1
  }

  // The raises a slow arm can take, checked on the operand's i64 BEFORE the
  // arm boxes anything: a box made in the arm is not released when the call
  // in that arm raises. A divisor or a count that is no i64 is not zero and
  // not negative-small, and the boxed call answers for it.
  func.func @LyLong_CheckDivisor(%value: i64, %valid: i1) attributes {ly.runtime.contract = "builtins.int", ly.runtime.primitive = "check_divisor.i64"} {
    %zero = arith.constant 0 : i64
    %is_zero = arith.cmpi eq, %value, %zero : i64
    %raises = arith.andi %valid, %is_zero : i1
    scf.if %raises {
      func.call @__ly_long_raise_division_by_zero() : () -> ()
    }
    func.return
  }

  func.func @LyLong_CheckShift(%value: i64, %valid: i1) attributes {ly.runtime.contract = "builtins.int", ly.runtime.primitive = "check_shift.i64"} {
    %zero = arith.constant 0 : i64
    %negative = arith.cmpi slt, %value, %zero : i64
    %raises = arith.andi %valid, %negative : i1
    scf.if %raises {
      func.call @__ly_long_raise_negative_shift() : () -> ()
    }
    func.return
  }

  // The slow arm of an int operator whose operands are deferred ints (see
  // `LyLong_MaterializeRead`), one per operator: the raises it can take,
  // decided on the i64s and the held objects before anything is made, then
  // the operator on objects -- a held object borrowed as it is, an int made
  // only for an i64 -- and what it made released again.
  // ⛔ Not the arm building those objects itself and calling the operator: a
  // box the arm makes is not released when the operator raises out of it,
  // and every int slow arm then had to be written out as blocks for the
  // unwind cleanup to see it (Runtime/Passes/RegionExits.cpp).
  // ⛔ Not one helper switching on the operator: it names every operator, so
  // a program that adds links the shifts and the bitwise ones too.
  func.func @LyLong_AddDeferred(%av: i64, %aok: i1, %aheld: memref<2xi64> {ly.ownership.object_header}, %bv: i64, %bok: i1, %bheld: memref<2xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.primitive = "deferred.__add__"} {
    %a, %a_made = func.call @__ly_long_deferred_operand(%av, %aok, %aheld) : (i64, i1, memref<2xi64>) -> (memref<2xi64>, i1)
    %b, %b_made = func.call @__ly_long_deferred_operand(%bv, %bok, %bheld) : (i64, i1, memref<2xi64>) -> (memref<2xi64>, i1)
    %result = func.call @LyLong_Add(%a, %b) : (memref<2xi64>, memref<2xi64>) -> memref<2xi64>
    func.call @__ly_long_deferred_release(%a, %a_made) : (memref<2xi64>, i1) -> ()
    func.call @__ly_long_deferred_release(%b, %b_made) : (memref<2xi64>, i1) -> ()
    func.return %result : memref<2xi64>
  }

  func.func @LyLong_SubDeferred(%av: i64, %aok: i1, %aheld: memref<2xi64> {ly.ownership.object_header}, %bv: i64, %bok: i1, %bheld: memref<2xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.primitive = "deferred.__sub__"} {
    %a, %a_made = func.call @__ly_long_deferred_operand(%av, %aok, %aheld) : (i64, i1, memref<2xi64>) -> (memref<2xi64>, i1)
    %b, %b_made = func.call @__ly_long_deferred_operand(%bv, %bok, %bheld) : (i64, i1, memref<2xi64>) -> (memref<2xi64>, i1)
    %result = func.call @LyLong_Sub(%a, %b) : (memref<2xi64>, memref<2xi64>) -> memref<2xi64>
    func.call @__ly_long_deferred_release(%a, %a_made) : (memref<2xi64>, i1) -> ()
    func.call @__ly_long_deferred_release(%b, %b_made) : (memref<2xi64>, i1) -> ()
    func.return %result : memref<2xi64>
  }

  func.func @LyLong_MulDeferred(%av: i64, %aok: i1, %aheld: memref<2xi64> {ly.ownership.object_header}, %bv: i64, %bok: i1, %bheld: memref<2xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.primitive = "deferred.__mul__"} {
    %a, %a_made = func.call @__ly_long_deferred_operand(%av, %aok, %aheld) : (i64, i1, memref<2xi64>) -> (memref<2xi64>, i1)
    %b, %b_made = func.call @__ly_long_deferred_operand(%bv, %bok, %bheld) : (i64, i1, memref<2xi64>) -> (memref<2xi64>, i1)
    %result = func.call @LyLong_Mul(%a, %b) : (memref<2xi64>, memref<2xi64>) -> memref<2xi64>
    func.call @__ly_long_deferred_release(%a, %a_made) : (memref<2xi64>, i1) -> ()
    func.call @__ly_long_deferred_release(%b, %b_made) : (memref<2xi64>, i1) -> ()
    func.return %result : memref<2xi64>
  }

  func.func @LyLong_FloorDivDeferred(%av: i64, %aok: i1, %aheld: memref<2xi64> {ly.ownership.object_header}, %bv: i64, %bok: i1, %bheld: memref<2xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.primitive = "deferred.__floordiv__"} {
    func.call @LyLong_CheckDivisor(%bv, %bok) : (i64, i1) -> ()
    %a, %a_made = func.call @__ly_long_deferred_operand(%av, %aok, %aheld) : (i64, i1, memref<2xi64>) -> (memref<2xi64>, i1)
    %b, %b_made = func.call @__ly_long_deferred_operand(%bv, %bok, %bheld) : (i64, i1, memref<2xi64>) -> (memref<2xi64>, i1)
    %result = func.call @LyLong_FloorDiv(%a, %b) : (memref<2xi64>, memref<2xi64>) -> memref<2xi64>
    func.call @__ly_long_deferred_release(%a, %a_made) : (memref<2xi64>, i1) -> ()
    func.call @__ly_long_deferred_release(%b, %b_made) : (memref<2xi64>, i1) -> ()
    func.return %result : memref<2xi64>
  }

  func.func @LyLong_ModDeferred(%av: i64, %aok: i1, %aheld: memref<2xi64> {ly.ownership.object_header}, %bv: i64, %bok: i1, %bheld: memref<2xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.primitive = "deferred.__mod__"} {
    func.call @LyLong_CheckDivisor(%bv, %bok) : (i64, i1) -> ()
    %a, %a_made = func.call @__ly_long_deferred_operand(%av, %aok, %aheld) : (i64, i1, memref<2xi64>) -> (memref<2xi64>, i1)
    %b, %b_made = func.call @__ly_long_deferred_operand(%bv, %bok, %bheld) : (i64, i1, memref<2xi64>) -> (memref<2xi64>, i1)
    %result = func.call @LyLong_Mod(%a, %b) : (memref<2xi64>, memref<2xi64>) -> memref<2xi64>
    func.call @__ly_long_deferred_release(%a, %a_made) : (memref<2xi64>, i1) -> ()
    func.call @__ly_long_deferred_release(%b, %b_made) : (memref<2xi64>, i1) -> ()
    func.return %result : memref<2xi64>
  }

  func.func @LyLong_LShiftDeferred(%av: i64, %aok: i1, %aheld: memref<2xi64> {ly.ownership.object_header}, %bv: i64, %bok: i1, %bheld: memref<2xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.primitive = "deferred.__lshift__"} {
    func.call @__ly_long_deferred_check_shift(%bv, %bok, %bheld) : (i64, i1, memref<2xi64>) -> ()
    %a, %a_made = func.call @__ly_long_deferred_operand(%av, %aok, %aheld) : (i64, i1, memref<2xi64>) -> (memref<2xi64>, i1)
    %b, %b_made = func.call @__ly_long_deferred_operand(%bv, %bok, %bheld) : (i64, i1, memref<2xi64>) -> (memref<2xi64>, i1)
    %result = func.call @LyLong_LShift(%a, %b) : (memref<2xi64>, memref<2xi64>) -> memref<2xi64>
    func.call @__ly_long_deferred_release(%a, %a_made) : (memref<2xi64>, i1) -> ()
    func.call @__ly_long_deferred_release(%b, %b_made) : (memref<2xi64>, i1) -> ()
    func.return %result : memref<2xi64>
  }

  func.func @LyLong_RShiftDeferred(%av: i64, %aok: i1, %aheld: memref<2xi64> {ly.ownership.object_header}, %bv: i64, %bok: i1, %bheld: memref<2xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.primitive = "deferred.__rshift__"} {
    func.call @__ly_long_deferred_check_shift(%bv, %bok, %bheld) : (i64, i1, memref<2xi64>) -> ()
    %a, %a_made = func.call @__ly_long_deferred_operand(%av, %aok, %aheld) : (i64, i1, memref<2xi64>) -> (memref<2xi64>, i1)
    %b, %b_made = func.call @__ly_long_deferred_operand(%bv, %bok, %bheld) : (i64, i1, memref<2xi64>) -> (memref<2xi64>, i1)
    %result = func.call @LyLong_RShift(%a, %b) : (memref<2xi64>, memref<2xi64>) -> memref<2xi64>
    func.call @__ly_long_deferred_release(%a, %a_made) : (memref<2xi64>, i1) -> ()
    func.call @__ly_long_deferred_release(%b, %b_made) : (memref<2xi64>, i1) -> ()
    func.return %result : memref<2xi64>
  }

  func.func @LyLong_BitAndDeferred(%av: i64, %aok: i1, %aheld: memref<2xi64> {ly.ownership.object_header}, %bv: i64, %bok: i1, %bheld: memref<2xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.primitive = "deferred.__and__"} {
    %a, %a_made = func.call @__ly_long_deferred_operand(%av, %aok, %aheld) : (i64, i1, memref<2xi64>) -> (memref<2xi64>, i1)
    %b, %b_made = func.call @__ly_long_deferred_operand(%bv, %bok, %bheld) : (i64, i1, memref<2xi64>) -> (memref<2xi64>, i1)
    %result = func.call @LyLong_BitAnd(%a, %b) : (memref<2xi64>, memref<2xi64>) -> memref<2xi64>
    func.call @__ly_long_deferred_release(%a, %a_made) : (memref<2xi64>, i1) -> ()
    func.call @__ly_long_deferred_release(%b, %b_made) : (memref<2xi64>, i1) -> ()
    func.return %result : memref<2xi64>
  }

  func.func @LyLong_BitOrDeferred(%av: i64, %aok: i1, %aheld: memref<2xi64> {ly.ownership.object_header}, %bv: i64, %bok: i1, %bheld: memref<2xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.primitive = "deferred.__or__"} {
    %a, %a_made = func.call @__ly_long_deferred_operand(%av, %aok, %aheld) : (i64, i1, memref<2xi64>) -> (memref<2xi64>, i1)
    %b, %b_made = func.call @__ly_long_deferred_operand(%bv, %bok, %bheld) : (i64, i1, memref<2xi64>) -> (memref<2xi64>, i1)
    %result = func.call @LyLong_BitOr(%a, %b) : (memref<2xi64>, memref<2xi64>) -> memref<2xi64>
    func.call @__ly_long_deferred_release(%a, %a_made) : (memref<2xi64>, i1) -> ()
    func.call @__ly_long_deferred_release(%b, %b_made) : (memref<2xi64>, i1) -> ()
    func.return %result : memref<2xi64>
  }

  func.func @LyLong_BitXorDeferred(%av: i64, %aok: i1, %aheld: memref<2xi64> {ly.ownership.object_header}, %bv: i64, %bok: i1, %bheld: memref<2xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.primitive = "deferred.__xor__"} {
    %a, %a_made = func.call @__ly_long_deferred_operand(%av, %aok, %aheld) : (i64, i1, memref<2xi64>) -> (memref<2xi64>, i1)
    %b, %b_made = func.call @__ly_long_deferred_operand(%bv, %bok, %bheld) : (i64, i1, memref<2xi64>) -> (memref<2xi64>, i1)
    %result = func.call @LyLong_BitXor(%a, %b) : (memref<2xi64>, memref<2xi64>) -> memref<2xi64>
    func.call @__ly_long_deferred_release(%a, %a_made) : (memref<2xi64>, i1) -> ()
    func.call @__ly_long_deferred_release(%b, %b_made) : (memref<2xi64>, i1) -> ()
    func.return %result : memref<2xi64>
  }

  func.func @LyLong_EqDeferred(%av: i64, %aok: i1, %aheld: memref<2xi64> {ly.ownership.object_header}, %bv: i64, %bok: i1, %bheld: memref<2xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.int", ly.runtime.primitive = "deferred.__eq__"} {
    %a, %a_made = func.call @__ly_long_deferred_operand(%av, %aok, %aheld) : (i64, i1, memref<2xi64>) -> (memref<2xi64>, i1)
    %b, %b_made = func.call @__ly_long_deferred_operand(%bv, %bok, %bheld) : (i64, i1, memref<2xi64>) -> (memref<2xi64>, i1)
    %result = func.call @LyLong_EqBool(%a, %b) : (memref<2xi64>, memref<2xi64>) -> i1
    func.call @__ly_long_deferred_release(%a, %a_made) : (memref<2xi64>, i1) -> ()
    func.call @__ly_long_deferred_release(%b, %b_made) : (memref<2xi64>, i1) -> ()
    func.return %result : i1
  }

  func.func @LyLong_NeDeferred(%av: i64, %aok: i1, %aheld: memref<2xi64> {ly.ownership.object_header}, %bv: i64, %bok: i1, %bheld: memref<2xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.int", ly.runtime.primitive = "deferred.__ne__"} {
    %a, %a_made = func.call @__ly_long_deferred_operand(%av, %aok, %aheld) : (i64, i1, memref<2xi64>) -> (memref<2xi64>, i1)
    %b, %b_made = func.call @__ly_long_deferred_operand(%bv, %bok, %bheld) : (i64, i1, memref<2xi64>) -> (memref<2xi64>, i1)
    %result = func.call @LyLong_NeBool(%a, %b) : (memref<2xi64>, memref<2xi64>) -> i1
    func.call @__ly_long_deferred_release(%a, %a_made) : (memref<2xi64>, i1) -> ()
    func.call @__ly_long_deferred_release(%b, %b_made) : (memref<2xi64>, i1) -> ()
    func.return %result : i1
  }

  func.func @LyLong_LtDeferred(%av: i64, %aok: i1, %aheld: memref<2xi64> {ly.ownership.object_header}, %bv: i64, %bok: i1, %bheld: memref<2xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.int", ly.runtime.primitive = "deferred.__lt__"} {
    %a, %a_made = func.call @__ly_long_deferred_operand(%av, %aok, %aheld) : (i64, i1, memref<2xi64>) -> (memref<2xi64>, i1)
    %b, %b_made = func.call @__ly_long_deferred_operand(%bv, %bok, %bheld) : (i64, i1, memref<2xi64>) -> (memref<2xi64>, i1)
    %result = func.call @LyLong_LtBool(%a, %b) : (memref<2xi64>, memref<2xi64>) -> i1
    func.call @__ly_long_deferred_release(%a, %a_made) : (memref<2xi64>, i1) -> ()
    func.call @__ly_long_deferred_release(%b, %b_made) : (memref<2xi64>, i1) -> ()
    func.return %result : i1
  }

  func.func @LyLong_LeDeferred(%av: i64, %aok: i1, %aheld: memref<2xi64> {ly.ownership.object_header}, %bv: i64, %bok: i1, %bheld: memref<2xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.int", ly.runtime.primitive = "deferred.__le__"} {
    %a, %a_made = func.call @__ly_long_deferred_operand(%av, %aok, %aheld) : (i64, i1, memref<2xi64>) -> (memref<2xi64>, i1)
    %b, %b_made = func.call @__ly_long_deferred_operand(%bv, %bok, %bheld) : (i64, i1, memref<2xi64>) -> (memref<2xi64>, i1)
    %result = func.call @LyLong_LeBool(%a, %b) : (memref<2xi64>, memref<2xi64>) -> i1
    func.call @__ly_long_deferred_release(%a, %a_made) : (memref<2xi64>, i1) -> ()
    func.call @__ly_long_deferred_release(%b, %b_made) : (memref<2xi64>, i1) -> ()
    func.return %result : i1
  }

  func.func @LyLong_GtDeferred(%av: i64, %aok: i1, %aheld: memref<2xi64> {ly.ownership.object_header}, %bv: i64, %bok: i1, %bheld: memref<2xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.int", ly.runtime.primitive = "deferred.__gt__"} {
    %a, %a_made = func.call @__ly_long_deferred_operand(%av, %aok, %aheld) : (i64, i1, memref<2xi64>) -> (memref<2xi64>, i1)
    %b, %b_made = func.call @__ly_long_deferred_operand(%bv, %bok, %bheld) : (i64, i1, memref<2xi64>) -> (memref<2xi64>, i1)
    %result = func.call @LyLong_GtBool(%a, %b) : (memref<2xi64>, memref<2xi64>) -> i1
    func.call @__ly_long_deferred_release(%a, %a_made) : (memref<2xi64>, i1) -> ()
    func.call @__ly_long_deferred_release(%b, %b_made) : (memref<2xi64>, i1) -> ()
    func.return %result : i1
  }

  func.func @LyLong_GeDeferred(%av: i64, %aok: i1, %aheld: memref<2xi64> {ly.ownership.object_header}, %bv: i64, %bok: i1, %bheld: memref<2xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.int", ly.runtime.primitive = "deferred.__ge__"} {
    %a, %a_made = func.call @__ly_long_deferred_operand(%av, %aok, %aheld) : (i64, i1, memref<2xi64>) -> (memref<2xi64>, i1)
    %b, %b_made = func.call @__ly_long_deferred_operand(%bv, %bok, %bheld) : (i64, i1, memref<2xi64>) -> (memref<2xi64>, i1)
    %result = func.call @LyLong_GeBool(%a, %b) : (memref<2xi64>, memref<2xi64>) -> i1
    func.call @__ly_long_deferred_release(%a, %a_made) : (memref<2xi64>, i1) -> ()
    func.call @__ly_long_deferred_release(%b, %b_made) : (memref<2xi64>, i1) -> ()
    func.return %result : i1
  }

  func.func @LyLong_NegDeferred(%v: i64, %ok: i1, %held: memref<2xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.primitive = "deferred.__neg__"} {
    %a, %a_made = func.call @__ly_long_deferred_operand(%v, %ok, %held) : (i64, i1, memref<2xi64>) -> (memref<2xi64>, i1)
    %result = func.call @LyLong_Neg(%a) : (memref<2xi64>) -> memref<2xi64>
    func.call @__ly_long_deferred_release(%a, %a_made) : (memref<2xi64>, i1) -> ()
    func.return %result : memref<2xi64>
  }

  func.func @LyLong_InvertDeferred(%v: i64, %ok: i1, %held: memref<2xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.primitive = "deferred.__invert__"} {
    %a, %a_made = func.call @__ly_long_deferred_operand(%v, %ok, %held) : (i64, i1, memref<2xi64>) -> (memref<2xi64>, i1)
    %result = func.call @LyLong_Invert(%a) : (memref<2xi64>) -> memref<2xi64>
    func.call @__ly_long_deferred_release(%a, %a_made) : (memref<2xi64>, i1) -> ()
    func.return %result : memref<2xi64>
  }

  func.func @LyLong_AbsDeferred(%v: i64, %ok: i1, %held: memref<2xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.primitive = "deferred.__abs__"} {
    %a, %a_made = func.call @__ly_long_deferred_operand(%v, %ok, %held) : (i64, i1, memref<2xi64>) -> (memref<2xi64>, i1)
    %result = func.call @LyLong_Abs(%a) : (memref<2xi64>) -> memref<2xi64>
    func.call @__ly_long_deferred_release(%a, %a_made) : (memref<2xi64>, i1) -> ()
    func.return %result : memref<2xi64>
  }

  // A shift's raise, on its count as a deferred int: a negative count, which
  // a count that is no i64 can be too.
  func.func private @__ly_long_deferred_check_shift(%value: i64, %valid: i1, %held: memref<2xi64>) attributes {ly.runtime.contract = "builtins.int"} {
    %negative = func.call @__ly_long_deferred_is_negative(%value, %valid, %held) : (i64, i1, memref<2xi64>) -> i1
    scf.if %negative {
      func.call @__ly_long_raise_negative_shift() : () -> ()
    }
    func.return
  }

  // The release of what `__ly_long_deferred_operand` made, and nothing when
  // it lent the held object.
  func.func private @__ly_long_deferred_release(%object: memref<2xi64>, %made: i1) attributes {ly.runtime.contract = "builtins.int"} {
    scf.if %made {
      func.call @LyLong_DecRef(%object) : (memref<2xi64>) -> ()
    }
    func.return
  }

  // An operand of those slow arms as an object to read: the held object,
  // borrowed, when it is one -- an object held with a valid i64 is that
  // value's, since only the stand-in is held in place of one -- else an int
  // made of the i64. `made` says whether the caller releases it.
  // Runtime code (the contract attribute): an owned-or-borrowed result is a
  // shape the frame-ownership insertion would "release before return".
  func.func private @__ly_long_deferred_operand(%value: i64, %valid: i1, %held: memref<2xi64>) -> (memref<2xi64>, i1) attributes {ly.runtime.contract = "builtins.int"} {
    %zero = arith.constant 0 : i64
    func.call @__ly_long_small_ensure() : () -> ()
    %stand_in = func.call @__ly_long_small_slot(%zero) : (i64) -> memref<2xi64>
    %held_idx = memref.extract_aligned_pointer_as_index %held : memref<2xi64> -> index
    %stand_idx = memref.extract_aligned_pointer_as_index %stand_in : memref<2xi64> -> index
    %is_stand_in = arith.cmpi eq, %held_idx, %stand_idx : index
    %make = arith.andi %valid, %is_stand_in : i1
    %object = scf.if %make -> (memref<2xi64>) {
      %fresh = func.call @LyLong_FromI64(%value) : (i64) -> memref<2xi64>
      scf.yield %fresh : memref<2xi64>
    } else {
      scf.yield %held : memref<2xi64>
    }
    func.return %object, %make : memref<2xi64>, i1
  }

  // Whether a deferred int is negative, read off its i64 or its held object.
  func.func private @__ly_long_deferred_is_negative(%value: i64, %valid: i1, %held: memref<2xi64>) -> i1 attributes {ly.runtime.contract = "builtins.int"} {
    %zero = arith.constant 0 : i64
    %negative = scf.if %valid -> (i1) {
      %below = arith.cmpi slt, %value, %zero : i64
      scf.yield %below : i1
    } else {
      %meta, %digits = func.call @__ly_long_parts(%held) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
      %sign_slot = arith.constant 0 : index
      %sign = memref.load %meta[%sign_slot] : memref<2xi64>
      %below = arith.cmpi slt, %sign, %zero : i64
      scf.yield %below : i1
    }
    func.return %negative : i1
  }

  // The stand-in, BORROWED: what a slow arm hands a deferred-int helper as the
  // held object of an int that is only an i64 (a literal), so the arm makes
  // nothing it would have to release.
  func.func @LyLong_DeferredStandInBorrowed() -> memref<2xi64> attributes {ly.runtime.contract = "builtins.int", ly.runtime.primitive = "deferred_stand_in_borrowed"} {
    %zero = arith.constant 0 : i64
    func.call @__ly_long_small_ensure() : () -> ()
    %held = func.call @__ly_long_small_slot(%zero) : (i64) -> memref<2xi64>
    func.return %held : memref<2xi64>
  }

  // What a deferred int holds while its i64 is the value: the immortal small
  // int 0, owned in name only (its release does nothing), so that both arms
  // of a deferred result yield an owned object and the frame owns the merge.
  func.func @LyLong_DeferredStandIn() -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.primitive = "deferred_stand_in"} {
    %zero = arith.constant 0 : i64
    %held = func.call @LyLong_FromI64(%zero) : (i64) -> memref<2xi64>
    func.return %held : memref<2xi64>
  }

  // The object a deferred read stands for, owned: the held object (retained
  // again) when it is one -- not valid, or an object of the value an edge
  // carried along -- else a new int of the value. Only the stand-in is held
  // in place of a value, and it is the small int 0, so an object held with a
  // valid i64 IS that value's object.
  func.func @LyLong_MaterializeRead(%value: i64, %valid: i1, %held: memref<2xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.primitive = "materialize_read"} {
    %zero = arith.constant 0 : i64
    func.call @__ly_long_small_ensure() : () -> ()
    %stand_in = func.call @__ly_long_small_slot(%zero) : (i64) -> memref<2xi64>
    %held_idx = memref.extract_aligned_pointer_as_index %held : memref<2xi64> -> index
    %stand_idx = memref.extract_aligned_pointer_as_index %stand_in : memref<2xi64> -> index
    %is_stand_in = arith.cmpi eq, %held_idx, %stand_idx : index
    %fresh_needed = arith.andi %valid, %is_stand_in : i1
    %header = scf.if %fresh_needed -> (memref<2xi64>) {
      %fresh = func.call @LyLong_FromI64(%value) : (i64) -> memref<2xi64>
      scf.yield %fresh : memref<2xi64>
    } else {
      %idx = memref.extract_aligned_pointer_as_index %held : memref<2xi64> -> index
      %word = arith.index_cast %idx : index to i64
      func.call @__ly_handle_retain_raw(%word) : (i64) -> ()
      scf.yield %held : memref<2xi64>
    }
    func.return %header : memref<2xi64>
  }

  // PyNumber_AsSsize_t(v, NULL): the word an index means -- the value, or the
  // nearest end of the word when the value is wider -- which is how
  // _PyEval_SliceIndex reads a slice bound and str.find its window, so
  // `xs[:2**70]` is the whole list rather than an OverflowError.
  func.func @LyLong_AsI64Clipped(%header: memref<2xi64> {ly.ownership.object_header}) -> i64 attributes {ly.runtime.contract = "builtins.int", ly.runtime.primitive = "unbox.i64.clip"} {
    %meta, %digits = func.call @__ly_long_parts(%header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %fits = func.call @__ly_long_view_fits_i64(%meta, %digits) : (memref<2xi64>, memref<?xi32>) -> i1
    %result = scf.if %fits -> (i64) {
      %v = func.call @__ly_long_view_as_i64(%meta, %digits) : (memref<2xi64>, memref<?xi32>) -> i64
      scf.yield %v : i64
    } else {
      %sign_slot = arith.constant 0 : index
      %zero = arith.constant 0 : i64
      %min = arith.constant -9223372036854775808 : i64
      %max = arith.constant 9223372036854775807 : i64
      %sign = memref.load %meta[%sign_slot] : memref<2xi64>
      %negative = arith.cmpi slt, %sign, %zero : i64
      %end = arith.select %negative, %min, %max : i64
      scf.yield %end : i64
    }
    func.return %result : i64
  }

  // The i64 a deferred read stands for, for an index input: clipped as
  // LyLong_AsI64Clipped clips when the value is wider.
  func.func @LyLong_ReadValueClipped(%value: i64, %valid: i1, %held: memref<2xi64> {ly.ownership.object_header}) -> i64 attributes {ly.runtime.contract = "builtins.int", ly.runtime.primitive = "read_value_clipped"} {
    %result = scf.if %valid -> (i64) {
      scf.yield %value : i64
    } else {
      %v = func.call @LyLong_AsI64Clipped(%held) : (memref<2xi64>) -> i64
      scf.yield %v : i64
    }
    func.return %result : i64
  }

  // The i64 a deferred read stands for, for a callee that takes one: raises
  // as LyLong_AsI64 does when the value is wider.
  func.func @LyLong_ReadValueChecked(%value: i64, %valid: i1, %held: memref<2xi64> {ly.ownership.object_header}) -> i64 attributes {ly.runtime.contract = "builtins.int", ly.runtime.primitive = "read_value_checked"} {
    %result = scf.if %valid -> (i64) {
      scf.yield %value : i64
    } else {
      %v = func.call @LyLong_AsI64(%held) : (memref<2xi64>) -> i64
      scf.yield %v : i64
    }
    func.return %result : i64
  }

  // The i64 a slot's entity word names when it has one: the immediate, or an
  // object's value when it fits. `fits` false means the object is wider.
  func.func @LyLong_SlotWordAsI64(%word: i64) -> (i64, i1) attributes {ly.runtime.contract = "builtins.int", ly.runtime.primitive = "slot_word_as_i64"} {
    %immediate = func.call @__ly_slot_word_is_immediate(%word) : (i64) -> i1
    %true = arith.constant true
    %value, %fits = scf.if %immediate -> (i64, i1) {
      %v = func.call @__ly_int_from_immediate(%word) : (i64) -> i64
      scf.yield %v, %true : i64, i1
    } else {
      %two = arith.constant 2 : i64
      %view = func.call @__ly_global_view_i64(%word, %two) : (i64, i64) -> memref<?xi64>
      %header = memref.cast %view : memref<?xi64> to memref<2xi64>
      %v, %ok = func.call @LyLong_TryAsI64(%header) : (memref<2xi64>) -> (i64, i1)
      scf.yield %v, %ok : i64, i1
    }
    func.return %value, %fits : i64, i1
  }

  // The entity word a slot stores for an int it is handed as an i64: the
  // immediate when it fits, else a new object whose reference the slot takes.
  func.func @LyLong_SlotWordFromI64(%value: i64) -> i64 attributes {ly.runtime.contract = "builtins.int", ly.runtime.primitive = "slot_word_from_i64"} {
    %fits = func.call @__ly_int_immediate_fits(%value) : (i64) -> i1
    %word = scf.if %fits -> (i64) {
      %w = func.call @__ly_int_to_immediate(%value) : (i64) -> i64
      scf.yield %w : i64
    } else {
      %header = func.call @LyLong_FromI64(%value) : (i64) -> memref<2xi64>
      %idx = memref.extract_aligned_pointer_as_index %header : memref<2xi64> -> index
      %w = arith.index_cast %idx : index to i64
      scf.yield %w : i64
    }
    func.return %word : i64
  }

  // The slot word for a deferred int (see `LyLong_MaterializeRead`) whose
  // held object the slot has just been given a reference of (the lowering's
  // aggregate retain), as `LyLong_SlotWordTakingRef` is for an object: its
  // i64's word when the i64 is the value, dropping that reference (the held
  // object is the stand-in, or an object of the same value that an edge
  // carried along), else the held object's, keeping it.
  func.func @LyLong_SlotWordTakingDeferred(%value: i64, %valid: i1, %held: memref<2xi64> {ly.ownership.object_header}) -> i64 attributes {ly.runtime.contract = "builtins.int", ly.runtime.primitive = "slot_word_taking_deferred"} {
    %word = scf.if %valid -> (i64) {
      %w = func.call @LyLong_SlotWordFromI64(%value) : (i64) -> i64
      func.call @LyLong_DecRef(%held) : (memref<2xi64>) -> ()
      scf.yield %w : i64
    } else {
      %idx = memref.extract_aligned_pointer_as_index %held : memref<2xi64> -> index
      %w = arith.index_cast %idx : index to i64
      scf.yield %w : i64
    }
    func.return %word : i64
  }

  // The int counterpart of `LyFloat_SlotWordTakingRef`.
  func.func @LyLong_SlotWordTakingRef(%header: memref<2xi64> {ly.ownership.object_header}) -> i64 attributes {ly.runtime.contract = "builtins.int", ly.runtime.primitive = "slot_word_taking_ref"} {
    %value, %ok = func.call @LyLong_TryAsI64(%header) : (memref<2xi64>) -> (i64, i1)
    %narrow = func.call @__ly_int_immediate_fits(%value) : (i64) -> i1
    %fits = arith.andi %ok, %narrow : i1
    %word = scf.if %fits -> (i64) {
      func.call @LyLong_DecRef(%header) : (memref<2xi64>) -> ()
      %w = func.call @__ly_int_to_immediate(%value) : (i64) -> i64
      scf.yield %w : i64
    } else {
      %idx = memref.extract_aligned_pointer_as_index %header : memref<2xi64> -> index
      %w = arith.index_cast %idx : index to i64
      scf.yield %w : i64
    }
    func.return %word : i64
  }

  func.func @LyLong_AsI64(%header: memref<2xi64> {ly.ownership.object_header}) -> i64 attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__int__", ly.runtime.primitive = "unbox.i64"} {
    %meta, %digits = func.call @__ly_long_parts(%header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    // Reading a wider value through the i64 window would silently truncate;
    // raise instead (never silently mis-execute).
    %fits = func.call @__ly_long_view_fits_i64(%meta, %digits) : (memref<2xi64>, memref<?xi32>) -> i1
    cf.cond_br %fits, ^ok, ^too_large

  ^too_large:
    func.call @__ly_long_raise_too_large() : () -> ()
    cf.br ^ok

  ^ok:
    %result = func.call @__ly_long_view_as_i64(%meta, %digits) : (memref<2xi64>, memref<?xi32>) -> i64
    func.return %result : i64
  }

  // 2^e for e in [-1022, 1023], built from the IEEE 754 exponent field. Powers
  // in this range are normal, so the significand is all zeros.
  func.func private @__ly_long_pow2_f64(%e: i64) -> f64 {
    %bias = arith.constant 1023 : i64
    %mant_bits = arith.constant 52 : i64
    %biased = arith.addi %e, %bias : i64
    %bits = arith.shli %biased, %mant_bits : i64
    %result = arith.bitcast %bits : i64 to f64
    func.return %result : f64
  }

  // Correctly-rounded int -> f64 (CPython PyLong_AsDouble / _PyLong_Frexp):
  // the top 55 bits are extracted exactly, every lower bit is folded into a
  // sticky bit, and the 55-bit window is rounded to 53 bits half-to-even.
  // The second result reports overflow (|x| rounds to >= 2^1024); the caller
  // decides which OverflowError-equivalent to raise.
  func.func private @__ly_long_view_as_f64_checked(%meta: memref<2xi64>, %digits: memref<?xi32>) -> (f64, i1) {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %five = arith.constant 5 : i64
    %thirty = arith.constant 30 : i64
    %c30_minus = arith.constant 30 : i64
    %c60 = arith.constant 60 : i64
    %false = arith.constant false
    %zero_f64 = arith.constant 0.0 : f64
    %sign_slot = arith.constant 0 : index
    %count_slot = arith.constant 1 : index
    %sign = memref.load %meta[%sign_slot] : memref<2xi64>
    %count = memref.load %meta[%count_slot] : memref<2xi64>
    %is_negative = arith.cmpi slt, %sign, %zero : i64
    %nbits = func.call @__ly_long_bit_length(%meta, %digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %is_zero = arith.cmpi eq, %nbits, %zero : i64
    %magnitude:2 = scf.if %is_zero -> (f64, i1) {
      scf.yield %zero_f64, %false : f64, i1
    } else {
      %c63 = arith.constant 63 : i64
      %small = arith.cmpi sle, %nbits, %c63 : i64
      %inner:2 = scf.if %small -> (f64, i1) {
        // Up to 63 bits: assemble the unsigned magnitude and let uitofp do the
        // (correct, half-to-even) rounding.
        %c0 = arith.constant 0 : index
        %digit0_i32 = memref.load %digits[%c0] : memref<?xi32>
        %digit0 = arith.extui %digit0_i32 : i32 to i64
        %has_digit1 = arith.cmpi uge, %count, %two : i64
        %digit1 = scf.if %has_digit1 -> (i64) {
          %idx = arith.constant 1 : index
          %d_i32 = memref.load %digits[%idx] : memref<?xi32>
          %d = arith.extui %d_i32 : i32 to i64
          scf.yield %d : i64
        } else {
          scf.yield %zero : i64
        }
        %has_digit2 = arith.cmpi uge, %count, %three : i64
        %digit2 = scf.if %has_digit2 -> (i64) {
          %idx = arith.constant 2 : index
          %d_i32 = memref.load %digits[%idx] : memref<?xi32>
          %d = arith.extui %d_i32 : i32 to i64
          scf.yield %d : i64
        } else {
          scf.yield %zero : i64
        }
        %d1_shifted = arith.shli %digit1, %thirty : i64
        %d2_shifted = arith.shli %digit2, %c60 : i64
        %with_d1 = arith.ori %digit0, %d1_shifted : i64
        %mag = arith.ori %with_d1, %d2_shifted : i64
        %mag_f = arith.uitofp %mag : i64 to f64
        scf.yield %mag_f, %false : f64, i1
      } else {
        // shift >= 9 here, so divui/remui below are safe.
        %c55 = arith.constant 55 : i64
        %shift = arith.subi %nbits, %c55 : i64
        %digit_pos = arith.divui %shift, %thirty : i64
        %bit_pos = arith.remui %shift, %thirty : i64
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        // Window limb j (digit_pos + 0/1/2), zero when past the top limb.
        %limb0_index = arith.index_cast %digit_pos : i64 to index
        %l0_i32 = memref.load %digits[%limb0_index] : memref<?xi32>
        %l0 = arith.extui %l0_i32 : i32 to i64
        %pos1 = arith.addi %digit_pos, %one : i64
        %has_l1 = arith.cmpi ult, %pos1, %count : i64
        %l1 = scf.if %has_l1 -> (i64) {
          %idx = arith.index_cast %pos1 : i64 to index
          %d_i32 = memref.load %digits[%idx] : memref<?xi32>
          %d = arith.extui %d_i32 : i32 to i64
          scf.yield %d : i64
        } else {
          scf.yield %zero : i64
        }
        %pos2 = arith.addi %digit_pos, %two : i64
        %has_l2 = arith.cmpi ult, %pos2, %count : i64
        %l2 = scf.if %has_l2 -> (i64) {
          %idx = arith.index_cast %pos2 : i64 to index
          %d_i32 = memref.load %digits[%idx] : memref<?xi32>
          %d = arith.extui %d_i32 : i32 to i64
          scf.yield %d : i64
        } else {
          scf.yield %zero : i64
        }
        // top55 = magnitude >> shift; a nonzero l2 implies bit_pos >= 6, so
        // the shift amounts stay below 64 and nothing is truncated.
        %l0_part = arith.shrui %l0, %bit_pos : i64
        %l1_amount = arith.subi %c30_minus, %bit_pos : i64
        %l1_part = arith.shli %l1, %l1_amount : i64
        %l2_amount = arith.subi %c60, %bit_pos : i64
        %l2_part = arith.shli %l2, %l2_amount : i64
        %top_a = arith.ori %l0_part, %l1_part : i64
        %top55 = arith.ori %top_a, %l2_part : i64
        // Sticky: any bit below the window.
        %low_mask_full = arith.shli %one, %bit_pos : i64
        %low_mask = arith.subi %low_mask_full, %one : i64
        %l0_low = arith.andi %l0, %low_mask : i64
        %sticky_low = arith.cmpi ne, %l0_low, %zero : i64
        %digit_pos_index = arith.index_cast %digit_pos : i64 to index
        %sticky_rest = scf.for %iv = %c0 to %digit_pos_index step %c1 iter_args(%seen = %false) -> (i1) {
          %d_i32 = memref.load %digits[%iv] : memref<?xi32>
          %zero_i32 = arith.constant 0 : i32
          %nonzero = arith.cmpi ne, %d_i32, %zero_i32 : i32
          %next = arith.ori %seen, %nonzero : i1
          scf.yield %next : i1
        }
        %sticky = arith.ori %sticky_low, %sticky_rest : i1
        %sticky_i64 = arith.extui %sticky : i1 to i64
        %low = arith.ori %top55, %sticky_i64 : i64
        // Round half-to-even over the 2 extra bits (guard mask = 2; 3*mask-1
        // = 5 covers the sticky bit and the bit above the guard).
        %guard = arith.andi %low, %two : i64
        %guard_set = arith.cmpi ne, %guard, %zero : i64
        %near = arith.andi %low, %five : i64
        %near_set = arith.cmpi ne, %near, %zero : i64
        %round_up = arith.andi %guard_set, %near_set : i1
        %bumped = arith.addi %low, %two : i64
        %rounded = arith.select %round_up, %bumped, %low : i1, i64
        %m53 = arith.shrui %rounded, %two : i64
        // Overflow when the rounded magnitude reaches 2^1024.
        %c1024 = arith.constant 1024 : i64
        %c2_53 = arith.constant 9007199254740992 : i64
        %too_wide = arith.cmpi sgt, %nbits, %c1024 : i64
        %at_limit = arith.cmpi eq, %nbits, %c1024 : i64
        %mant_carry = arith.cmpi eq, %m53, %c2_53 : i64
        %carry_overflow = arith.andi %at_limit, %mant_carry : i1
        %overflow = arith.ori %too_wide, %carry_overflow : i1
        %big:2 = scf.if %overflow -> (f64, i1) {
          %true = arith.constant true
          scf.yield %zero_f64, %true : f64, i1
        } else {
          %c53 = arith.constant 53 : i64
          %e2 = arith.subi %nbits, %c53 : i64
          %scale = func.call @__ly_long_pow2_f64(%e2) : (i64) -> f64
          %m53_f = arith.uitofp %m53 : i64 to f64
          %mag_f = arith.mulf %m53_f, %scale : f64
          scf.yield %mag_f, %false : f64, i1
        }
        scf.yield %big#0, %big#1 : f64, i1
      }
      scf.yield %inner#0, %inner#1 : f64, i1
    }
    %negated = arith.negf %magnitude#0 : f64
    %signed = arith.select %is_negative, %negated, %magnitude#0 : i1, f64
    func.return %signed, %magnitude#1 : f64, i1
  }

  // Conversion entry used by unbox.f64 / __float__: correctly rounded, and
  // magnitudes at or beyond 2^1024 raise like CPython's float(int).
  func.func private @__ly_long_view_as_f64(%meta: memref<2xi64>, %digits: memref<?xi32>) -> f64 {
    %value, %overflow = func.call @__ly_long_view_as_f64_checked(%meta, %digits) : (memref<2xi64>, memref<?xi32>) -> (f64, i1)
    cf.cond_br %overflow, ^too_large, ^ok

  ^too_large:
    func.call @__ly_long_raise_too_large_for_float() : () -> ()
    cf.br ^ok

  ^ok:
    func.return %value : f64
  }

  func.func @LyLong_AsF64(%header: memref<2xi64> {ly.ownership.object_header}) -> f64 attributes {ly.runtime.contract = "builtins.int", ly.runtime.primitive = "unbox.f64"} {
    %meta_raw, %digits_raw = func.call @__ly_long_parts(%header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %meta, %digits = func.call @__ly_long_operand_view(%meta_raw, %digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %value = func.call @__ly_long_view_as_f64(%meta, %digits) : (memref<2xi64>, memref<?xi32>) -> f64
    func.return %value : f64
  }

  // Runtime-level float(int) (`__float__` on int): correctly-rounded digit
  // conversion boxed as a float object.
  func.func @LyLong_Float(%header: memref<2xi64> {ly.ownership.object_header}) -> memref<3xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__float__", ly.runtime.result_contract = "builtins.float"} {
    %meta_raw, %digits_raw = func.call @__ly_long_parts(%header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %meta, %digits = func.call @__ly_long_operand_view(%meta_raw, %digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %value = func.call @__ly_long_view_as_f64(%meta, %digits) : (memref<2xi64>, memref<?xi32>) -> f64
    %h = func.call @LyFloat_FromF64(%value) : (f64) -> memref<3xi64>
    func.return %h : memref<3xi64>
  }

  func.func @LyLong_Init(%header: memref<2xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__init__"} {
    func.return
  }

  func.func @LyLong_Bool(%header: memref<2xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__bool__"} {
    %meta_raw, %digits_raw = func.call @__ly_long_parts(%header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %meta, %digits = func.call @__ly_long_operand_view(%meta_raw, %digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %count_slot = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %count = memref.load %meta[%count_slot] : memref<2xi64>
    %result = arith.cmpi ne, %count, %zero : i64
    func.return %result : i1
  }

  func.func @LyLong_Pos(%header: memref<2xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__pos__"} {
    %meta_raw, %digits_raw = func.call @__ly_long_parts(%header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %meta, %digits = func.call @__ly_long_operand_view(%meta_raw, %digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %h = func.call @__ly_long_copy(%meta, %digits) : (memref<2xi64>, memref<?xi32>) -> memref<2xi64>
    func.return %h : memref<2xi64>
  }

  func.func @LyLong_Neg(%header: memref<2xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__neg__"} {
    %meta_raw, %digits_raw = func.call @__ly_long_parts(%header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %meta, %digits = func.call @__ly_long_operand_view(%meta_raw, %digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %sign_slot = arith.constant 0 : index
    %zero = arith.constant 0 : i64
    %sign = memref.load %meta[%sign_slot] : memref<2xi64>
    %negated = arith.subi %zero, %sign : i64
    %h = func.call @__ly_long_copy_with_sign(%negated, %meta, %digits) : (i64, memref<2xi64>, memref<?xi32>) -> memref<2xi64>
    func.return %h : memref<2xi64>
  }

  func.func @LyLong_Invert(%header: memref<2xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__invert__"} {
    %neg_h = func.call @LyLong_Neg(%header) : (memref<2xi64>) -> memref<2xi64>
    %one = arith.constant 1 : i64
    %one_h = func.call @LyLong_FromI64(%one) : (i64) -> memref<2xi64>
    %result_h = func.call @LyLong_Sub(%neg_h, %one_h) : (memref<2xi64>, memref<2xi64>) -> memref<2xi64>
    func.call @LyLong_DecRef(%neg_h) : (memref<2xi64>) -> ()
    func.call @LyLong_DecRef(%one_h) : (memref<2xi64>) -> ()
    func.return %result_h : memref<2xi64>
  }

  // Reads a (meta, digits) view whose magnitude is known to fit i64. Used by
  // AsI64 and by the small-operand fast paths of the arithmetic entry points.
  func.func private @__ly_long_view_as_i64(%meta: memref<2xi64>, %digits: memref<?xi32>) -> i64 {
    %sign_slot = arith.constant 0 : index
    %digit_count_slot = arith.constant 1 : index
    %sign = memref.load %meta[%sign_slot] : memref<2xi64>
    %ndigits = memref.load %meta[%digit_count_slot] : memref<2xi64>
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %shift30 = arith.constant 30 : i64
    %shift60 = arith.constant 60 : i64
    %digit0_index = arith.constant 0 : index
    %digit0_i32 = memref.load %digits[%digit0_index] : memref<?xi32>
    %digit0 = arith.extui %digit0_i32 : i32 to i64
    %has_digit1 = arith.cmpi uge, %ndigits, %two : i64
    %digit1 = scf.if %has_digit1 -> (i64) {
      %idx = arith.constant 1 : index
      %digit_i32 = memref.load %digits[%idx] : memref<?xi32>
      %digit = arith.extui %digit_i32 : i32 to i64
      scf.yield %digit : i64
    } else {
      scf.yield %zero : i64
    }
    %has_digit2 = arith.cmpi uge, %ndigits, %three : i64
    %digit2 = scf.if %has_digit2 -> (i64) {
      %idx = arith.constant 2 : index
      %digit_i32 = memref.load %digits[%idx] : memref<?xi32>
      %digit = arith.extui %digit_i32 : i32 to i64
      scf.yield %digit : i64
    } else {
      scf.yield %zero : i64
    }
    %digit1_shifted = arith.shli %digit1, %shift30 : i64
    %with_digit1 = arith.ori %digit0, %digit1_shifted : i64
    %digit2_shifted = arith.shli %digit2, %shift60 : i64
    %magnitude = arith.ori %with_digit1, %digit2_shifted : i64
    %negated = arith.subi %zero, %magnitude : i64
    %is_negative = arith.cmpi slt, %sign, %zero : i64
    %signed = arith.select %is_negative, %negated, %magnitude : i1, i64
    %is_zero = arith.cmpi eq, %ndigits, %zero : i64
    %result = arith.select %is_zero, %zero, %signed : i1, i64
    func.return %result : i64
  }

  func.func private @__ly_long_view_fits_i64(%meta: memref<2xi64>, %digits: memref<?xi32>) -> i1 {
    %zero = arith.constant 0 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %eight = arith.constant 8 : i64
    %sign_slot = arith.constant 0 : index
    %digit_count_slot = arith.constant 1 : index
    %ndigits = memref.load %meta[%digit_count_slot] : memref<2xi64>
    %fits_two_limbs = arith.cmpi sle, %ndigits, %two : i64
    %fits = scf.if %fits_two_limbs -> (i1) {
      %true = arith.constant true
      scf.yield %true : i1
    } else {
      %has_three_limbs = arith.cmpi eq, %ndigits, %three : i64
      %three_limb_fits = scf.if %has_three_limbs -> (i1) {
        %digit0_slot = arith.constant 0 : index
        %digit1_slot = arith.constant 1 : index
        %digit2_slot = arith.constant 2 : index
        %digit0_i32 = memref.load %digits[%digit0_slot] : memref<?xi32>
        %digit1_i32 = memref.load %digits[%digit1_slot] : memref<?xi32>
        %digit2_i32 = memref.load %digits[%digit2_slot] : memref<?xi32>
        %digit0 = arith.extui %digit0_i32 : i32 to i64
        %digit1 = arith.extui %digit1_i32 : i32 to i64
        %digit2 = arith.extui %digit2_i32 : i32 to i64
        %high_lt_limit = arith.cmpi ult, %digit2, %eight : i64
        %high_eq_limit = arith.cmpi eq, %digit2, %eight : i64
        %low0_zero = arith.cmpi eq, %digit0, %zero : i64
        %low1_zero = arith.cmpi eq, %digit1, %zero : i64
        %low_zero = arith.andi %low0_zero, %low1_zero : i1
        %sign = memref.load %meta[%sign_slot] : memref<2xi64>
        %negative = arith.cmpi slt, %sign, %zero : i64
        %min_i64 = arith.andi %high_eq_limit, %low_zero : i1
        %negative_min_i64 = arith.andi %min_i64, %negative : i1
        %result = arith.ori %high_lt_limit, %negative_min_i64 : i1
        scf.yield %result : i1
      } else {
        %false = arith.constant false
        scf.yield %false : i1
      }
      scf.yield %three_limb_fits : i1
    }
    func.return %fits : i1
  }

  func.func private @__ly_long_raise_too_large() {
    %class_id = arith.constant 104 : i64
    %length = arith.constant 51 : i64
    %message_static = memref.get_global @__ly_long_msg_int_too_large : memref<51xi8>
    %message = memref.cast %message_static : memref<51xi8> to memref<?xi8>
    func.call @__ly_raise_static_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  // CPython appends the repr of the offending input:
  //   ValueError: invalid literal for int() with base 10: 'abc'
  // The prefix alone left the caller unable to tell WHICH string failed, which
  // is the whole content of the message. The concatenation the older comment
  // said was unavailable is `LyUnicode_Concat` (objects/unicode.mlir).
  func.func private @__ly_long_raise_invalid_literal(%subject_header: memref<2xi64> {ly.ownership.object_header}, %subject_bytes: memref<?xi8>) {
    %class_id = arith.constant 53 : i64
    %start = arith.constant 0 : index
    %prefix_length = arith.constant 40 : i64
    %prefix_static = memref.get_global @__ly_long_msg_invalid_int_literal_prefix : memref<40xi8>
    %prefix_bytes = memref.cast %prefix_static : memref<40xi8> to memref<?xi8>
    %prefix_h, %prefix_b = func.call @__ly_unicode_from_valid_utf8(%prefix_bytes, %start, %prefix_length) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %quoted_h, %quoted_b = func.call @LyUnicode_Repr(%subject_header, %subject_bytes) : (memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    %full_h, %full_b = func.call @LyUnicode_Concat(%prefix_h, %prefix_b, %quoted_h, %quoted_b) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.call @LyUnicode_DecRef(%prefix_h) : (memref<2xi64>) -> ()
    func.call @LyUnicode_DecRef(%quoted_h) : (memref<2xi64>) -> ()
    %exception:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    %initialized:3 = func.call @LyBaseException_Init(%exception#0, %exception#1, %exception#2, %full_h, %full_b) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.call @LyEH_ThrowException(%initialized#0, %initialized#1, %initialized#2) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @__ly_long_raise_float_nan() {
    %class_id = arith.constant 53 : i64
    %length = arith.constant 35 : i64
    %message_static = memref.get_global @__ly_long_msg_float_nan : memref<35xi8>
    %message = memref.cast %message_static : memref<35xi8> to memref<?xi8>
    func.call @__ly_raise_static_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @__ly_long_raise_float_infinity() {
    %class_id = arith.constant 104 : i64
    %length = arith.constant 40 : i64
    %message_static = memref.get_global @__ly_long_msg_float_infinity : memref<40xi8>
    %message = memref.cast %message_static : memref<40xi8> to memref<?xi8>
    func.call @__ly_raise_static_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @__ly_long_raise_too_large_for_float() {
    %class_id = arith.constant 104 : i64
    %length = arith.constant 33 : i64
    %message_static = memref.get_global @__ly_long_msg_int_too_large_float : memref<33xi8>
    %message = memref.cast %message_static : memref<33xi8> to memref<?xi8>
    func.call @__ly_raise_static_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @__ly_long_raise_div_result_too_large() {
    %class_id = arith.constant 104 : i64
    %length = arith.constant 45 : i64
    %message_static = memref.get_global @__ly_long_msg_div_result_too_large : memref<45xi8>
    %message = memref.cast %message_static : memref<45xi8> to memref<?xi8>
    func.call @__ly_raise_static_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @__ly_long_raise_zero_negative_power() {
    %class_id = arith.constant 61 : i64
    %length = arith.constant 24 : i64
    %message_static = memref.get_global @__ly_long_msg_zero_negative_power : memref<24xi8>
    %message = memref.cast %message_static : memref<24xi8> to memref<?xi8>
    func.call @__ly_raise_static_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @__ly_long_raise_fractional_power_negative() {
    %class_id = arith.constant 53 : i64
    %length = arith.constant 54 : i64
    %message_static = memref.get_global @__ly_long_msg_fractional_power_negative : memref<54xi8>
    %message = memref.cast %message_static : memref<54xi8> to memref<?xi8>
    func.call @__ly_raise_static_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @__ly_long_raise_pow_negative() {
    %class_id = arith.constant 53 : i64
    %length = arith.constant 91 : i64
    %message_static = memref.get_global @__ly_long_msg_pow_negative_exponent : memref<91xi8>
    %message = memref.cast %message_static : memref<91xi8> to memref<?xi8>
    func.call @__ly_raise_static_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  // CPython's _PyLong_New past MAX_LONG_DIGITS, which a left shift by a count
  // past the word reaches too (divmod_shift clips it there).
  func.func private @__ly_long_raise_too_many_digits() {
    %class_id = arith.constant 104 : i64
    %length = arith.constant 26 : i64
    %message_static = memref.get_global @__ly_long_msg_too_many_digits : memref<26xi8>
    %message = memref.cast %message_static : memref<26xi8> to memref<?xi8>
    func.call @__ly_raise_static_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  // CPython's int.bit_length(): the width of the ABSOLUTE value, 0 for zero.
  // The magnitude view has already dropped the sign, so the shared helper below
  // -- the one pow, division and _random's bounded draw count with -- answers it
  // unchanged, and this wrapper exists only to put it on the manifest surface.
  func.func @LyLong_BitLength(%header: memref<2xi64> {ly.ownership.object_header}) -> i64 attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "bit_length"} {
    %meta_raw, %digits_raw = func.call @__ly_long_parts(%header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %meta, %digits = func.call @__ly_long_operand_view(%meta_raw, %digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %bits = func.call @__ly_long_bit_length(%meta, %digits) : (memref<2xi64>, memref<?xi32>) -> i64
    func.return %bits : i64
  }

  func.func private @__ly_long_bit_length(%meta: memref<2xi64>, %digits: memref<?xi32>) -> i64 {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %thirty = arith.constant 30 : i64
    %count_slot = arith.constant 1 : index
    %count = memref.load %meta[%count_slot] : memref<2xi64>
    %is_zero = arith.cmpi eq, %count, %zero : i64
    %result = scf.if %is_zero -> (i64) {
      scf.yield %zero : i64
    } else {
      %c1 = arith.constant 1 : index
      %count_index = arith.index_cast %count : i64 to index
      %top_index = arith.subi %count_index, %c1 : index
      %top_i32 = memref.load %digits[%top_index] : memref<?xi32>
      %top = arith.extui %top_i32 : i32 to i64
      %c0 = arith.constant 0 : index
      %c30 = arith.constant 30 : index
      %top_bits = scf.for %iv = %c0 to %c30 step %c1 iter_args(%bits = %zero) -> (i64) {
        %iv_i64 = arith.index_cast %iv : index to i64
        %shifted = arith.shrui %top, %iv_i64 : i64
        %nonzero = arith.cmpi ne, %shifted, %zero : i64
        %next_bits = arith.addi %iv_i64, %one : i64
        %next = arith.select %nonzero, %next_bits, %bits : i1, i64
        scf.yield %next : i64
      }
      %full = arith.subi %count, %one : i64
      %full_bits = arith.muli %full, %thirty : i64
      %total = arith.addi %full_bits, %top_bits : i64
      scf.yield %total : i64
    }
    func.return %result : i64
  }

  // |lhs| = q * |rhs| + r with 0 <= r < |rhs|. Requires rhs != 0. Both results
  // are freshly allocated (never the immortal small-int cache), so callers may
  // flip signs / increment magnitudes in place before publishing them.
  //
  // Multi-digit divisors use bit-by-bit shift-subtract long division rather
  // than Knuth's algorithm D: the quotient-digit estimation/correction loop is
  // hard to verify in handwritten scf/arith, and correctness is the current
  // gate; swap in D behind this same contract if division ever profiles hot.
  func.func private @__ly_long_divmod_abs(%lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>) attributes {ly.ownership.owned_result_contracts = ["builtins.int", "builtins.int"], ly.ownership.owned_results = [0, 1]} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %thirty = arith.constant 30 : i64
    %base = arith.constant 1073741824 : i64
    %mask = arith.constant 1073741823 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %sign_slot = arith.constant 0 : index
    %count_slot = arith.constant 1 : index
    %cmp = func.call @__ly_long_abs_compare(%lhs_meta, %lhs_digits, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<?xi32>) -> i64
    %lhs_smaller = arith.cmpi slt, %cmp, %zero : i64
    %result:6 = scf.if %lhs_smaller -> (memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<2xi64>, memref<?xi32>) {
      %qh = func.call @__ly_long_alloc_raw(%zero, %zero) : (i64, i64) -> memref<2xi64>
      %qm, %qd = func.call @__ly_long_parts(%qh) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
      %rh = func.call @__ly_long_copy_with_sign(%one, %lhs_meta, %lhs_digits) : (i64, memref<2xi64>, memref<?xi32>) -> memref<2xi64>
      %rm, %rd = func.call @__ly_long_parts(%rh) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
      scf.yield %qh, %qm, %qd, %rh, %rm, %rd : memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<2xi64>, memref<?xi32>
    } else {
      %lhs_count = memref.load %lhs_meta[%count_slot] : memref<2xi64>
      %rhs_count = memref.load %rhs_meta[%count_slot] : memref<2xi64>
      // One spare limb so the floor adjustment (q + 1 in magnitude) can never
      // carry out of the allocation.
      %q_capacity = arith.addi %lhs_count, %one : i64
      %single = arith.cmpi eq, %rhs_count, %one : i64
      %inner:6 = scf.if %single -> (memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<2xi64>, memref<?xi32>) {
        %divisor_i32 = memref.load %rhs_digits[%c0] : memref<?xi32>
        %divisor = arith.extui %divisor_i32 : i32 to i64
        %qh = func.call @__ly_long_alloc_raw(%one, %q_capacity) : (i64, i64) -> memref<2xi64>
        %qm, %qd = func.call @__ly_long_parts(%qh) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
        %count_index = arith.index_cast %lhs_count : i64 to index
        %rem_final = scf.for %iv = %c0 to %count_index step %c1 iter_args(%rem = %zero) -> (i64) {
          %iv_next = arith.addi %iv, %c1 : index
          %rev = arith.subi %count_index, %iv_next : index
          %digit_i32 = memref.load %lhs_digits[%rev] : memref<?xi32>
          %digit = arith.extui %digit_i32 : i32 to i64
          %scaled = arith.muli %rem, %base : i64
          %acc = arith.addi %scaled, %digit : i64
          %q_digit = arith.divui %acc, %divisor : i64
          %rem_next = arith.remui %acc, %divisor : i64
          %q_digit_i32 = arith.trunci %q_digit : i64 to i32
          memref.store %q_digit_i32, %qd[%rev] : memref<?xi32>
          scf.yield %rem_next : i64
        }
        func.call @__ly_long_normalize(%qm, %qd, %q_capacity) : (memref<2xi64>, memref<?xi32>, i64) -> ()
        %rem_sign = arith.cmpi ne, %rem_final, %zero : i64
        %r_sign = arith.select %rem_sign, %one, %zero : i1, i64
        %rh = func.call @__ly_long_alloc_raw(%r_sign, %one) : (i64, i64) -> memref<2xi64>
        %rm, %rd = func.call @__ly_long_parts(%rh) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
        %rem_i32 = arith.trunci %rem_final : i64 to i32
        memref.store %rem_i32, %rd[%c0] : memref<?xi32>
        func.call @__ly_long_normalize(%rm, %rd, %one) : (memref<2xi64>, memref<?xi32>, i64) -> ()
        scf.yield %qh, %qm, %qd, %rh, %rm, %rd : memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<2xi64>, memref<?xi32>
      } else {
        %nbits = func.call @__ly_long_bit_length(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
        %qh = func.call @__ly_long_alloc_raw(%one, %q_capacity) : (i64, i64) -> memref<2xi64>
        %qm, %qd = func.call @__ly_long_parts(%qh) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
        %r_capacity = arith.addi %rhs_count, %one : i64
        %rh = func.call @__ly_long_alloc_raw(%one, %r_capacity) : (i64, i64) -> memref<2xi64>
        %rm, %rd = func.call @__ly_long_parts(%rh) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
        %r_limbs = arith.index_cast %r_capacity : i64 to index
        %nbits_index = arith.index_cast %nbits : i64 to index
        scf.for %k = %c0 to %nbits_index step %c1 {
          %k_i64 = arith.index_cast %k : index to i64
          %j_plus = arith.subi %nbits, %k_i64 : i64
          %j = arith.subi %j_plus, %one : i64
          %digit_pos = arith.divui %j, %thirty : i64
          %bit_pos = arith.remui %j, %thirty : i64
          %digit_index = arith.index_cast %digit_pos : i64 to index
          %src_i32 = memref.load %lhs_digits[%digit_index] : memref<?xi32>
          %src = arith.extui %src_i32 : i32 to i64
          %shifted_src = arith.shrui %src, %bit_pos : i64
          %bit = arith.andi %shifted_src, %one : i64
          // r = (r << 1) | bit; r stays < 2*|rhs| <= capacity of r_capacity limbs.
          %shift_carry = scf.for %i = %c0 to %r_limbs step %c1 iter_args(%carry = %bit) -> (i64) {
            %d_i32 = memref.load %rd[%i] : memref<?xi32>
            %d = arith.extui %d_i32 : i32 to i64
            %doubled = arith.shli %d, %one : i64
            %with_carry = arith.ori %doubled, %carry : i64
            %out = arith.andi %with_carry, %mask : i64
            %out_i32 = arith.trunci %out : i64 to i32
            memref.store %out_i32, %rd[%i] : memref<?xi32>
            %twenty_nine = arith.constant 29 : i64
            %next_carry = arith.shrui %d, %twenty_nine : i64
            scf.yield %next_carry : i64
          }
          // ge = (r >= |rhs|), scanning limbs from the top with rhs padded to
          // r_capacity limbs.
          %cmp_state = scf.for %i = %c0 to %r_limbs step %c1 iter_args(%state = %zero) -> (i64) {
            %i_next = arith.addi %i, %c1 : index
            %rev = arith.subi %r_limbs, %i_next : index
            %r_digit_i32 = memref.load %rd[%rev] : memref<?xi32>
            %r_digit = arith.extui %r_digit_i32 : i32 to i64
            %rev_i64 = arith.index_cast %rev : index to i64
            %has_rhs = arith.cmpi slt, %rev_i64, %rhs_count : i64
            %rhs_digit = scf.if %has_rhs -> (i64) {
              %d_i32 = memref.load %rhs_digits[%rev] : memref<?xi32>
              %d = arith.extui %d_i32 : i32 to i64
              scf.yield %d : i64
            } else {
              scf.yield %zero : i64
            }
            %still_equal = arith.cmpi eq, %state, %zero : i64
            %gt = arith.cmpi ugt, %r_digit, %rhs_digit : i64
            %lt = arith.cmpi ult, %r_digit, %rhs_digit : i64
            %neg_one = arith.constant -1 : i64
            %c = arith.select %gt, %one, %zero : i1, i64
            %c2 = arith.select %lt, %neg_one, %c : i1, i64
            %next = arith.select %still_equal, %c2, %state : i1, i64
            scf.yield %next : i64
          }
          %r_ge = arith.cmpi sge, %cmp_state, %zero : i64
          scf.if %r_ge {
            // r -= |rhs|; record the quotient bit.
            %borrow_out = scf.for %i = %c0 to %r_limbs step %c1 iter_args(%borrow = %zero) -> (i64) {
              %r_digit_i32 = memref.load %rd[%i] : memref<?xi32>
              %r_digit = arith.extui %r_digit_i32 : i32 to i64
              %i_i64 = arith.index_cast %i : index to i64
              %has_rhs = arith.cmpi slt, %i_i64, %rhs_count : i64
              %rhs_digit = scf.if %has_rhs -> (i64) {
                %d_i32 = memref.load %rhs_digits[%i] : memref<?xi32>
                %d = arith.extui %d_i32 : i32 to i64
                scf.yield %d : i64
              } else {
                scf.yield %zero : i64
              }
              %sub = arith.addi %rhs_digit, %borrow : i64
              %needs_borrow = arith.cmpi ult, %r_digit, %sub : i64
              %raw = arith.subi %r_digit, %sub : i64
              %borrowed = arith.addi %raw, %base : i64
              %val = arith.select %needs_borrow, %borrowed, %raw : i1, i64
              %val_i32 = arith.trunci %val : i64 to i32
              memref.store %val_i32, %rd[%i] : memref<?xi32>
              %next_borrow = arith.select %needs_borrow, %one, %zero : i1, i64
              scf.yield %next_borrow : i64
            }
            %q_slot = arith.index_cast %digit_pos : i64 to index
            %q_old_i32 = memref.load %qd[%q_slot] : memref<?xi32>
            %q_old = arith.extui %q_old_i32 : i32 to i64
            %q_bit = arith.shli %one, %bit_pos : i64
            %q_new = arith.ori %q_old, %q_bit : i64
            %q_new_i32 = arith.trunci %q_new : i64 to i32
            memref.store %q_new_i32, %qd[%q_slot] : memref<?xi32>
          }
        }
        func.call @__ly_long_normalize(%qm, %qd, %q_capacity) : (memref<2xi64>, memref<?xi32>, i64) -> ()
        func.call @__ly_long_normalize(%rm, %rd, %r_capacity) : (memref<2xi64>, memref<?xi32>, i64) -> ()
        scf.yield %qh, %qm, %qd, %rh, %rm, %rd : memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<2xi64>, memref<?xi32>
      }
      scf.yield %inner#0, %inner#1, %inner#2, %inner#3, %inner#4, %inner#5 : memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<2xi64>, memref<?xi32>
    }
    func.return %result#0, %result#3 : memref<2xi64>, memref<2xi64>
  }

  // Floor-divmod on signed views. Owns nothing on entry; returns fresh owned
  // (q, r) satisfying lhs = q * rhs + r with r sharing rhs's sign (CPython).
  // Requires rhs != 0 (callers raise the operator-specific ZeroDivisionError).
  func.func private @__ly_long_floor_divmod(%lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>) attributes {ly.ownership.owned_result_contracts = ["builtins.int", "builtins.int"], ly.ownership.owned_results = [0, 1]} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %neg_one = arith.constant -1 : i64
    %sign_slot = arith.constant 0 : index
    %count_slot = arith.constant 1 : index
    %lhs_sign = memref.load %lhs_meta[%sign_slot] : memref<2xi64>
    %rhs_sign = memref.load %rhs_meta[%sign_slot] : memref<2xi64>
    %q_abs:2 = func.call @__ly_long_divmod_abs(%lhs_meta, %lhs_digits, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>)
    %q_abs_p0_meta, %q_abs_p0_digits = func.call @__ly_long_parts(%q_abs#0) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %q_abs_p3_meta, %q_abs_p3_digits = func.call @__ly_long_parts(%q_abs#1) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %sign_product = arith.muli %lhs_sign, %rhs_sign : i64
    %opposite = arith.cmpi slt, %sign_product, %zero : i64
    %r_count = memref.load %q_abs_p3_meta[%count_slot] : memref<2xi64>
    %r_nonzero = arith.cmpi ne, %r_count, %zero : i64
    %adjust = arith.andi %opposite, %r_nonzero : i1
    // cf blocks (not scf.if): the ownership verifier tracks the conditional
    // consumption of q_abs#3 per path, but not through scf.if region yields.
    cf.cond_br %adjust, ^flip, ^keep

  ^flip:
    // q = -(|q| + 1). divmod_abs left one spare limb, and |rhs| >= 2 here
    // (a remainder forces it), so the in-place increment cannot carry out.
    %q_count = memref.load %q_abs_p0_meta[%count_slot] : memref<2xi64>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %mask = arith.constant 1073741823 : i64
    %count_index = arith.index_cast %q_count : i64 to index
    %carry_final = scf.for %iv = %c0 to %count_index step %c1 iter_args(%carry = %one) -> (i64) {
      %d_i32 = memref.load %q_abs_p0_digits[%iv] : memref<?xi32>
      %d = arith.extui %d_i32 : i32 to i64
      %sum = arith.addi %d, %carry : i64
      %out = arith.andi %sum, %mask : i64
      %out_i32 = arith.trunci %out : i64 to i32
      memref.store %out_i32, %q_abs_p0_digits[%iv] : memref<?xi32>
      %thirty = arith.constant 30 : i64
      %next = arith.shrui %sum, %thirty : i64
      scf.yield %next : i64
    }
    %has_carry = arith.cmpi ne, %carry_final, %zero : i64
    %new_count = scf.if %has_carry -> (i64) {
      %one_i32 = arith.constant 1 : i32
      memref.store %one_i32, %q_abs_p0_digits[%count_index] : memref<?xi32>
      %grown = arith.addi %q_count, %one : i64
      scf.yield %grown : i64
    } else {
      scf.yield %q_count : i64
    }
    memref.store %new_count, %q_abs_p0_meta[%count_slot] : memref<2xi64>
    memref.store %neg_one, %q_abs_p0_meta[%sign_slot] : memref<2xi64>
    // r = sign(rhs) * (|rhs| - |r|).
    %rh = func.call @__ly_long_sub_abs(%rhs_sign, %rhs_meta, %rhs_digits, %q_abs_p3_meta, %q_abs_p3_digits) : (i64, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<?xi32>) -> memref<2xi64>
    func.call @LyLong_DecRef(%q_abs#1) : (memref<2xi64>) -> ()
    func.return %q_abs#0, %rh : memref<2xi64>, memref<2xi64>

  ^keep:
    // Same signs (or exact division): q keeps sign_product's sign, r keeps
    // rhs's sign. Both are fresh divmod_abs allocations, so store in place;
    // a zero magnitude keeps sign 0 from normalize.
    %kq_count = memref.load %q_abs_p0_meta[%count_slot] : memref<2xi64>
    %q_zero = arith.cmpi eq, %kq_count, %zero : i64
    %q_negative = arith.cmpi slt, %sign_product, %zero : i64
    %q_signed = arith.select %q_negative, %neg_one, %one : i1, i64
    %q_sign = arith.select %q_zero, %zero, %q_signed : i1, i64
    memref.store %q_sign, %q_abs_p0_meta[%sign_slot] : memref<2xi64>
    %rhs_negative = arith.cmpi slt, %rhs_sign, %zero : i64
    %r_signed = arith.select %rhs_negative, %neg_one, %one : i1, i64
    %r_sign = arith.select %r_nonzero, %r_signed, %zero : i1, i64
    memref.store %r_sign, %q_abs_p3_meta[%sign_slot] : memref<2xi64>
    func.return %q_abs#0, %q_abs#1 : memref<2xi64>, memref<2xi64>
  }

  func.func @LyLong_Add(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_header: memref<2xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__add__"} {
    %lhs_meta_raw, %lhs_digits_raw = func.call @__ly_long_parts(%lhs_header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %rhs_meta_raw, %rhs_digits_raw = func.call @__ly_long_parts(%rhs_header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %lhs_meta, %lhs_digits = func.call @__ly_long_operand_view(%lhs_meta_raw, %lhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %rhs_meta, %rhs_digits = func.call @__ly_long_operand_view(%rhs_meta_raw, %rhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %small_two = arith.constant 2 : i64
    %small_count_slot = arith.constant 1 : index
    %lhs_view_count = memref.load %lhs_meta[%small_count_slot] : memref<2xi64>
    %rhs_view_count = memref.load %rhs_meta[%small_count_slot] : memref<2xi64>
    %lhs_two_limb = arith.cmpi sle, %lhs_view_count, %small_two : i64
    %rhs_two_limb = arith.cmpi sle, %rhs_view_count, %small_two : i64
    %both_two_limb = arith.andi %lhs_two_limb, %rhs_two_limb : i1
    cf.cond_br %both_two_limb, ^small, ^maybe_i64

  ^maybe_i64:
    %lhs_i64 = func.call @__ly_long_view_fits_i64(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    %rhs_i64 = func.call @__ly_long_view_fits_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    %both_i64 = arith.andi %lhs_i64, %rhs_i64 : i1
    cf.cond_br %both_i64, ^small, ^digits

  ^small:
    // Use primitive arithmetic only while the result also fits signed i64.
    %small_a = func.call @__ly_long_view_as_i64(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %small_b = func.call @__ly_long_view_as_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %small_sum = arith.addi %small_a, %small_b : i64
    %small_zero = arith.constant 0 : i64
    %small_a_negative = arith.cmpi slt, %small_a, %small_zero : i64
    %small_b_negative = arith.cmpi slt, %small_b, %small_zero : i64
    %small_sum_negative = arith.cmpi slt, %small_sum, %small_zero : i64
    %same_input_sign = arith.cmpi eq, %small_a_negative, %small_b_negative : i1
    %result_changed_sign = arith.cmpi ne, %small_sum_negative, %small_a_negative : i1
    %overflow = arith.andi %same_input_sign, %result_changed_sign : i1
    cf.cond_br %overflow, ^digits, ^small_fit

  ^small_fit:
    %sh = func.call @LyLong_FromI64(%small_sum) : (i64) -> memref<2xi64>
    func.return %sh : memref<2xi64>

  ^digits:
    %sign_slot = arith.constant 0 : index
    %rhs_sign = memref.load %rhs_meta[%sign_slot] : memref<2xi64>
    %h = func.call @__ly_long_add_signed_general(%lhs_meta, %lhs_digits, %rhs_sign, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>, i64, memref<2xi64>, memref<?xi32>) -> memref<2xi64>
    func.return %h : memref<2xi64>
  }

  func.func @LyLong_Sub(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_header: memref<2xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__sub__"} {
    %lhs_meta, %lhs_digits = func.call @__ly_long_parts(%lhs_header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %rhs_meta_raw, %rhs_digits_raw = func.call @__ly_long_parts(%rhs_header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %lhs_meta_view, %lhs_digits_view = func.call @__ly_long_operand_view(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %rhs_meta, %rhs_digits = func.call @__ly_long_operand_view(%rhs_meta_raw, %rhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %small_two = arith.constant 2 : i64
    %small_count_slot = arith.constant 1 : index
    %lhs_view_count = memref.load %lhs_meta_view[%small_count_slot] : memref<2xi64>
    %rhs_view_count = memref.load %rhs_meta[%small_count_slot] : memref<2xi64>
    %lhs_two_limb = arith.cmpi sle, %lhs_view_count, %small_two : i64
    %rhs_two_limb = arith.cmpi sle, %rhs_view_count, %small_two : i64
    %both_two_limb = arith.andi %lhs_two_limb, %rhs_two_limb : i1
    cf.cond_br %both_two_limb, ^small, ^maybe_i64

  ^maybe_i64:
    %lhs_i64 = func.call @__ly_long_view_fits_i64(%lhs_meta_view, %lhs_digits_view) : (memref<2xi64>, memref<?xi32>) -> i1
    %rhs_i64 = func.call @__ly_long_view_fits_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    %both_i64 = arith.andi %lhs_i64, %rhs_i64 : i1
    cf.cond_br %both_i64, ^small, ^digits

  ^small:
    %small_a = func.call @__ly_long_view_as_i64(%lhs_meta_view, %lhs_digits_view) : (memref<2xi64>, memref<?xi32>) -> i64
    %small_b = func.call @__ly_long_view_as_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %small_diff = arith.subi %small_a, %small_b : i64
    %small_zero = arith.constant 0 : i64
    %small_a_negative = arith.cmpi slt, %small_a, %small_zero : i64
    %small_b_negative = arith.cmpi slt, %small_b, %small_zero : i64
    %small_diff_negative = arith.cmpi slt, %small_diff, %small_zero : i64
    %different_input_sign = arith.cmpi ne, %small_a_negative, %small_b_negative : i1
    %result_changed_sign = arith.cmpi ne, %small_diff_negative, %small_a_negative : i1
    %overflow = arith.andi %different_input_sign, %result_changed_sign : i1
    cf.cond_br %overflow, ^digits, ^small_fit

  ^small_fit:
    %sh = func.call @LyLong_FromI64(%small_diff) : (i64) -> memref<2xi64>
    func.return %sh : memref<2xi64>

  ^digits:
    %zero = arith.constant 0 : i64
    %sign_slot = arith.constant 0 : index
    %rhs_sign = memref.load %rhs_meta[%sign_slot] : memref<2xi64>
    %rhs_neg = arith.subi %zero, %rhs_sign : i64
    %h = func.call @__ly_long_add_signed_general(%lhs_meta_view, %lhs_digits_view, %rhs_neg, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>, i64, memref<2xi64>, memref<?xi32>) -> memref<2xi64>
    func.return %h : memref<2xi64>
  }

  func.func @LyLong_Mul(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_header: memref<2xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__mul__"} {
    %lhs_meta_raw, %lhs_digits_raw = func.call @__ly_long_parts(%lhs_header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %rhs_meta_raw, %rhs_digits_raw = func.call @__ly_long_parts(%rhs_header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %lhs_meta, %lhs_digits = func.call @__ly_long_operand_view(%lhs_meta_raw, %lhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %rhs_meta, %rhs_digits = func.call @__ly_long_operand_view(%rhs_meta_raw, %rhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %small_two = arith.constant 2 : i64
    %small_count_slot = arith.constant 1 : index
    %lhs_view_count = memref.load %lhs_meta[%small_count_slot] : memref<2xi64>
    %rhs_view_count = memref.load %rhs_meta[%small_count_slot] : memref<2xi64>
    %lhs_two_limb = arith.cmpi sle, %lhs_view_count, %small_two : i64
    %rhs_two_limb = arith.cmpi sle, %rhs_view_count, %small_two : i64
    %both_two_limb = arith.andi %lhs_two_limb, %rhs_two_limb : i1
    cf.cond_br %both_two_limb, ^small, ^maybe_i64

  ^maybe_i64:
    %lhs_i64 = func.call @__ly_long_view_fits_i64(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    %rhs_i64 = func.call @__ly_long_view_fits_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    %both_i64 = arith.andi %lhs_i64, %rhs_i64 : i1
    cf.cond_br %both_i64, ^small, ^digits

  ^small:
    // Use primitive multiplication only when the product remains in i64.
    %small_a = func.call @__ly_long_view_as_i64(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %small_b = func.call @__ly_long_view_as_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %small_low, %small_high = arith.mulsi_extended %small_a, %small_b : i64
    %small_c63 = arith.constant 63 : i64
    %small_sign_ext = arith.shrsi %small_low, %small_c63 : i64
    %small_fits = arith.cmpi eq, %small_high, %small_sign_ext : i64
    cf.cond_br %small_fits, ^small_fit, ^digits

  ^small_fit:
    %sh = func.call @LyLong_FromI64(%small_low) : (i64) -> memref<2xi64>
    func.return %sh : memref<2xi64>

  ^digits:
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %mask = arith.constant 1073741823 : i64
    %shift30 = arith.constant 30 : i64
    %sign_slot = arith.constant 0 : index
    %count_slot = arith.constant 1 : index
    %lhs_sign = memref.load %lhs_meta[%sign_slot] : memref<2xi64>
    %rhs_sign = memref.load %rhs_meta[%sign_slot] : memref<2xi64>
    %lhs_count = memref.load %lhs_meta[%count_slot] : memref<2xi64>
    %rhs_count = memref.load %rhs_meta[%count_slot] : memref<2xi64>
    %lhs_zero = arith.cmpi eq, %lhs_sign, %zero : i64
    %rhs_zero = arith.cmpi eq, %rhs_sign, %zero : i64
    %any_zero = arith.ori %lhs_zero, %rhs_zero : i1
    %result:3 = scf.if %any_zero -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) {
      %h = func.call @LyLong_FromI64(%zero) : (i64) -> memref<2xi64>
      %m, %d = func.call @__ly_long_parts(%h) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
      scf.yield %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
    } else {
      %sign = arith.muli %lhs_sign, %rhs_sign : i64
      %sum_count = arith.addi %lhs_count, %rhs_count : i64
      %capacity = arith.addi %sum_count, %one : i64
      %h = func.call @__ly_long_alloc_raw(%sign, %capacity) : (i64, i64) -> memref<2xi64>
      %m, %d = func.call @__ly_long_parts(%h) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %lhs_index = arith.index_cast %lhs_count : i64 to index
      %rhs_index = arith.index_cast %rhs_count : i64 to index
      %capacity_index = arith.index_cast %capacity : i64 to index
      scf.for %i = %c0 to %lhs_index step %c1 {
        %lhs_digit_i32 = memref.load %lhs_digits[%i] : memref<?xi32>
        %lhs_digit = arith.extui %lhs_digit_i32 : i32 to i64
        %carry = scf.for %j = %c0 to %rhs_index step %c1 iter_args(%carry_iter = %zero) -> (i64) {
          %out_index = arith.addi %i, %j : index
          %rhs_digit_i32 = memref.load %rhs_digits[%j] : memref<?xi32>
          %rhs_digit = arith.extui %rhs_digit_i32 : i32 to i64
          %old_i32 = memref.load %d[%out_index] : memref<?xi32>
          %old = arith.extui %old_i32 : i32 to i64
          %product = arith.muli %lhs_digit, %rhs_digit : i64
          %with_old = arith.addi %product, %old : i64
          %sum = arith.addi %with_old, %carry_iter : i64
          %out_i64 = arith.andi %sum, %mask : i64
          %out = arith.trunci %out_i64 : i64 to i32
          memref.store %out, %d[%out_index] : memref<?xi32>
          %next_carry = arith.shrui %sum, %shift30 : i64
          scf.yield %next_carry : i64
        }
        %tail_index = arith.addi %i, %rhs_index : index
        %tail_old_i32 = memref.load %d[%tail_index] : memref<?xi32>
        %tail_old = arith.extui %tail_old_i32 : i32 to i64
        %tail_sum = arith.addi %tail_old, %carry : i64
        %tail_out_i64 = arith.andi %tail_sum, %mask : i64
        %tail_out = arith.trunci %tail_out_i64 : i64 to i32
        memref.store %tail_out, %d[%tail_index] : memref<?xi32>
        %tail_carry = arith.shrui %tail_sum, %shift30 : i64
        %prop_start = arith.addi %tail_index, %c1 : index
        %ignored = scf.for %k = %prop_start to %capacity_index step %c1 iter_args(%carry_prop = %tail_carry) -> (i64) {
          %has_carry = arith.cmpi ne, %carry_prop, %zero : i64
          %next_carry = scf.if %has_carry -> (i64) {
            %old_i32 = memref.load %d[%k] : memref<?xi32>
            %old = arith.extui %old_i32 : i32 to i64
            %sum = arith.addi %old, %carry_prop : i64
            %out_i64 = arith.andi %sum, %mask : i64
            %out = arith.trunci %out_i64 : i64 to i32
            memref.store %out, %d[%k] : memref<?xi32>
            %carry_next = arith.shrui %sum, %shift30 : i64
            scf.yield %carry_next : i64
          } else {
            scf.yield %zero : i64
          }
          scf.yield %next_carry : i64
        }
      }
      func.call @__ly_long_normalize(%m, %d, %capacity) : (memref<2xi64>, memref<?xi32>, i64) -> ()
      scf.yield %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
    }
    func.return %result#0 : memref<2xi64>
  }

  // |a| << k as a fresh positive magnitude (k >= 0). Sequential carry form of
  // CPython's v_lshift over the 30-bit limbs.
  func.func private @__ly_long_abs_lshift_raw(%meta: memref<2xi64>, %digits: memref<?xi32>, %k: i64) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0]} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %thirty = arith.constant 30 : i64
    %mask = arith.constant 1073741823 : i64
    %count_slot = arith.constant 1 : index
    %count = memref.load %meta[%count_slot] : memref<2xi64>
    %offset = arith.divui %k, %thirty : i64
    %d = arith.remui %k, %thirty : i64
    %inv_d = arith.subi %thirty, %d : i64
    %count_plus = arith.addi %count, %offset : i64
    %capacity = arith.addi %count_plus, %one : i64
    %h = func.call @__ly_long_alloc_raw(%one, %capacity) : (i64, i64) -> memref<2xi64>
    %m, %out = func.call @__ly_long_parts(%h) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %count_index = arith.index_cast %count : i64 to index
    %offset_index = arith.index_cast %offset : i64 to index
    %carry = scf.for %iv = %c0 to %count_index step %c1 iter_args(%carry_iter = %zero) -> (i64) {
      %digit_i32 = memref.load %digits[%iv] : memref<?xi32>
      %digit = arith.extui %digit_i32 : i32 to i64
      %shifted = arith.shli %digit, %d : i64
      %with_carry = arith.ori %shifted, %carry_iter : i64
      %low = arith.andi %with_carry, %mask : i64
      %low_i32 = arith.trunci %low : i64 to i32
      %slot = arith.addi %iv, %offset_index : index
      memref.store %low_i32, %out[%slot] : memref<?xi32>
      // d == 0 shifts a 30-bit digit right by 30, which is zero: no carry.
      %next_carry = arith.shrui %digit, %inv_d : i64
      scf.yield %next_carry : i64
    }
    %carry_slot = arith.index_cast %count_plus : i64 to index
    %carry_i32 = arith.trunci %carry : i64 to i32
    memref.store %carry_i32, %out[%carry_slot] : memref<?xi32>
    func.call @__ly_long_normalize(%m, %out, %capacity) : (memref<2xi64>, memref<?xi32>, i64) -> ()
    func.return %h : memref<2xi64>
  }

  // |a| >> k as a fresh positive magnitude (0 <= k < bit_length(a)), plus a
  // sticky flag: whether any shifted-out bit was nonzero.
  func.func private @__ly_long_abs_rshift_raw(%meta: memref<2xi64>, %digits: memref<?xi32>, %k: i64) -> (memref<2xi64>, i1) attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0]} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %thirty = arith.constant 30 : i64
    %mask = arith.constant 1073741823 : i64
    %false = arith.constant false
    %count_slot = arith.constant 1 : index
    %count = memref.load %meta[%count_slot] : memref<2xi64>
    %offset = arith.divui %k, %thirty : i64
    %d = arith.remui %k, %thirty : i64
    %inv_d = arith.subi %thirty, %d : i64
    %out_count = arith.subi %count, %offset : i64
    %h = func.call @__ly_long_alloc_raw(%one, %out_count) : (i64, i64) -> memref<2xi64>
    %m, %out = func.call @__ly_long_parts(%h) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %out_index = arith.index_cast %out_count : i64 to index
    %offset_index = arith.index_cast %offset : i64 to index
    scf.for %iv = %c0 to %out_index step %c1 {
      %src = arith.addi %iv, %offset_index : index
      %digit_i32 = memref.load %digits[%src] : memref<?xi32>
      %digit = arith.extui %digit_i32 : i32 to i64
      %low = arith.shrui %digit, %d : i64
      %src_next = arith.addi %src, %c1 : index
      %src_next_i64 = arith.index_cast %src_next : index to i64
      %has_next = arith.cmpi slt, %src_next_i64, %count : i64
      %high = scf.if %has_next -> (i64) {
        %next_i32 = memref.load %digits[%src_next] : memref<?xi32>
        %next = arith.extui %next_i32 : i32 to i64
        // d == 0 shifts left by 30 and masks to zero, as required.
        %shifted = arith.shli %next, %inv_d : i64
        %masked = arith.andi %shifted, %mask : i64
        scf.yield %masked : i64
      } else {
        scf.yield %zero : i64
      }
      %combined = arith.ori %low, %high : i64
      %combined_i32 = arith.trunci %combined : i64 to i32
      memref.store %combined_i32, %out[%iv] : memref<?xi32>
    }
    %low_mask_full = arith.shli %one, %d : i64
    %low_mask = arith.subi %low_mask_full, %one : i64
    %boundary_i32 = memref.load %digits[%offset_index] : memref<?xi32>
    %boundary = arith.extui %boundary_i32 : i32 to i64
    %boundary_low = arith.andi %boundary, %low_mask : i64
    %sticky_low = arith.cmpi ne, %boundary_low, %zero : i64
    %sticky = scf.for %iv = %c0 to %offset_index step %c1 iter_args(%seen = %false) -> (i1) {
      %digit_i32 = memref.load %digits[%iv] : memref<?xi32>
      %zero_i32 = arith.constant 0 : i32
      %nonzero = arith.cmpi ne, %digit_i32, %zero_i32 : i32
      %next = arith.ori %seen, %nonzero : i1
      scf.yield %next : i1
    }
    %sticky_any = arith.ori %sticky_low, %sticky : i1
    func.call @__ly_long_normalize(%m, %out, %out_count) : (memref<2xi64>, memref<?xi32>, i64) -> ()
    func.return %h, %sticky_any : memref<2xi64>, i1
  }

  // Correctly-rounded int / int (port of CPython long_true_divide): compute
  // x = floor(|a| * 2^-shift / |b|) with 55..57 significant bits plus an
  // inexactness flag, round x half-to-even at the float precision implied by
  // shift, and scale back by 2^shift. The chosen shift clamps at DBL_MIN_EXP
  // so subnormal results round once, exactly as a double would.
  func.func @LyLong_TrueDiv(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_header: memref<2xi64> {ly.ownership.object_header}) -> memref<3xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__truediv__", ly.runtime.result_contract = "builtins.float"} {
    %lhs_meta_raw, %lhs_digits_raw = func.call @__ly_long_parts(%lhs_header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %rhs_meta_raw, %rhs_digits_raw = func.call @__ly_long_parts(%rhs_header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %lhs_meta, %lhs_digits = func.call @__ly_long_operand_view(%lhs_meta_raw, %lhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %rhs_meta, %rhs_digits = func.call @__ly_long_operand_view(%rhs_meta_raw, %rhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %sign_slot = arith.constant 0 : index
    %lhs_sign = memref.load %lhs_meta[%sign_slot] : memref<2xi64>
    %rhs_sign = memref.load %rhs_meta[%sign_slot] : memref<2xi64>
    %lhs_negative = arith.cmpi slt, %lhs_sign, %zero : i64
    %rhs_negative = arith.cmpi slt, %rhs_sign, %zero : i64
    %negate = arith.cmpi ne, %lhs_negative, %rhs_negative : i1
    %rhs_zero = arith.cmpi eq, %rhs_sign, %zero : i64
    cf.cond_br %rhs_zero, ^zero_divisor, ^check_zero_lhs

  ^zero_divisor:
    func.call @__ly_long_raise_division_by_zero() : () -> ()
    %dummy = arith.constant 0.0 : f64
    %zh = func.call @LyFloat_FromF64(%dummy) : (f64) -> memref<3xi64>
    func.return %zh : memref<3xi64>

  ^check_zero_lhs:
    %lhs_zero = arith.cmpi eq, %lhs_sign, %zero : i64
    cf.cond_br %lhs_zero, ^signed_zero, ^widths

  ^signed_zero:
    // 0 / b keeps the sign of the quotient (CPython returns -0.0 for 0/-b).
    %pos_zero = arith.constant 0.0 : f64
    %neg_zero = arith.negf %pos_zero : f64
    %signed_zero = arith.select %negate, %neg_zero, %pos_zero : i1, f64
    %szh = func.call @LyFloat_FromF64(%signed_zero) : (f64) -> memref<3xi64>
    func.return %szh : memref<3xi64>

  ^widths:
    %a_bits = func.call @__ly_long_bit_length(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %b_bits = func.call @__ly_long_bit_length(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %diff = arith.subi %a_bits, %b_bits : i64
    %max_exp = arith.constant 1024 : i64
    %overflow_early = arith.cmpi sgt, %diff, %max_exp : i64
    cf.cond_br %overflow_early, ^overflow, ^check_underflow

  ^overflow:
    func.call @__ly_long_raise_div_result_too_large() : () -> ()
    %odummy = arith.constant 0.0 : f64
    %oh = func.call @LyFloat_FromF64(%odummy) : (f64) -> memref<3xi64>
    func.return %oh : memref<3xi64>

  ^check_underflow:
    %underflow_limit = arith.constant -1075 : i64
    %underflows = arith.cmpi slt, %diff, %underflow_limit : i64
    cf.cond_br %underflows, ^signed_zero, ^divide

  ^divide:
    // shift = max(diff, DBL_MIN_EXP) - DBL_MANT_DIG - 2 (see the CPython
    // comment for why the DBL_MIN_EXP clamp avoids double rounding).
    %min_exp = arith.constant -1021 : i64
    %diff_ge_min = arith.cmpi sge, %diff, %min_exp : i64
    %clamped = arith.select %diff_ge_min, %diff, %min_exp : i1, i64
    %c55 = arith.constant 55 : i64
    %shift = arith.subi %clamped, %c55 : i64
    %shift_positive = arith.cmpi sgt, %shift, %zero : i64
    %x:4 = scf.if %shift_positive -> (memref<2xi64>, memref<2xi64>, memref<?xi32>, i1) {
      %sh:2 = func.call @__ly_long_abs_rshift_raw(%lhs_meta, %lhs_digits, %shift) : (memref<2xi64>, memref<?xi32>, i64) -> (memref<2xi64>, i1)
      %sh_p0_meta, %sh_p0_digits = func.call @__ly_long_parts(%sh#0) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
      scf.yield %sh, %sh_p0_meta, %sh_p0_digits, %sh#1 : memref<2xi64>, memref<2xi64>, memref<?xi32>, i1
    } else {
      %neg_shift = arith.subi %zero, %shift : i64
      %sh = func.call @__ly_long_abs_lshift_raw(%lhs_meta, %lhs_digits, %neg_shift) : (memref<2xi64>, memref<?xi32>, i64) -> memref<2xi64>
      %sh_p0_meta, %sh_p0_digits = func.call @__ly_long_parts(%sh) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
      %exact = arith.constant false
      scf.yield %sh, %sh_p0_meta, %sh_p0_digits, %exact : memref<2xi64>, memref<2xi64>, memref<?xi32>, i1
    }
    %qr:2 = func.call @__ly_long_divmod_abs(%x#1, %x#2, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>)
    %qr_p0_meta, %qr_p0_digits = func.call @__ly_long_parts(%qr#0) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %qr_p3_meta, %qr_p3_digits = func.call @__ly_long_parts(%qr#1) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    func.call @LyLong_DecRef(%x#0) : (memref<2xi64>) -> ()
    %count_slot = arith.constant 1 : index
    %rem_count = memref.load %qr_p3_meta[%count_slot] : memref<2xi64>
    %rem_nonzero = arith.cmpi ne, %rem_count, %zero : i64
    func.call @LyLong_DecRef(%qr#1) : (memref<2xi64>) -> ()
    %inexact = arith.ori %x#3, %rem_nonzero : i1
    %x_bits = func.call @__ly_long_bit_length(%qr_p0_meta, %qr_p0_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    // val = x as an integer; x has at most 57 bits, so it fits i64 exactly.
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %thirty = arith.constant 30 : i64
    %q_count = memref.load %qr_p0_meta[%count_slot] : memref<2xi64>
    %q_count_index = arith.index_cast %q_count : i64 to index
    %val = scf.for %iv = %c0 to %q_count_index step %c1 iter_args(%acc = %zero) -> (i64) {
      %iv_next = arith.addi %iv, %c1 : index
      %rev = arith.subi %q_count_index, %iv_next : index
      %digit_i32 = memref.load %qr_p0_digits[%rev] : memref<?xi32>
      %digit = arith.extui %digit_i32 : i32 to i64
      %scaled = arith.shli %acc, %thirty : i64
      %next = arith.ori %scaled, %digit : i64
      scf.yield %next : i64
    }
    func.call @LyLong_DecRef(%qr#0) : (memref<2xi64>) -> ()
    // extra_bits = max(x_bits, DBL_MIN_EXP - shift) - DBL_MANT_DIG (2 or 3).
    %c53 = arith.constant 53 : i64
    %min_minus_shift = arith.subi %min_exp, %shift : i64
    %xb_ge = arith.cmpi sge, %x_bits, %min_minus_shift : i64
    %effective = arith.select %xb_ge, %x_bits, %min_minus_shift : i1, i64
    %extra_bits = arith.subi %effective, %c53 : i64
    %extra_minus = arith.subi %extra_bits, %one : i64
    %mask = arith.shli %one, %extra_minus : i64
    // Round half-to-even: bump iff the guard bit is set and any of {sticky
    // bits below it, the inexact flag, the bit above it} is set.
    %inexact_i64 = arith.extui %inexact : i1 to i64
    %low = arith.ori %val, %inexact_i64 : i64
    %two_const = arith.constant 2 : i64
    %neg_one_const = arith.constant -1 : i64
    %triple_scale = arith.constant 3 : i64
    %three_mask = arith.muli %mask, %triple_scale : i64
    %near_mask = arith.subi %three_mask, %one : i64
    %guard = arith.andi %low, %mask : i64
    %guard_set = arith.cmpi ne, %guard, %zero : i64
    %near = arith.andi %low, %near_mask : i64
    %near_set = arith.cmpi ne, %near, %zero : i64
    %round_up = arith.andi %guard_set, %near_set : i1
    %bumped = arith.addi %low, %mask : i64
    %rounded = arith.select %round_up, %bumped, %low : i1, i64
    %clear_full = arith.muli %mask, %two_const : i64
    %clear_mask = arith.subi %clear_full, %one : i64
    %keep_mask = arith.xori %clear_mask, %neg_one_const : i64
    %final_val = arith.andi %rounded, %keep_mask : i64
    %dx = arith.uitofp %final_val : i64 to f64
    // Overflow check before scaling (CPython checks against ldexp(1, x_bits)).
    %shift_plus_bits = arith.addi %shift, %x_bits : i64
    %at_limit = arith.cmpi eq, %shift_plus_bits, %max_exp : i64
    %past_limit = arith.cmpi sgt, %shift_plus_bits, %max_exp : i64
    %pow_xbits = func.call @__ly_long_pow2_f64(%x_bits) : (i64) -> f64
    %dx_carries = arith.cmpf oeq, %dx, %pow_xbits : f64
    %limit_carry = arith.andi %at_limit, %dx_carries : i1
    %overflows = arith.ori %past_limit, %limit_carry : i1
    cf.cond_br %overflows, ^overflow, ^scale

  ^scale:
    // dx * 2^shift; shift can reach -1076, below the smallest normal power,
    // so split the scaling. Both steps are exact: the first keeps a normal
    // result, and the rounded dx * 2^shift is representable by construction.
    %deep = arith.constant -1021 : i64
    %shift_small = arith.cmpi slt, %shift, %deep : i64
    %result = scf.if %shift_small -> (f64) {
      %pre = arith.constant -900 : i64
      %pre_pow = func.call @__ly_long_pow2_f64(%pre) : (i64) -> f64
      %partial = arith.mulf %dx, %pre_pow : f64
      %rest = arith.subi %shift, %pre : i64
      %rest_pow = func.call @__ly_long_pow2_f64(%rest) : (i64) -> f64
      %scaled = arith.mulf %partial, %rest_pow : f64
      scf.yield %scaled : f64
    } else {
      %pow = func.call @__ly_long_pow2_f64(%shift) : (i64) -> f64
      %scaled = arith.mulf %dx, %pow : f64
      scf.yield %scaled : f64
    }
    %negated = arith.negf %result : f64
    %signed = arith.select %negate, %negated, %result : i1, f64
    %h = func.call @LyFloat_FromF64(%signed) : (f64) -> memref<3xi64>
    func.return %h : memref<3xi64>
  }

  func.func @LyLong_FloorDiv(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_header: memref<2xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__floordiv__"} {
    %lhs_meta_raw, %lhs_digits_raw = func.call @__ly_long_parts(%lhs_header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %rhs_meta_raw, %rhs_digits_raw = func.call @__ly_long_parts(%rhs_header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %lhs_meta, %lhs_digits = func.call @__ly_long_operand_view(%lhs_meta_raw, %lhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %rhs_meta, %rhs_digits = func.call @__ly_long_operand_view(%rhs_meta_raw, %rhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %zero = arith.constant 0 : i64
    %sign_slot = arith.constant 0 : index
    %rhs_sign = memref.load %rhs_meta[%sign_slot] : memref<2xi64>
    %rhs_zero = arith.cmpi eq, %rhs_sign, %zero : i64
    cf.cond_br %rhs_zero, ^zero_divisor, ^nonzero

  ^zero_divisor:
    func.call @__ly_long_raise_division_by_zero() : () -> ()
    %zh = func.call @LyLong_FromI64(%zero) : (i64) -> memref<2xi64>
    func.return %zh : memref<2xi64>

  ^nonzero:
    %lhs_fits = func.call @__ly_long_view_fits_i64(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    %rhs_fits = func.call @__ly_long_view_fits_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    %both_fit = arith.andi %lhs_fits, %rhs_fits : i1
    cf.cond_br %both_fit, ^small, ^digits

  ^small:
    %lhs = func.call @__ly_long_view_as_i64(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %rhs = func.call @__ly_long_view_as_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    // divsi INT64_MIN / -1 is poison; that single case promotes to digits.
    %min_i64 = arith.constant -9223372036854775808 : i64
    %neg_one = arith.constant -1 : i64
    %lhs_is_min = arith.cmpi eq, %lhs, %min_i64 : i64
    %rhs_is_neg_one = arith.cmpi eq, %rhs, %neg_one : i64
    %overflows = arith.andi %lhs_is_min, %rhs_is_neg_one : i1
    cf.cond_br %overflows, ^digits, ^small_fit

  ^small_fit:
    %one = arith.constant 1 : i64
    %trunc_q = arith.divsi %lhs, %rhs : i64
    %trunc_r = arith.remsi %lhs, %rhs : i64
    %has_remainder = arith.cmpi ne, %trunc_r, %zero : i64
    %lhs_negative = arith.cmpi slt, %lhs, %zero : i64
    %rhs_negative = arith.cmpi slt, %rhs, %zero : i64
    %different_sign = arith.cmpi ne, %lhs_negative, %rhs_negative : i1
    %adjust = arith.andi %has_remainder, %different_sign : i1
    %decremented = arith.subi %trunc_q, %one : i64
    %floor_q = arith.select %adjust, %decremented, %trunc_q : i1, i64
    %h = func.call @LyLong_FromI64(%floor_q) : (i64) -> memref<2xi64>
    func.return %h : memref<2xi64>

  ^digits:
    %divmod:2 = func.call @__ly_long_floor_divmod(%lhs_meta, %lhs_digits, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>)
    func.call @LyLong_DecRef(%divmod#1) : (memref<2xi64>) -> ()
    func.return %divmod#0 : memref<2xi64>
  }

  func.func @LyLong_Mod(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_header: memref<2xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__mod__"} {
    %lhs_meta_raw, %lhs_digits_raw = func.call @__ly_long_parts(%lhs_header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %rhs_meta_raw, %rhs_digits_raw = func.call @__ly_long_parts(%rhs_header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %lhs_meta, %lhs_digits = func.call @__ly_long_operand_view(%lhs_meta_raw, %lhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %rhs_meta, %rhs_digits = func.call @__ly_long_operand_view(%rhs_meta_raw, %rhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %zero = arith.constant 0 : i64
    %sign_slot = arith.constant 0 : index
    %rhs_sign = memref.load %rhs_meta[%sign_slot] : memref<2xi64>
    %rhs_zero = arith.cmpi eq, %rhs_sign, %zero : i64
    cf.cond_br %rhs_zero, ^zero_divisor, ^nonzero

  ^zero_divisor:
    func.call @__ly_long_raise_division_by_zero() : () -> ()
    %zh = func.call @LyLong_FromI64(%zero) : (i64) -> memref<2xi64>
    func.return %zh : memref<2xi64>

  ^nonzero:
    %lhs_fits = func.call @__ly_long_view_fits_i64(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    %rhs_fits = func.call @__ly_long_view_fits_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    %both_fit = arith.andi %lhs_fits, %rhs_fits : i1
    cf.cond_br %both_fit, ^small, ^digits

  ^small:
    %lhs = func.call @__ly_long_view_as_i64(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %rhs = func.call @__ly_long_view_as_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    // remsi INT64_MIN % -1 is poison alongside its divsi; promote that case.
    %min_i64 = arith.constant -9223372036854775808 : i64
    %neg_one = arith.constant -1 : i64
    %lhs_is_min = arith.cmpi eq, %lhs, %min_i64 : i64
    %rhs_is_neg_one = arith.cmpi eq, %rhs, %neg_one : i64
    %overflows = arith.andi %lhs_is_min, %rhs_is_neg_one : i1
    cf.cond_br %overflows, ^digits, ^small_fit

  ^small_fit:
    %trunc_r = arith.remsi %lhs, %rhs : i64
    %has_remainder = arith.cmpi ne, %trunc_r, %zero : i64
    %remainder_negative = arith.cmpi slt, %trunc_r, %zero : i64
    %rhs_negative = arith.cmpi slt, %rhs, %zero : i64
    %different_sign = arith.cmpi ne, %remainder_negative, %rhs_negative : i1
    %adjust = arith.andi %has_remainder, %different_sign : i1
    %adjusted = arith.addi %trunc_r, %rhs : i64
    %mod = arith.select %adjust, %adjusted, %trunc_r : i1, i64
    %h = func.call @LyLong_FromI64(%mod) : (i64) -> memref<2xi64>
    func.return %h : memref<2xi64>

  ^digits:
    %divmod:2 = func.call @__ly_long_floor_divmod(%lhs_meta, %lhs_digits, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>)
    func.call @LyLong_DecRef(%divmod#0) : (memref<2xi64>) -> ()
    func.return %divmod#1 : memref<2xi64>
  }

  // Infinite two's-complement bitwise op over digit forms (kind: 0 = and,
  // 1 = or, 2 = xor). Negative operands are materialized as two's-complement
  // limb streams over max(n, m) + 1 limbs (the extra limb is pure sign
  // extension, so the result's sign is just the op on the extension limbs),
  // then a negative result is converted back to sign-magnitude. Scratch
  // buffers are plain allocations, not objects, so no ownership is threaded
  // through the loops.
  func.func private @__ly_long_bitop_general(%kind: i64, %lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0]} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %neg_one = arith.constant -1 : i64
    %two = arith.constant 2 : i64
    %thirty = arith.constant 30 : i64
    %mask = arith.constant 1073741823 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %sign_slot = arith.constant 0 : index
    %count_slot = arith.constant 1 : index
    %lhs_sign = memref.load %lhs_meta[%sign_slot] : memref<2xi64>
    %rhs_sign = memref.load %rhs_meta[%sign_slot] : memref<2xi64>
    %lhs_count = memref.load %lhs_meta[%count_slot] : memref<2xi64>
    %rhs_count = memref.load %rhs_meta[%count_slot] : memref<2xi64>
    %lhs_ge = arith.cmpi sge, %lhs_count, %rhs_count : i64
    %max_count = arith.select %lhs_ge, %lhs_count, %rhs_count : i1, i64
    %limbs = arith.addi %max_count, %one : i64
    %limbs_index = arith.index_cast %limbs : i64 to index
    %scratch = memref.alloc(%limbs_index) : memref<?xi32>
    %lhs_negative = arith.cmpi slt, %lhs_sign, %zero : i64
    %rhs_negative = arith.cmpi slt, %rhs_sign, %zero : i64
    %lhs_carry_init = arith.select %lhs_negative, %one, %zero : i1, i64
    %rhs_carry_init = arith.select %rhs_negative, %one, %zero : i1, i64
    %is_and = arith.cmpi eq, %kind, %zero : i64
    %is_or = arith.cmpi eq, %kind, %one : i64
    %ignored:2 = scf.for %iv = %c0 to %limbs_index step %c1 iter_args(%lhs_carry = %lhs_carry_init, %rhs_carry = %rhs_carry_init) -> (i64, i64) {
      %iv_i64 = arith.index_cast %iv : index to i64
      %has_lhs = arith.cmpi slt, %iv_i64, %lhs_count : i64
      %lhs_digit = scf.if %has_lhs -> (i64) {
        %digit_i32 = memref.load %lhs_digits[%iv] : memref<?xi32>
        %digit = arith.extui %digit_i32 : i32 to i64
        scf.yield %digit : i64
      } else {
        scf.yield %zero : i64
      }
      %has_rhs = arith.cmpi slt, %iv_i64, %rhs_count : i64
      %rhs_digit = scf.if %has_rhs -> (i64) {
        %digit_i32 = memref.load %rhs_digits[%iv] : memref<?xi32>
        %digit = arith.extui %digit_i32 : i32 to i64
        scf.yield %digit : i64
      } else {
        scf.yield %zero : i64
      }
      // Two's complement limb: invert (mask ^ d) and add the incoming carry.
      %lhs_inverted = arith.xori %lhs_digit, %mask : i64
      %lhs_tc_raw = arith.addi %lhs_inverted, %lhs_carry : i64
      %lhs_tc = arith.andi %lhs_tc_raw, %mask : i64
      %lhs_carry_next_raw = arith.shrui %lhs_tc_raw, %thirty : i64
      %lhs_a = arith.select %lhs_negative, %lhs_tc, %lhs_digit : i1, i64
      %lhs_carry_next = arith.select %lhs_negative, %lhs_carry_next_raw, %zero : i1, i64
      %rhs_inverted = arith.xori %rhs_digit, %mask : i64
      %rhs_tc_raw = arith.addi %rhs_inverted, %rhs_carry : i64
      %rhs_tc = arith.andi %rhs_tc_raw, %mask : i64
      %rhs_carry_next_raw = arith.shrui %rhs_tc_raw, %thirty : i64
      %rhs_a = arith.select %rhs_negative, %rhs_tc, %rhs_digit : i1, i64
      %rhs_carry_next = arith.select %rhs_negative, %rhs_carry_next_raw, %zero : i1, i64
      %and_limb = arith.andi %lhs_a, %rhs_a : i64
      %or_limb = arith.ori %lhs_a, %rhs_a : i64
      %xor_limb = arith.xori %lhs_a, %rhs_a : i64
      %or_or_xor = arith.select %is_or, %or_limb, %xor_limb : i1, i64
      %result_limb = arith.select %is_and, %and_limb, %or_or_xor : i1, i64
      %result_i32 = arith.trunci %result_limb : i64 to i32
      memref.store %result_i32, %scratch[%iv] : memref<?xi32>
      scf.yield %lhs_carry_next, %rhs_carry_next : i64, i64
    }
    // The top limb is pure sign extension (0 or mask); mask means negative.
    %top_index = arith.subi %limbs_index, %c1 : index
    %top_i32 = memref.load %scratch[%top_index] : memref<?xi32>
    %top = arith.extui %top_i32 : i32 to i64
    %result_negative = arith.cmpi eq, %top, %mask : i64
    %result_sign = arith.select %result_negative, %neg_one, %one : i1, i64
    %h = func.call @__ly_long_alloc_raw(%result_sign, %limbs) : (i64, i64) -> memref<2xi64>
    %m, %d = func.call @__ly_long_parts(%h) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %back_carry_init = arith.select %result_negative, %one, %zero : i1, i64
    %ignored2 = scf.for %iv = %c0 to %limbs_index step %c1 iter_args(%carry = %back_carry_init) -> (i64) {
      %limb_i32 = memref.load %scratch[%iv] : memref<?xi32>
      %limb = arith.extui %limb_i32 : i32 to i64
      %inverted = arith.xori %limb, %mask : i64
      %tc_raw = arith.addi %inverted, %carry : i64
      %tc = arith.andi %tc_raw, %mask : i64
      %carry_next_raw = arith.shrui %tc_raw, %thirty : i64
      %magnitude = arith.select %result_negative, %tc, %limb : i1, i64
      %carry_next = arith.select %result_negative, %carry_next_raw, %zero : i1, i64
      %magnitude_i32 = arith.trunci %magnitude : i64 to i32
      memref.store %magnitude_i32, %d[%iv] : memref<?xi32>
      scf.yield %carry_next : i64
    }
    memref.dealloc %scratch : memref<?xi32>
    func.call @__ly_long_normalize(%m, %d, %limbs) : (memref<2xi64>, memref<?xi32>, i64) -> ()
    func.return %h : memref<2xi64>
  }

  func.func @LyLong_BitAnd(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_header: memref<2xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__and__"} {
    %lhs_meta_raw, %lhs_digits_raw = func.call @__ly_long_parts(%lhs_header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %rhs_meta_raw, %rhs_digits_raw = func.call @__ly_long_parts(%rhs_header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %lhs_meta, %lhs_digits = func.call @__ly_long_operand_view(%lhs_meta_raw, %lhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %rhs_meta, %rhs_digits = func.call @__ly_long_operand_view(%rhs_meta_raw, %rhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %lhs_fits = func.call @__ly_long_view_fits_i64(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    %rhs_fits = func.call @__ly_long_view_fits_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    %both_fit = arith.andi %lhs_fits, %rhs_fits : i1
    cf.cond_br %both_fit, ^small, ^digits

  ^small:
    %lhs = func.call @__ly_long_view_as_i64(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %rhs = func.call @__ly_long_view_as_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %value = arith.andi %lhs, %rhs : i64
    %h = func.call @LyLong_FromI64(%value) : (i64) -> memref<2xi64>
    func.return %h : memref<2xi64>

  ^digits:
    %kind = arith.constant 0 : i64
    %gh = func.call @__ly_long_bitop_general(%kind, %lhs_meta, %lhs_digits, %rhs_meta, %rhs_digits) : (i64, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<?xi32>) -> memref<2xi64>
    func.return %gh : memref<2xi64>
  }

  func.func @LyLong_BitOr(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_header: memref<2xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__or__"} {
    %lhs_meta_raw, %lhs_digits_raw = func.call @__ly_long_parts(%lhs_header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %rhs_meta_raw, %rhs_digits_raw = func.call @__ly_long_parts(%rhs_header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %lhs_meta, %lhs_digits = func.call @__ly_long_operand_view(%lhs_meta_raw, %lhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %rhs_meta, %rhs_digits = func.call @__ly_long_operand_view(%rhs_meta_raw, %rhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %lhs_fits = func.call @__ly_long_view_fits_i64(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    %rhs_fits = func.call @__ly_long_view_fits_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    %both_fit = arith.andi %lhs_fits, %rhs_fits : i1
    cf.cond_br %both_fit, ^small, ^digits

  ^small:
    %lhs = func.call @__ly_long_view_as_i64(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %rhs = func.call @__ly_long_view_as_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %value = arith.ori %lhs, %rhs : i64
    %h = func.call @LyLong_FromI64(%value) : (i64) -> memref<2xi64>
    func.return %h : memref<2xi64>

  ^digits:
    %kind = arith.constant 1 : i64
    %gh = func.call @__ly_long_bitop_general(%kind, %lhs_meta, %lhs_digits, %rhs_meta, %rhs_digits) : (i64, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<?xi32>) -> memref<2xi64>
    func.return %gh : memref<2xi64>
  }

  func.func @LyLong_BitXor(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_header: memref<2xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__xor__"} {
    %lhs_meta_raw, %lhs_digits_raw = func.call @__ly_long_parts(%lhs_header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %rhs_meta_raw, %rhs_digits_raw = func.call @__ly_long_parts(%rhs_header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %lhs_meta, %lhs_digits = func.call @__ly_long_operand_view(%lhs_meta_raw, %lhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %rhs_meta, %rhs_digits = func.call @__ly_long_operand_view(%rhs_meta_raw, %rhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %lhs_fits = func.call @__ly_long_view_fits_i64(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    %rhs_fits = func.call @__ly_long_view_fits_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    %both_fit = arith.andi %lhs_fits, %rhs_fits : i1
    cf.cond_br %both_fit, ^small, ^digits

  ^small:
    %lhs = func.call @__ly_long_view_as_i64(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %rhs = func.call @__ly_long_view_as_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %value = arith.xori %lhs, %rhs : i64
    %h = func.call @LyLong_FromI64(%value) : (i64) -> memref<2xi64>
    func.return %h : memref<2xi64>

  ^digits:
    %kind = arith.constant 2 : i64
    %gh = func.call @__ly_long_bitop_general(%kind, %lhs_meta, %lhs_digits, %rhs_meta, %rhs_digits) : (i64, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<?xi32>) -> memref<2xi64>
    func.return %gh : memref<2xi64>
  }

  func.func @LyLong_LShift(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_header: memref<2xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__lshift__"} {
    %lhs_meta_raw, %lhs_digits_raw = func.call @__ly_long_parts(%lhs_header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %rhs_meta_raw, %rhs_digits_raw = func.call @__ly_long_parts(%rhs_header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %lhs_meta, %lhs_digits = func.call @__ly_long_operand_view(%lhs_meta_raw, %lhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %rhs_meta, %rhs_digits = func.call @__ly_long_operand_view(%rhs_meta_raw, %rhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %thirty = arith.constant 30 : i64
    %mask = arith.constant 1073741823 : i64
    %sign_slot = arith.constant 0 : index
    %count_slot = arith.constant 1 : index
    %rhs_sign = memref.load %rhs_meta[%sign_slot] : memref<2xi64>
    %negative_count = arith.cmpi slt, %rhs_sign, %zero : i64
    cf.cond_br %negative_count, ^negative, ^check_width

  ^negative:
    func.call @__ly_long_raise_negative_shift() : () -> ()
    %gh = func.call @LyLong_FromI64(%zero) : (i64) -> memref<2xi64>
    func.return %gh : memref<2xi64>

  ^check_width:
    %lhs_sign = memref.load %lhs_meta[%sign_slot] : memref<2xi64>
    %count_fits = func.call @__ly_long_view_fits_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    cf.cond_br %count_fits, ^shift, ^huge_count

  ^huge_count:
    // 0 << anything is 0; any other base would exceed memory.
    %lhs_is_zero = arith.cmpi eq, %lhs_sign, %zero : i64
    cf.cond_br %lhs_is_zero, ^zero_base, ^too_large

  ^zero_base:
    %zh = func.call @LyLong_FromI64(%zero) : (i64) -> memref<2xi64>
    func.return %zh : memref<2xi64>

  ^too_large:
    func.call @__ly_long_raise_too_many_digits() : () -> ()
    %th = func.call @LyLong_FromI64(%zero) : (i64) -> memref<2xi64>
    func.return %th : memref<2xi64>

  ^shift:
    %n = func.call @__ly_long_view_as_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %lhs_fits = func.call @__ly_long_view_fits_i64(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    %sixty_two = arith.constant 62 : i64
    %small_count = arith.cmpi sle, %n, %sixty_two : i64
    %try_small = arith.andi %lhs_fits, %small_count : i1
    cf.cond_br %try_small, ^small, ^digits

  ^small:
    %value = func.call @__ly_long_view_as_i64(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %shifted = arith.shli %value, %n : i64
    %round_trip = arith.shrsi %shifted, %n : i64
    %exact = arith.cmpi eq, %round_trip, %value : i64
    cf.cond_br %exact, ^small_fit, ^digits

  ^small_fit:
    %sh = func.call @LyLong_FromI64(%shifted) : (i64) -> memref<2xi64>
    func.return %sh : memref<2xi64>

  ^digits:
    %lhs_zero_mag = arith.cmpi eq, %lhs_sign, %zero : i64
    cf.cond_br %lhs_zero_mag, ^zero_base, ^digits_shift

  ^digits_shift:
    %lhs_count = memref.load %lhs_meta[%count_slot] : memref<2xi64>
    %dig_shift = arith.divui %n, %thirty : i64
    %bit_shift = arith.remui %n, %thirty : i64
    %grow = arith.addi %lhs_count, %dig_shift : i64
    // A digit for the bits shifted out of the top only when there are some,
    // CPython's newsize: the count is what MAX_LONG_DIGITS is checked against.
    %has_bits = arith.cmpi ne, %bit_shift, %zero : i64
    %carry_digit = arith.extui %has_bits : i1 to i64
    %capacity = arith.addi %grow, %carry_digit : i64
    // CPython 3.14's MAX_LONG_DIGITS, PY_SSIZE_T_MAX / PyLong_SHIFT: the most
    // digits whose bit count is still a Py_ssize_t; under it, a result past
    // the allocator's reach is MemoryError.
    %max_digits = arith.constant 307445734561825860 : i64
    %too_many_digits = arith.cmpi sgt, %capacity, %max_digits : i64
    scf.if %too_many_digits {
      func.call @__ly_long_raise_too_many_digits() : () -> ()
    }
    %digit_bytes_i64 = arith.constant 4 : i64
    %digit_prefix = arith.constant 32 : i64
    func.call @__ly_check_alloc_count(%capacity, %digit_bytes_i64, %digit_prefix) : (i64, i64, i64) -> ()
    %h = func.call @__ly_long_alloc_raw(%lhs_sign, %capacity) : (i64, i64) -> memref<2xi64>
    %m, %d = func.call @__ly_long_parts(%h) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %lhs_index = arith.index_cast %lhs_count : i64 to index
    %dig_shift_index = arith.index_cast %dig_shift : i64 to index
    %down = arith.subi %thirty, %bit_shift : i64
    %carry_out = scf.for %iv = %c0 to %lhs_index step %c1 iter_args(%carry = %zero) -> (i64) {
      %digit_i32 = memref.load %lhs_digits[%iv] : memref<?xi32>
      %digit = arith.extui %digit_i32 : i32 to i64
      %up = arith.shli %digit, %bit_shift : i64
      %with_carry = arith.ori %up, %carry : i64
      %out = arith.andi %with_carry, %mask : i64
      %out_i32 = arith.trunci %out : i64 to i32
      %slot = arith.addi %iv, %dig_shift_index : index
      memref.store %out_i32, %d[%slot] : memref<?xi32>
      // down == 30 when bit_shift == 0; a 30-bit digit shifted right by 30
      // is 0, so the carry degenerates correctly.
      %next_carry = arith.shrui %digit, %down : i64
      scf.yield %next_carry : i64
    }
    %tail = arith.addi %lhs_index, %dig_shift_index : index
    %carry_i32 = arith.trunci %carry_out : i64 to i32
    scf.if %has_bits {
      memref.store %carry_i32, %d[%tail] : memref<?xi32>
    }
    func.call @__ly_long_normalize(%m, %d, %capacity) : (memref<2xi64>, memref<?xi32>, i64) -> ()
    func.return %h : memref<2xi64>
  }

  func.func @LyLong_RShift(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_header: memref<2xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__rshift__"} {
    %lhs_meta_raw, %lhs_digits_raw = func.call @__ly_long_parts(%lhs_header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %rhs_meta_raw, %rhs_digits_raw = func.call @__ly_long_parts(%rhs_header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %lhs_meta, %lhs_digits = func.call @__ly_long_operand_view(%lhs_meta_raw, %lhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %rhs_meta, %rhs_digits = func.call @__ly_long_operand_view(%rhs_meta_raw, %rhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %neg_one = arith.constant -1 : i64
    %thirty = arith.constant 30 : i64
    %mask = arith.constant 1073741823 : i64
    %sign_slot = arith.constant 0 : index
    %count_slot = arith.constant 1 : index
    %rhs_sign = memref.load %rhs_meta[%sign_slot] : memref<2xi64>
    %negative_count = arith.cmpi slt, %rhs_sign, %zero : i64
    cf.cond_br %negative_count, ^negative, ^check_width

  ^negative:
    func.call @__ly_long_raise_negative_shift() : () -> ()
    %gh = func.call @LyLong_FromI64(%zero) : (i64) -> memref<2xi64>
    func.return %gh : memref<2xi64>

  ^check_width:
    %lhs_sign = memref.load %lhs_meta[%sign_slot] : memref<2xi64>
    %count_fits = func.call @__ly_long_view_fits_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    cf.cond_br %count_fits, ^shift, ^saturated

  ^saturated:
    // Shifting everything out: 0 for non-negative, -1 for negative (floor).
    %lhs_negative_s = arith.cmpi slt, %lhs_sign, %zero : i64
    %sat = arith.select %lhs_negative_s, %neg_one, %zero : i1, i64
    %vh = func.call @LyLong_FromI64(%sat) : (i64) -> memref<2xi64>
    func.return %vh : memref<2xi64>

  ^shift:
    %n = func.call @__ly_long_view_as_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %lhs_fits = func.call @__ly_long_view_fits_i64(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    cf.cond_br %lhs_fits, ^small, ^digits

  ^small:
    %value = func.call @__ly_long_view_as_i64(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    // shrsi is poison for counts >= 64; arithmetic shift saturates at 63.
    %sixty_three = arith.constant 63 : i64
    %clamp = arith.minsi %n, %sixty_three : i64
    %shifted = arith.shrsi %value, %clamp : i64
    %sh = func.call @LyLong_FromI64(%shifted) : (i64) -> memref<2xi64>
    func.return %sh : memref<2xi64>

  ^digits:
    %lhs_count = memref.load %lhs_meta[%count_slot] : memref<2xi64>
    %dig_shift = arith.divui %n, %thirty : i64
    %bit_shift = arith.remui %n, %thirty : i64
    %all_out = arith.cmpi sge, %dig_shift, %lhs_count : i64
    cf.cond_br %all_out, ^saturated, ^digits_shift

  ^digits_shift:
    %new_count = arith.subi %lhs_count, %dig_shift : i64
    // One spare limb: the floor adjustment for negative values increments
    // the magnitude in place.
    %capacity = arith.addi %new_count, %one : i64
    %h = func.call @__ly_long_alloc_raw(%lhs_sign, %capacity) : (i64, i64) -> memref<2xi64>
    %m, %d = func.call @__ly_long_parts(%h) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %new_index = arith.index_cast %new_count : i64 to index
    %dig_shift_index = arith.index_cast %dig_shift : i64 to index
    %lhs_index = arith.index_cast %lhs_count : i64 to index
    %up = arith.subi %thirty, %bit_shift : i64
    scf.for %iv = %c0 to %new_index step %c1 {
      %src = arith.addi %iv, %dig_shift_index : index
      %digit_i32 = memref.load %lhs_digits[%src] : memref<?xi32>
      %digit = arith.extui %digit_i32 : i32 to i64
      %low = arith.shrui %digit, %bit_shift : i64
      %src_next = arith.addi %src, %c1 : index
      %has_next = arith.cmpi slt, %src_next, %lhs_index : index
      %high = scf.if %has_next -> (i64) {
        %next_i32 = memref.load %lhs_digits[%src_next] : memref<?xi32>
        %next = arith.extui %next_i32 : i32 to i64
        %shifted_up = arith.shli %next, %up : i64
        scf.yield %shifted_up : i64
      } else {
        scf.yield %zero : i64
      }
      %combined = arith.ori %low, %high : i64
      %out = arith.andi %combined, %mask : i64
      %out_i32 = arith.trunci %out : i64 to i32
      memref.store %out_i32, %d[%iv] : memref<?xi32>
    }
    %lhs_negative = arith.cmpi slt, %lhs_sign, %zero : i64
    scf.if %lhs_negative {
      // floor(-x / 2^n) = -((x >> n) + 1) when any bit was shifted out.
      %dropped_digits = scf.for %iv = %c0 to %dig_shift_index step %c1 iter_args(%sticky = %zero) -> (i64) {
        %digit_i32 = memref.load %lhs_digits[%iv] : memref<?xi32>
        %digit = arith.extui %digit_i32 : i32 to i64
        %merged = arith.ori %sticky, %digit : i64
        scf.yield %merged : i64
      }
      %boundary_i32 = memref.load %lhs_digits[%dig_shift_index] : memref<?xi32>
      %boundary = arith.extui %boundary_i32 : i32 to i64
      %low_mask_full = arith.shli %one, %bit_shift : i64
      %low_mask = arith.subi %low_mask_full, %one : i64
      %boundary_dropped = arith.andi %boundary, %low_mask : i64
      %all_dropped = arith.ori %dropped_digits, %boundary_dropped : i64
      %has_dropped = arith.cmpi ne, %all_dropped, %zero : i64
      scf.if %has_dropped {
        %carry_final = scf.for %iv = %c0 to %new_index step %c1 iter_args(%carry = %one) -> (i64) {
          %digit_i32 = memref.load %d[%iv] : memref<?xi32>
          %digit = arith.extui %digit_i32 : i32 to i64
          %sum = arith.addi %digit, %carry : i64
          %out = arith.andi %sum, %mask : i64
          %out_i32 = arith.trunci %out : i64 to i32
          memref.store %out_i32, %d[%iv] : memref<?xi32>
          %next = arith.shrui %sum, %thirty : i64
          scf.yield %next : i64
        }
        %overflowed = arith.cmpi ne, %carry_final, %zero : i64
        scf.if %overflowed {
          %one_i32 = arith.constant 1 : i32
          memref.store %one_i32, %d[%new_index] : memref<?xi32>
        }
      }
    }
    func.call @__ly_long_normalize(%m, %d, %capacity) : (memref<2xi64>, memref<?xi32>, i64) -> ()
    func.return %h : memref<2xi64>
  }

  // round(int, ndigits): identity for ndigits >= 0; otherwise round to the
  // nearest multiple of 10^-ndigits with ties to even (CPython int.__round__,
  // exact at any width: 10^n via square-and-multiply, then one divmod).
  func.func @LyLong_Round(%header: memref<2xi64> {ly.ownership.object_header}, %ndigits: i64 {ly.runtime.default_i64 = 0 : i64}) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__round__"} {
    %meta_raw, %digits_raw = func.call @__ly_long_parts(%header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %meta, %digits = func.call @__ly_long_operand_view(%meta_raw, %digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %nonneg = arith.cmpi sge, %ndigits, %zero : i64
    cf.cond_br %nonneg, ^identity, ^negative

  ^identity:
    %ih = func.call @__ly_long_copy(%meta, %digits) : (memref<2xi64>, memref<?xi32>) -> memref<2xi64>
    func.return %ih : memref<2xi64>

  ^negative:
    %places = arith.subi %zero, %ndigits : i64
    // -INT64_MIN wraps negative; such a scale dwarfs any representable int.
    %wrapped = arith.cmpi sle, %places, %zero : i64
    cf.cond_br %wrapped, ^zero_result, ^check_width

  ^zero_result:
    %zh = func.call @LyLong_FromI64(%zero) : (i64) -> memref<2xi64>
    func.return %zh : memref<2xi64>

  ^check_width:
    // 10^places >= 2^(3*places) > 2*|x| makes the result 0 (a tie needs
    // 10^places == 2*|x| exactly, which the general path handles). Checking
    // places >= bit_length first keeps 3*places from overflowing and bounds
    // the 10^places allocation by the width of x itself.
    %bit_len = func.call @__ly_long_bit_length(%meta, %digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %input_zero = arith.cmpi eq, %bit_len, %zero : i64
    cf.cond_br %input_zero, ^zero_result, ^check_places

  ^check_places:
    %huge_places = arith.cmpi sge, %places, %bit_len : i64
    cf.cond_br %huge_places, ^zero_result, ^check_three

  ^check_three:
    %digit_bits_scale = arith.constant 3 : i64
    %two = arith.constant 2 : i64
    %three_places = arith.muli %places, %digit_bits_scale : i64
    %bit_len_plus = arith.addi %bit_len, %two : i64
    %dominates = arith.cmpi sge, %three_places, %bit_len_plus : i64
    cf.cond_br %dominates, ^zero_result, ^general

  ^general:
    %ten = arith.constant 10 : i64
    %ten_h = func.call @LyLong_FromI64(%ten) : (i64) -> memref<2xi64>
    %scale = func.call @__ly_long_pow_rec(%ten_h, %places) : (memref<2xi64>, i64) -> memref<2xi64>
    %scale_p0_meta, %scale_p0_digits = func.call @__ly_long_parts(%scale) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    func.call @LyLong_DecRef(%ten_h) : (memref<2xi64>) -> ()
    %qr:2 = func.call @__ly_long_divmod_abs(%meta, %digits, %scale_p0_meta, %scale_p0_digits) : (memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>)
    %qr_p0_meta, %qr_p0_digits = func.call @__ly_long_parts(%qr#0) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %qr_p3_meta, %qr_p3_digits = func.call @__ly_long_parts(%qr#1) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %two_r = func.call @__ly_long_add_abs(%one, %qr_p3_meta, %qr_p3_digits, %qr_p3_meta, %qr_p3_digits) : (i64, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<?xi32>) -> memref<2xi64>
    %two_r_p0_meta, %two_r_p0_digits = func.call @__ly_long_parts(%two_r) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    func.call @LyLong_DecRef(%qr#1) : (memref<2xi64>) -> ()
    %cmp = func.call @__ly_long_abs_compare(%two_r_p0_meta, %two_r_p0_digits, %scale_p0_meta, %scale_p0_digits) : (memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<?xi32>) -> i64
    func.call @LyLong_DecRef(%two_r) : (memref<2xi64>) -> ()
    %above_half = arith.cmpi sgt, %cmp, %zero : i64
    %exactly_half = arith.cmpi eq, %cmp, %zero : i64
    %q_digit0_slot = arith.constant 0 : index
    %q_digit0_i32 = memref.load %qr_p0_digits[%q_digit0_slot] : memref<?xi32>
    %q_digit0 = arith.extui %q_digit0_i32 : i32 to i64
    %q_low_bit = arith.andi %q_digit0, %one : i64
    %q_odd = arith.cmpi ne, %q_low_bit, %zero : i64
    %tie_up = arith.andi %exactly_half, %q_odd : i1
    %round_up = arith.ori %above_half, %tie_up : i1
    cf.cond_br %round_up, ^bump, ^scale_back

  ^bump:
    %one_meta = memref.get_global @__ly_long_one_meta : memref<2xi64>
    %one_digits_static = memref.get_global @__ly_long_one_digits : memref<1xi32>
    %one_digits = memref.cast %one_digits_static : memref<1xi32> to memref<?xi32>
    %bumped = func.call @__ly_long_add_abs(%one, %qr_p0_meta, %qr_p0_digits, %one_meta, %one_digits) : (i64, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<?xi32>) -> memref<2xi64>
    func.call @LyLong_DecRef(%qr#0) : (memref<2xi64>) -> ()
    %bm = func.call @LyLong_Mul(%bumped, %scale) : (memref<2xi64>, memref<2xi64>) -> memref<2xi64>
    %bm_p0_meta, %bm_p0_digits = func.call @__ly_long_parts(%bm) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    func.call @LyLong_DecRef(%bumped) : (memref<2xi64>) -> ()
    func.call @LyLong_DecRef(%scale) : (memref<2xi64>) -> ()
    // Reapply the input's sign on a copy; the product of nonzero magnitudes
    // is nonzero.
    // ⛔ Not stored into the product: a small one is the shared immortal
    // object (`__ly_long_small_ints`), and `round(-15, -1)` made every 20 in
    // the program -20.
    %b_sign_slot = arith.constant 0 : index
    %b_input_sign = memref.load %meta[%b_sign_slot] : memref<2xi64>
    %br = func.call @__ly_long_copy_with_sign(%b_input_sign, %bm_p0_meta, %bm_p0_digits) : (i64, memref<2xi64>, memref<?xi32>) -> memref<2xi64>
    func.call @LyLong_DecRef(%bm) : (memref<2xi64>) -> ()
    func.return %br : memref<2xi64>

  ^scale_back:
    // A zero quotient short-circuits: the product would be the immortal zero
    // cache, whose meta must never be written.
    %q_count_slot = arith.constant 1 : index
    %q_count = memref.load %qr_p0_meta[%q_count_slot] : memref<2xi64>
    %q_zero = arith.cmpi eq, %q_count, %zero : i64
    cf.cond_br %q_zero, ^scale_back_zero, ^scale_back_mul

  ^scale_back_zero:
    func.call @LyLong_DecRef(%qr#0) : (memref<2xi64>) -> ()
    func.call @LyLong_DecRef(%scale) : (memref<2xi64>) -> ()
    %qz_h = func.call @LyLong_FromI64(%zero) : (i64) -> memref<2xi64>
    func.return %qz_h : memref<2xi64>

  ^scale_back_mul:
    %sm = func.call @LyLong_Mul(%qr#0, %scale) : (memref<2xi64>, memref<2xi64>) -> memref<2xi64>
    %sm_p0_meta, %sm_p0_digits = func.call @__ly_long_parts(%sm) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    func.call @LyLong_DecRef(%qr#0) : (memref<2xi64>) -> ()
    func.call @LyLong_DecRef(%scale) : (memref<2xi64>) -> ()
    // The product of nonzero magnitudes is nonzero: reapply the input's sign,
    // on a copy for the reason above.
    %s_sign_slot = arith.constant 0 : index
    %s_input_sign = memref.load %meta[%s_sign_slot] : memref<2xi64>
    %sr = func.call @__ly_long_copy_with_sign(%s_input_sign, %sm_p0_meta, %sm_p0_digits) : (i64, memref<2xi64>, memref<?xi32>) -> memref<2xi64>
    func.call @LyLong_DecRef(%sm) : (memref<2xi64>) -> ()
    func.return %sr : memref<2xi64>
  }

  // Base-10 int(str) parse: optional surrounding ASCII whitespace, optional
  // sign, digits with single interior underscores. Arbitrary length via
  // in-place multiply-by-10-and-add over the digit limbs. Runtime-level
  // __int__ on str (not part of the typed manifest surface: CPython has no
  // str.__int__; only the emitter's int(x) rewrite targets it). Unicode
  // digits are not accepted yet (ASCII only until the UCD tables land).
  func.func private @__ly_long_from_ascii(%bytes: memref<?xi8>) -> (memref<2xi64>, i1) attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0]} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %neg_one = arith.constant -1 : i64
    %ten = arith.constant 10 : i64
    %mask = arith.constant 1073741823 : i64
    %thirty = arith.constant 30 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %len = memref.dim %bytes, %c0 : memref<?xi8>
    %len_i64 = arith.index_cast %len : index to i64
    // First and last non-whitespace positions (space, \t..\r).
    %space = arith.constant 32 : i8
    %tab = arith.constant 9 : i8
    %cr = arith.constant 13 : i8
    %first = scf.for %iv = %c0 to %len step %c1 iter_args(%found = %neg_one) -> (i64) {
      %c = memref.load %bytes[%iv] : memref<?xi8>
      %is_space = arith.cmpi eq, %c, %space : i8
      %ge_tab = arith.cmpi sge, %c, %tab : i8
      %le_cr = arith.cmpi sle, %c, %cr : i8
      %is_ctl = arith.andi %ge_tab, %le_cr : i1
      %is_ws = arith.ori %is_space, %is_ctl : i1
      %unset = arith.cmpi eq, %found, %neg_one : i64
      %iv_i64 = arith.index_cast %iv : index to i64
      // found records the first non-ws index once, then sticks.
      %candidate = arith.select %is_ws, %found, %iv_i64 : i1, i64
      %next = arith.select %unset, %candidate, %found : i1, i64
      scf.yield %next : i64
    }
    %last = scf.for %iv = %c0 to %len step %c1 iter_args(%found = %neg_one) -> (i64) {
      %iv_next = arith.addi %iv, %c1 : index
      %rev = arith.subi %len, %iv_next : index
      %c = memref.load %bytes[%rev] : memref<?xi8>
      %is_space = arith.cmpi eq, %c, %space : i8
      %ge_tab = arith.cmpi sge, %c, %tab : i8
      %le_cr = arith.cmpi sle, %c, %cr : i8
      %is_ctl = arith.andi %ge_tab, %le_cr : i1
      %is_ws = arith.ori %is_space, %is_ctl : i1
      %unset = arith.cmpi eq, %found, %neg_one : i64
      %rev_i64 = arith.index_cast %rev : index to i64
      %candidate = arith.select %is_ws, %found, %rev_i64 : i1, i64
      %next = arith.select %unset, %candidate, %found : i1, i64
      scf.yield %next : i64
    }
    %no_content = arith.cmpi eq, %first, %neg_one : i64
    cf.cond_br %no_content, ^invalid_early, ^signed

  ^invalid_early:
    %false_bit = arith.constant false
    %eh = func.call @LyLong_FromI64(%zero) : (i64) -> memref<2xi64>
    func.return %eh, %false_bit : memref<2xi64>, i1

  ^signed:
    %end = arith.addi %last, %one : i64
    %first_char_index = arith.index_cast %first : i64 to index
    %first_char = memref.load %bytes[%first_char_index] : memref<?xi8>
    %plus = arith.constant 43 : i8
    %minus = arith.constant 45 : i8
    %is_plus = arith.cmpi eq, %first_char, %plus : i8
    %is_minus = arith.cmpi eq, %first_char, %minus : i8
    %has_sign = arith.ori %is_plus, %is_minus : i1
    %after_sign = arith.addi %first, %one : i64
    %digits_start = arith.select %has_sign, %after_sign, %first : i1, i64
    %sign = arith.select %is_minus, %neg_one, %one : i1, i64
    %no_digits = arith.cmpi sge, %digits_start, %end : i64
    cf.cond_br %no_digits, ^invalid_early, ^parse

  ^parse:
    // Capacity: 4 bits per char comfortably over-approximates log2(10).
    %nchars = arith.subi %end, %digits_start : i64
    %four = arith.constant 4 : i64
    %bits = arith.muli %nchars, %four : i64
    %limbs = arith.divui %bits, %thirty : i64
    %two = arith.constant 2 : i64
    %capacity = arith.addi %limbs, %two : i64
    %h = func.call @__ly_long_alloc_raw(%sign, %capacity) : (i64, i64) -> memref<2xi64>
    %m, %d = func.call @__ly_long_parts(%h) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %start_index = arith.index_cast %digits_start : i64 to index
    %end_index = arith.index_cast %end : i64 to index
    %ascii_zero = arith.constant 48 : i8
    %ascii_nine = arith.constant 57 : i8
    %underscore = arith.constant 95 : i8
    %parse:3 = scf.for %iv = %start_index to %end_index step %c1 iter_args(%count = %one, %invalid = %zero, %prev_us = %one) -> (i64, i64, i64) {
      // prev_us starts as 1 so a leading underscore is rejected.
      %c = memref.load %bytes[%iv] : memref<?xi8>
      %is_us = arith.cmpi eq, %c, %underscore : i8
      %ge_zero = arith.cmpi sge, %c, %ascii_zero : i8
      %le_nine = arith.cmpi sle, %c, %ascii_nine : i8
      %is_digit = arith.andi %ge_zero, %le_nine : i1
      %prev_was_us = arith.cmpi ne, %prev_us, %zero : i64
      %bad_us = arith.andi %is_us, %prev_was_us : i1
      %not_token = arith.ori %is_us, %is_digit : i1
      %true_bit = arith.constant true
      %bad_char = arith.xori %not_token, %true_bit : i1
      %new_invalid_flag = arith.ori %bad_us, %bad_char : i1
      %new_invalid_i64 = arith.extui %new_invalid_flag : i1 to i64
      %invalid_next = arith.ori %invalid, %new_invalid_i64 : i64
      %digit_val_i8 = arith.subi %c, %ascii_zero : i8
      %digit_val_raw = arith.extui %digit_val_i8 : i8 to i64
      %digit_val = arith.select %is_digit, %digit_val_raw, %zero : i1, i64
      %count_next = scf.if %is_digit -> (i64) {
        // x = x * 10 + digit over the active limbs; carry-in is the new digit.
        %count_index = arith.index_cast %count : i64 to index
        %carry_out = scf.for %i = %c0 to %count_index step %c1 iter_args(%carry = %digit_val) -> (i64) {
          %limb_i32 = memref.load %d[%i] : memref<?xi32>
          %limb = arith.extui %limb_i32 : i32 to i64
          %scaled = arith.muli %limb, %ten : i64
          %sum = arith.addi %scaled, %carry : i64
          %out = arith.andi %sum, %mask : i64
          %out_i32 = arith.trunci %out : i64 to i32
          memref.store %out_i32, %d[%i] : memref<?xi32>
          %next_carry = arith.shrui %sum, %thirty : i64
          scf.yield %next_carry : i64
        }
        %has_carry = arith.cmpi ne, %carry_out, %zero : i64
        %grown = scf.if %has_carry -> (i64) {
          %slot = arith.index_cast %count : i64 to index
          %carry_i32 = arith.trunci %carry_out : i64 to i32
          memref.store %carry_i32, %d[%slot] : memref<?xi32>
          %count_grown = arith.addi %count, %one : i64
          scf.yield %count_grown : i64
        } else {
          scf.yield %count : i64
        }
        scf.yield %grown : i64
      } else {
        scf.yield %count : i64
      }
      %prev_us_next = arith.extui %is_us : i1 to i64
      scf.yield %count_next, %invalid_next, %prev_us_next : i64, i64, i64
    }
    %trailing_us = arith.cmpi ne, %parse#2, %zero : i64
    %char_invalid = arith.cmpi ne, %parse#1, %zero : i64
    %any_invalid = arith.ori %trailing_us, %char_invalid : i1
    cf.cond_br %any_invalid, ^invalid_parsed, ^done

  ^invalid_parsed:
    %parsed_false = arith.constant false
    func.call @LyLong_DecRef(%h) : (memref<2xi64>) -> ()
    %ph = func.call @LyLong_FromI64(%zero) : (i64) -> memref<2xi64>
    func.return %ph, %parsed_false : memref<2xi64>, i1

  ^done:
    %done_true = arith.constant true
    func.call @__ly_long_normalize(%m, %d, %capacity) : (memref<2xi64>, memref<?xi32>, i64) -> ()
    func.return %h, %done_true : memref<2xi64>, i1
  }

  func.func @LyLong_FromStr(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__int__", ly.runtime.result_contract = "builtins.int"} {
    %zero = arith.constant 0 : i64
    %parsed:2 = func.call @__ly_long_from_ascii(%bytes) : (memref<?xi8>) -> (memref<2xi64>, i1)
    cf.cond_br %parsed#1, ^ok, ^invalid

  ^ok:
    func.return %parsed#0 : memref<2xi64>

  ^invalid:
    // The failed parse still returns an owned zero; releasing it BEFORE the
    // raise is what keeps its release reachable (the raise does not return).
    func.call @LyLong_DecRef(%parsed#0) : (memref<2xi64>) -> ()
    func.call @__ly_long_raise_invalid_literal(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> ()
    %uh = func.call @LyLong_FromI64(%zero) : (i64) -> memref<2xi64>
    func.return %uh : memref<2xi64>
  }

  // Square-and-multiply as recursion (depth <= 63): each frame creates,
  // consumes, and returns owned values linearly, which the affine-ownership
  // verifier can follow; a loop would have to thread owned iter_args, which
  // it cannot.
  func.func private @__ly_long_pow_rec(%base_header: memref<2xi64>, %exp: i64) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0]} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %is_zero = arith.cmpi eq, %exp, %zero : i64
    cf.cond_br %is_zero, ^base_case, ^recurse

  ^base_case:
    %oh = func.call @LyLong_FromI64(%one) : (i64) -> memref<2xi64>
    func.return %oh : memref<2xi64>

  ^recurse:
    %half_exp = arith.shrui %exp, %one : i64
    %half = func.call @__ly_long_pow_rec(%base_header, %half_exp) : (memref<2xi64>, i64) -> memref<2xi64>
    %square = func.call @LyLong_Mul(%half, %half) : (memref<2xi64>, memref<2xi64>) -> memref<2xi64>
    func.call @LyLong_DecRef(%half) : (memref<2xi64>) -> ()
    %low_bit = arith.andi %exp, %one : i64
    %is_odd = arith.cmpi ne, %low_bit, %zero : i64
    cf.cond_br %is_odd, ^odd, ^even

  ^odd:
    %with_base = func.call @LyLong_Mul(%square, %base_header) : (memref<2xi64>, memref<2xi64>) -> memref<2xi64>
    func.call @LyLong_DecRef(%square) : (memref<2xi64>) -> ()
    func.return %with_base : memref<2xi64>

  ^even:
    func.return %square : memref<2xi64>
  }

  func.func @LyLong_Pow(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_header: memref<2xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__pow__"} {
    %lhs_meta_raw, %lhs_digits_raw = func.call @__ly_long_parts(%lhs_header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %rhs_meta_raw, %rhs_digits_raw = func.call @__ly_long_parts(%rhs_header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %lhs_meta, %lhs_digits = func.call @__ly_long_operand_view(%lhs_meta_raw, %lhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %rhs_meta, %rhs_digits = func.call @__ly_long_operand_view(%rhs_meta_raw, %rhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %sign_slot = arith.constant 0 : index
    %rhs_sign = memref.load %rhs_meta[%sign_slot] : memref<2xi64>
    %negative_exp = arith.cmpi slt, %rhs_sign, %zero : i64
    cf.cond_br %negative_exp, ^negative, ^check_width

  ^negative:
    // CPython returns a float here; the static result type is int, so reject
    // loudly instead of changing the result type (deviation, see message).
    func.call @__ly_long_raise_pow_negative() : () -> ()
    %gh = func.call @LyLong_FromI64(%zero) : (i64) -> memref<2xi64>
    func.return %gh : memref<2xi64>

  ^check_width:
    %exp_fits = func.call @__ly_long_view_fits_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    cf.cond_br %exp_fits, ^pow, ^huge_exp

  ^huge_exp:
    // Bases with |base| > 1 would not fit in memory; 0/1/-1 with a 2^63+
    // exponent are legal but degenerate. -1 keeps the parity of the lowest
    // exponent limb.
    %count_slot = arith.constant 1 : index
    %lhs_sign_h = memref.load %lhs_meta[%sign_slot] : memref<2xi64>
    %lhs_count_h = memref.load %lhs_meta[%count_slot] : memref<2xi64>
    %is_zero_base = arith.cmpi eq, %lhs_count_h, %zero : i64
    %is_one_limb = arith.cmpi eq, %lhs_count_h, %one : i64
    %c0g = arith.constant 0 : index
    %digit0_g = memref.load %lhs_digits[%c0g] : memref<?xi32>
    %digit0_g_i64 = arith.extui %digit0_g : i32 to i64
    %magnitude_le_one = arith.cmpi ule, %digit0_g_i64, %one : i64
    %is_unit_base = arith.andi %is_one_limb, %magnitude_le_one : i1
    %is_small_base = arith.ori %is_zero_base, %is_unit_base : i1
    cf.cond_br %is_small_base, ^huge_exp_small_base, ^huge_exp_reject

  ^huge_exp_reject:
    func.call @__ly_long_raise_too_large() : () -> ()
    %rh = func.call @LyLong_FromI64(%zero) : (i64) -> memref<2xi64>
    func.return %rh : memref<2xi64>

  ^huge_exp_small_base:
    %c0h = arith.constant 0 : index
    %digit0_h = memref.load %rhs_digits[%c0h] : memref<?xi32>
    %digit0_h_i64 = arith.extui %digit0_h : i32 to i64
    %exp_odd_h = arith.andi %digit0_h_i64, %one : i64
    %exp_is_odd = arith.cmpi ne, %exp_odd_h, %zero : i64
    %neg_base = arith.cmpi slt, %lhs_sign_h, %zero : i64
    %flip = arith.andi %neg_base, %exp_is_odd : i1
    %neg_one_h = arith.constant -1 : i64
    %abs_result = arith.cmpi eq, %lhs_sign_h, %zero : i64
    %zero_or_sign = arith.select %flip, %neg_one_h, %one : i1, i64
    %small_value = arith.select %abs_result, %zero, %zero_or_sign : i1, i64
    %sh = func.call @LyLong_FromI64(%small_value) : (i64) -> memref<2xi64>
    func.return %sh : memref<2xi64>

  ^pow:
    %exp = func.call @__ly_long_view_as_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %h = func.call @__ly_long_pow_rec(%lhs_header, %exp) : (memref<2xi64>, i64) -> memref<2xi64>
    func.return %h : memref<2xi64>
  }

  // CPython-compatible int hash: reduction of the magnitude modulo the
  // Mersenne prime 2^61 - 1 (digit-wise 30-bit rotation), sign applied,
  // -1 remapped to -2. Keeping the modulus scheme means float can later
  // satisfy hash(1) == hash(1.0). Int hashing is not randomized in CPython
  // either (SipHash applies to str/bytes).
  func.func @LyLong_Hash(%header: memref<2xi64> {ly.ownership.object_header}) -> i64 attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__hash__"} {
    %meta_raw, %digits_raw = func.call @__ly_long_parts(%header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %meta, %digits = func.call @__ly_long_operand_view(%meta_raw, %digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %zero = arith.constant 0 : i64
    %thirty = arith.constant 30 : i64
    %thirty_one = arith.constant 31 : i64
    %modulus = arith.constant 2305843009213693951 : i64
    %sign_slot = arith.constant 0 : index
    %count_slot = arith.constant 1 : index
    %sign = memref.load %meta[%sign_slot] : memref<2xi64>
    %count = memref.load %meta[%count_slot] : memref<2xi64>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %count_index = arith.index_cast %count : i64 to index
    %reduced = scf.for %iv = %c0 to %count_index step %c1 iter_args(%x = %zero) -> (i64) {
      %iv_next = arith.addi %iv, %c1 : index
      %rev = arith.subi %count_index, %iv_next : index
      %digit_i32 = memref.load %digits[%rev] : memref<?xi32>
      %digit = arith.extui %digit_i32 : i32 to i64
      // 61-bit rotate left by 30: keep low bits shifted up (mod trick), pull
      // the high 31 bits down.
      %up = arith.shli %x, %thirty : i64
      %up_masked = arith.andi %up, %modulus : i64
      %down = arith.shrui %x, %thirty_one : i64
      %rotated = arith.ori %up_masked, %down : i64
      %with_digit = arith.addi %rotated, %digit : i64
      %needs_reduce = arith.cmpi uge, %with_digit, %modulus : i64
      %reduced_once = arith.subi %with_digit, %modulus : i64
      %next = arith.select %needs_reduce, %reduced_once, %with_digit : i1, i64
      scf.yield %next : i64
    }
    %negated = arith.subi %zero, %reduced : i64
    %is_negative = arith.cmpi slt, %sign, %zero : i64
    %signed = arith.select %is_negative, %negated, %reduced : i1, i64
    %neg_one = arith.constant -1 : i64
    %neg_two = arith.constant -2 : i64
    %is_neg_one = arith.cmpi eq, %signed, %neg_one : i64
    %result = arith.select %is_neg_one, %neg_two, %signed : i1, i64
    func.return %result : i64
  }

  func.func @LyLong_Compare(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_header: memref<2xi64> {ly.ownership.object_header}) -> i64 attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__richcompare__"} {
    %lhs_meta_raw, %lhs_digits_raw = func.call @__ly_long_parts(%lhs_header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %rhs_meta_raw, %rhs_digits_raw = func.call @__ly_long_parts(%rhs_header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %lhs_meta, %lhs_digits = func.call @__ly_long_operand_view(%lhs_meta_raw, %lhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %rhs_meta, %rhs_digits = func.call @__ly_long_operand_view(%rhs_meta_raw, %rhs_digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %small_two = arith.constant 2 : i64
    %small_count_slot = arith.constant 1 : index
    %lhs_view_count = memref.load %lhs_meta[%small_count_slot] : memref<2xi64>
    %rhs_view_count = memref.load %rhs_meta[%small_count_slot] : memref<2xi64>
    %lhs_two_limb = arith.cmpi sle, %lhs_view_count, %small_two : i64
    %rhs_two_limb = arith.cmpi sle, %rhs_view_count, %small_two : i64
    %both_two_limb = arith.andi %lhs_two_limb, %rhs_two_limb : i1
    cf.cond_br %both_two_limb, ^small, ^maybe_i64

  ^maybe_i64:
    %lhs_i64 = func.call @__ly_long_view_fits_i64(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    %rhs_i64 = func.call @__ly_long_view_fits_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    %both_i64 = arith.andi %lhs_i64, %rhs_i64 : i1
    cf.cond_br %both_i64, ^small, ^digits

  ^small:
    // Signed i64 is sufficient for this comparison.
    %small_a = func.call @__ly_long_view_as_i64(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %small_b = func.call @__ly_long_view_as_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %small_zero = arith.constant 0 : i64
    %small_one = arith.constant 1 : i64
    %small_neg_one = arith.constant -1 : i64
    %small_gt = arith.cmpi sgt, %small_a, %small_b : i64
    %small_lt = arith.cmpi slt, %small_a, %small_b : i64
    %small_pos = arith.select %small_gt, %small_one, %small_zero : i1, i64
    %small_cmp = arith.select %small_lt, %small_neg_one, %small_pos : i1, i64
    func.return %small_cmp : i64

  ^digits:
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %neg_one = arith.constant -1 : i64
    %sign_slot = arith.constant 0 : index
    %lhs_sign = memref.load %lhs_meta[%sign_slot] : memref<2xi64>
    %rhs_sign = memref.load %rhs_meta[%sign_slot] : memref<2xi64>
    %sign_gt = arith.cmpi sgt, %lhs_sign, %rhs_sign : i64
    %sign_lt = arith.cmpi slt, %lhs_sign, %rhs_sign : i64
    %sign_cmp = arith.select %sign_gt, %one, %zero : i1, i64
    %sign_cmp2 = arith.select %sign_lt, %neg_one, %sign_cmp : i1, i64
    %same_sign = arith.cmpi eq, %sign_cmp2, %zero : i64
    %result = scf.if %same_sign -> (i64) {
      %lhs_zero = arith.cmpi eq, %lhs_sign, %zero : i64
      %same_sign_cmp = scf.if %lhs_zero -> (i64) {
        scf.yield %zero : i64
      } else {
        %abs_cmp = func.call @__ly_long_abs_compare(%lhs_meta, %lhs_digits, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<?xi32>) -> i64
        %is_negative = arith.cmpi slt, %lhs_sign, %zero : i64
        %neg_abs = arith.muli %abs_cmp, %neg_one : i64
        %signed_cmp = arith.select %is_negative, %neg_abs, %abs_cmp : i1, i64
        scf.yield %signed_cmp : i64
      }
      scf.yield %same_sign_cmp : i64
    } else {
      scf.yield %sign_cmp2 : i64
    }
    func.return %result : i64
  }

  func.func @LyLong_EqBool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_header: memref<2xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__eq__"} {
    %cmp = func.call @LyLong_Compare(%lhs_header, %rhs_header) : (memref<2xi64>, memref<2xi64>) -> i64
    %zero = arith.constant 0 : i64
    %result = arith.cmpi eq, %cmp, %zero : i64
    func.return %result : i1
  }

  func.func @LyLong_NeBool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_header: memref<2xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__ne__"} {
    %cmp = func.call @LyLong_Compare(%lhs_header, %rhs_header) : (memref<2xi64>, memref<2xi64>) -> i64
    %zero = arith.constant 0 : i64
    %result = arith.cmpi ne, %cmp, %zero : i64
    func.return %result : i1
  }

  func.func @LyLong_LtBool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_header: memref<2xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__lt__"} {
    %cmp = func.call @LyLong_Compare(%lhs_header, %rhs_header) : (memref<2xi64>, memref<2xi64>) -> i64
    %zero = arith.constant 0 : i64
    %result = arith.cmpi slt, %cmp, %zero : i64
    func.return %result : i1
  }

  func.func @LyLong_LeBool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_header: memref<2xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__le__"} {
    %cmp = func.call @LyLong_Compare(%lhs_header, %rhs_header) : (memref<2xi64>, memref<2xi64>) -> i64
    %zero = arith.constant 0 : i64
    %result = arith.cmpi sle, %cmp, %zero : i64
    func.return %result : i1
  }

  func.func @LyLong_GtBool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_header: memref<2xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__gt__"} {
    %cmp = func.call @LyLong_Compare(%lhs_header, %rhs_header) : (memref<2xi64>, memref<2xi64>) -> i64
    %zero = arith.constant 0 : i64
    %result = arith.cmpi sgt, %cmp, %zero : i64
    func.return %result : i1
  }

  func.func @LyLong_GeBool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_header: memref<2xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__ge__"} {
    %cmp = func.call @LyLong_Compare(%lhs_header, %rhs_header) : (memref<2xi64>, memref<2xi64>) -> i64
    %zero = arith.constant 0 : i64
    %result = arith.cmpi sge, %cmp, %zero : i64
    func.return %result : i1
  }

  // Exact int-vs-double ordering (CPython float_richcompare): -1/0/1 when
  // the int orders below/equal/above the double, 2 for unordered (NaN).
  // Converting the int to double first would collapse a 1-ulp neighborhood
  // (2**53 + 1 == 2.0**53 under that scheme), so the comparison stays in
  // integer space: an i64-ranged int splits the double into trunc + fraction
  // (both exact below 2**63); a wider int against |d| < 2**63 is decided by
  // sign alone; and |d| >= 2**63 is integer-valued, so it converts EXACTLY
  // through the int(float) mantissa-shift path and compares digit-wise.
  func.func private @__ly_long_cmp_f64(%meta_raw: memref<2xi64>, %digits_raw: memref<?xi32>, %d: f64) -> i64 {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %neg_one = arith.constant -1 : i64
    %two = arith.constant 2 : i64
    %meta, %digits = func.call @__ly_long_operand_view(%meta_raw, %digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %is_nan = arith.cmpf uno, %d, %d : f64
    cf.cond_br %is_nan, ^unordered, ^check_inf

  ^unordered:
    func.return %two : i64

  ^check_inf:
    %bits = arith.bitcast %d : f64 to i64
    %exp_shift = arith.constant 52 : i64
    %exp_mask = arith.constant 2047 : i64
    %exp_raw_shifted = arith.shrui %bits, %exp_shift : i64
    %exp_raw = arith.andi %exp_raw_shifted, %exp_mask : i64
    %is_inf = arith.cmpi eq, %exp_raw, %exp_mask : i64
    cf.cond_br %is_inf, ^infinity, ^classify

  ^infinity:
    %d_negative = arith.cmpi slt, %bits, %zero : i64
    %inf_cmp = arith.select %d_negative, %one, %neg_one : i1, i64
    func.return %inf_cmp : i64

  ^classify:
    %upper = arith.constant 9.2233720368547758E+18 : f64
    %lower = arith.constant -9.2233720368547758E+18 : f64
    %fits = func.call @__ly_long_view_fits_i64(%meta, %digits) : (memref<2xi64>, memref<?xi32>) -> i1
    cf.cond_br %fits, ^small, ^big

  ^small:
    %v = func.call @__ly_long_view_as_i64(%meta, %digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %d_ge_upper = arith.cmpf oge, %d, %upper : f64
    cf.cond_br %d_ge_upper, ^small_below, ^small_check_lower

  ^small_below:
    func.return %neg_one : i64

  ^small_check_lower:
    %d_lt_lower = arith.cmpf olt, %d, %lower : f64
    cf.cond_br %d_lt_lower, ^small_above, ^small_check_min

  ^small_above:
    func.return %one : i64

  ^small_check_min:
    // d == -2**63 exactly: fptosi would overflow, but the answer is direct.
    %d_eq_lower = arith.cmpf oeq, %d, %lower : f64
    cf.cond_br %d_eq_lower, ^small_min, ^small_split

  ^small_min:
    %int64_min = arith.constant -9223372036854775808 : i64
    %v_is_min = arith.cmpi eq, %v, %int64_min : i64
    %min_cmp = arith.select %v_is_min, %zero, %one : i1, i64
    func.return %min_cmp : i64

  ^small_split:
    // |d| < 2**63: trunc and fraction are both exact doubles (any double of
    // magnitude >= 2**52 is already an integer; below that trunc fits the
    // mantissa).
    %w = arith.fptosi %d : f64 to i64
    %v_lt_w = arith.cmpi slt, %v, %w : i64
    %v_gt_w = arith.cmpi sgt, %v, %w : i64
    %wf = arith.sitofp %w : i64 to f64
    %frac = arith.subf %d, %wf : f64
    %fzero = arith.constant 0.0 : f64
    %frac_pos = arith.cmpf ogt, %frac, %fzero : f64
    %frac_neg = arith.cmpf olt, %frac, %fzero : f64
    %tie = arith.select %frac_pos, %neg_one, %zero : i1, i64
    %tie2 = arith.select %frac_neg, %one, %tie : i1, i64
    %after_gt = arith.select %v_gt_w, %one, %tie2 : i1, i64
    %small_cmp = arith.select %v_lt_w, %neg_one, %after_gt : i1, i64
    func.return %small_cmp : i64

  ^big:
    // |v| >= 2**63 here, so any |d| < 2**63 (including d == -2**63 exactly,
    // which a NEGATIVE v is strictly below) is decided by v's sign.
    %sign_slot = arith.constant 0 : index
    %v_sign = memref.load %meta[%sign_slot] : memref<2xi64>
    %v_negative = arith.cmpi slt, %v_sign, %zero : i64
    %d_lt_upper = arith.cmpf olt, %d, %upper : f64
    %d_ge_lower = arith.cmpf oge, %d, %lower : f64
    %d_small = arith.andi %d_lt_upper, %d_ge_lower : i1
    cf.cond_br %d_small, ^big_sign, ^big_both

  ^big_sign:
    %sign_cmp = arith.select %v_negative, %neg_one, %one : i1, i64
    func.return %sign_cmp : i64

  ^big_both:
    // |d| >= 2**63 is integer-valued: int(d) is exact, compare digit-wise.
    %fh = func.call @LyFloat_FromF64(%d) : (f64) -> memref<3xi64>
    %th = func.call @LyFloat_Int(%fh) : (memref<3xi64>) -> memref<2xi64>
    %tm, %td = func.call @__ly_long_parts(%th) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %t_sign = memref.load %tm[%sign_slot] : memref<2xi64>
    %sign_gt = arith.cmpi sgt, %v_sign, %t_sign : i64
    %sign_lt = arith.cmpi slt, %v_sign, %t_sign : i64
    %abs_cmp = func.call @__ly_long_abs_compare(%meta, %digits, %tm, %td) : (memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<?xi32>) -> i64
    %neg_abs = arith.muli %abs_cmp, %neg_one : i64
    %signed_abs = arith.select %v_negative, %neg_abs, %abs_cmp : i1, i64
    %after_sgt = arith.select %sign_gt, %one, %signed_abs : i1, i64
    %big_cmp = arith.select %sign_lt, %neg_one, %after_sgt : i1, i64
    func.call @LyFloat_DecRef(%fh) : (memref<3xi64>) -> ()
    func.call @LyLong_DecRef(%th) : (memref<2xi64>) -> ()
    func.return %big_cmp : i64
  }

  func.func @LyLong_LtF64Bool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_header: memref<3xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__lt__"} {
    %lhs_meta, %lhs_digits = func.call @__ly_long_parts(%lhs_header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %d = func.call @LyFloat_AsF64(%rhs_header) : (memref<3xi64>) -> f64
    %cmp = func.call @__ly_long_cmp_f64(%lhs_meta, %lhs_digits, %d) : (memref<2xi64>, memref<?xi32>, f64) -> i64
    %neg_one = arith.constant -1 : i64
    %result = arith.cmpi eq, %cmp, %neg_one : i64
    func.return %result : i1
  }

  func.func @LyLong_LeF64Bool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_header: memref<3xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__le__"} {
    %lhs_meta, %lhs_digits = func.call @__ly_long_parts(%lhs_header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %d = func.call @LyFloat_AsF64(%rhs_header) : (memref<3xi64>) -> f64
    %cmp = func.call @__ly_long_cmp_f64(%lhs_meta, %lhs_digits, %d) : (memref<2xi64>, memref<?xi32>, f64) -> i64
    %zero = arith.constant 0 : i64
    %result = arith.cmpi sle, %cmp, %zero : i64
    func.return %result : i1
  }

  func.func @LyLong_GtF64Bool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_header: memref<3xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__gt__"} {
    %lhs_meta, %lhs_digits = func.call @__ly_long_parts(%lhs_header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %d = func.call @LyFloat_AsF64(%rhs_header) : (memref<3xi64>) -> f64
    %cmp = func.call @__ly_long_cmp_f64(%lhs_meta, %lhs_digits, %d) : (memref<2xi64>, memref<?xi32>, f64) -> i64
    %one = arith.constant 1 : i64
    %result = arith.cmpi eq, %cmp, %one : i64
    func.return %result : i1
  }

  func.func @LyLong_GeF64Bool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_header: memref<3xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__ge__"} {
    %lhs_meta, %lhs_digits = func.call @__ly_long_parts(%lhs_header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %d = func.call @LyFloat_AsF64(%rhs_header) : (memref<3xi64>) -> f64
    %cmp = func.call @__ly_long_cmp_f64(%lhs_meta, %lhs_digits, %d) : (memref<2xi64>, memref<?xi32>, f64) -> i64
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %is_ge = arith.cmpi sge, %cmp, %zero : i64
    %is_ordered = arith.cmpi sle, %cmp, %one : i64
    %result = arith.andi %is_ge, %is_ordered : i1
    func.return %result : i1
  }

  func.func @LyLong_EqF64Bool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_header: memref<3xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__eq__"} {
    %lhs_meta, %lhs_digits = func.call @__ly_long_parts(%lhs_header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %d = func.call @LyFloat_AsF64(%rhs_header) : (memref<3xi64>) -> f64
    %cmp = func.call @__ly_long_cmp_f64(%lhs_meta, %lhs_digits, %d) : (memref<2xi64>, memref<?xi32>, f64) -> i64
    %zero = arith.constant 0 : i64
    %result = arith.cmpi eq, %cmp, %zero : i64
    func.return %result : i1
  }

  func.func @LyLong_NeF64Bool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_header: memref<3xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__ne__"} {
    %lhs_meta, %lhs_digits = func.call @__ly_long_parts(%lhs_header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %d = func.call @LyFloat_AsF64(%rhs_header) : (memref<3xi64>) -> f64
    %cmp = func.call @__ly_long_cmp_f64(%lhs_meta, %lhs_digits, %d) : (memref<2xi64>, memref<?xi32>, f64) -> i64
    %zero = arith.constant 0 : i64
    %result = arith.cmpi ne, %cmp, %zero : i64
    func.return %result : i1
  }

  // str(int) == repr(int) in CPython; delegate.
  func.func @LyLong_Str(%header: memref<2xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %str_header, %str_bytes = func.call @LyLong_Repr(%header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi8>)
    func.return %str_header, %str_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyLong_Repr(%header: memref<2xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %meta_raw, %digits_raw = func.call @__ly_long_parts(%header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %meta, %digits = func.call @__ly_long_operand_view(%meta_raw, %digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %ten = arith.constant 10 : i64
    %base = arith.constant 1073741824 : i64
    %ascii_zero = arith.constant 48 : i64
    %ascii_minus = arith.constant 45 : i8
    %sign_slot = arith.constant 0 : index
    %count_slot = arith.constant 1 : index
    %sign = memref.load %meta[%sign_slot] : memref<2xi64>
    %count = memref.load %meta[%count_slot] : memref<2xi64>
    %is_zero = arith.cmpi eq, %count, %zero : i64
    %result:2 = scf.if %is_zero -> (memref<2xi64>, memref<?xi8>) {
      %h, %b = func.call @LyUnicode_FromI64(%zero) : (i64) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %h, %b : memref<2xi64>, memref<?xi8>
    } else {
      %count_index = arith.index_cast %count : i64 to index
      %tmp = memref.alloc(%count_index) : memref<?xi32>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %iv = %c0 to %count_index step %c1 {
        %digit = memref.load %digits[%iv] : memref<?xi32>
        memref.store %digit, %tmp[%iv] : memref<?xi32>
      }
      %negative = arith.cmpi slt, %sign, %zero : i64
      %sign_extra = arith.select %negative, %one, %zero : i1, i64
      %decimal_capacity_base = arith.muli %count, %ten : i64
      %decimal_capacity = arith.addi %decimal_capacity_base, %sign_extra : i64
      %decimal_capacity_index = arith.index_cast %decimal_capacity : i64 to index
      %buffer = memref.alloc(%decimal_capacity_index) : memref<?xi8>
      %conversion:2 = scf.for %step_iv = %c0 to %decimal_capacity_index step %c1 iter_args(%active_count = %count, %pos = %decimal_capacity_index) -> (i64, index) {
        %active = arith.cmpi ne, %active_count, %zero : i64
        %next:2 = scf.if %active -> (i64, index) {
          %active_index = arith.index_cast %active_count : i64 to index
          %division:2 = scf.for %scan = %c0 to %active_index step %c1 iter_args(%rem_iter = %zero, %last_iter = %zero) -> (i64, i64) {
            %scan_next = arith.addi %scan, %c1 : index
            %rev = arith.subi %active_index, %scan_next : index
            %digit_i32 = memref.load %tmp[%rev] : memref<?xi32>
            %digit = arith.extui %digit_i32 : i32 to i64
            %scaled = arith.muli %rem_iter, %base : i64
            %accum = arith.addi %scaled, %digit : i64
            %quotient = arith.divui %accum, %ten : i64
            %remainder = arith.remui %accum, %ten : i64
            %quotient_i32 = arith.trunci %quotient : i64 to i32
            memref.store %quotient_i32, %tmp[%rev] : memref<?xi32>
            %nonzero = arith.cmpi ne, %quotient, %zero : i64
            %no_last = arith.cmpi eq, %last_iter, %zero : i64
            %take_last = arith.andi %nonzero, %no_last : i1
            %rev_next = arith.addi %rev, %c1 : index
            %rev_count = arith.index_cast %rev_next : index to i64
            %last = arith.select %take_last, %rev_count, %last_iter : i1, i64
            scf.yield %remainder, %last : i64, i64
          }
          %ascii_digit_i64 = arith.addi %division#0, %ascii_zero : i64
          %ascii_digit = arith.trunci %ascii_digit_i64 : i64 to i8
          %next_pos = arith.subi %pos, %c1 : index
          memref.store %ascii_digit, %buffer[%next_pos] : memref<?xi8>
          scf.yield %division#1, %next_pos : i64, index
        } else {
          scf.yield %active_count, %pos : i64, index
        }
        scf.yield %next#0, %next#1 : i64, index
      }
      %start = scf.if %negative -> (index) {
        %minus_pos = arith.subi %conversion#1, %c1 : index
        memref.store %ascii_minus, %buffer[%minus_pos] : memref<?xi8>
        scf.yield %minus_pos : index
      } else {
        scf.yield %conversion#1 : index
      }
      %length_index = arith.subi %decimal_capacity_index, %start : index
      %length = arith.index_cast %length_index : index to i64
      %h, %b = func.call @__ly_unicode_from_valid_utf8(%buffer, %start, %length) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
      memref.dealloc %buffer : memref<?xi8>
      memref.dealloc %tmp : memref<?xi32>
      scf.yield %h, %b : memref<2xi64>, memref<?xi8>
    }
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  // Integer presentation over a parsed spec; bool delegates here with its
  // own type name so error texts match CPython's.
  func.func private @__ly_long_format_impl(%header: memref<2xi64>, %spec: memref<?xi64>, %tname: memref<?xi8>, %tname_len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]} {
    %meta_raw, %digits_raw = func.call @__ly_long_parts(%header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %minus_one = arith.constant -1 : i64
    %true_i = arith.constant true
    %s0 = arith.constant 0 : index
    %s1 = arith.constant 1 : index
    %s2 = arith.constant 2 : index
    %s3 = arith.constant 3 : index
    %s4 = arith.constant 4 : index
    %s5 = arith.constant 5 : index
    %s6 = arith.constant 6 : index
    %s7 = arith.constant 7 : index
    %s8 = arith.constant 8 : index
    %s9 = arith.constant 9 : index
    %fill_rec = memref.load %spec[%s0] : memref<?xi64>
    %align_rec = memref.load %spec[%s1] : memref<?xi64>
    %sign_rec = memref.load %spec[%s2] : memref<?xi64>
    %alt_rec = memref.load %spec[%s3] : memref<?xi64>
    %zero_rec = memref.load %spec[%s4] : memref<?xi64>
    %width_rec = memref.load %spec[%s5] : memref<?xi64>
    %group_rec = memref.load %spec[%s6] : memref<?xi64>
    %prec_rec = memref.load %spec[%s7] : memref<?xi64>
    %type_rec = memref.load %spec[%s8] : memref<?xi64>
    %z_rec = memref.load %spec[%s9] : memref<?xi64>
    %meta, %digits = func.call @__ly_long_operand_view(%meta_raw, %digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)

    %ce = arith.constant 101 : i64
    %cE = arith.constant 69 : i64
    %cf = arith.constant 102 : i64
    %cF = arith.constant 70 : i64
    %cg = arith.constant 103 : i64
    %cG = arith.constant 71 : i64
    %cpct = arith.constant 37 : i64
    %f0 = arith.cmpi eq, %type_rec, %ce : i64
    %f1 = arith.cmpi eq, %type_rec, %cE : i64
    %f2 = arith.cmpi eq, %type_rec, %cf : i64
    %f3 = arith.cmpi eq, %type_rec, %cF : i64
    %f4 = arith.cmpi eq, %type_rec, %cg : i64
    %f5 = arith.cmpi eq, %type_rec, %cG : i64
    %f6 = arith.cmpi eq, %type_rec, %cpct : i64
    %fa = arith.ori %f0, %f1 : i1
    %fb = arith.ori %f2, %f3 : i1
    %fc = arith.ori %f4, %f5 : i1
    %fd = arith.ori %fa, %fb : i1
    %fe = arith.ori %fc, %f6 : i1
    %is_float_code = arith.ori %fd, %fe : i1
    cf.cond_br %is_float_code, ^float_path, ^int_path

  ^float_path:
    %fv:2 = func.call @__ly_long_view_as_f64_checked(%meta, %digits) : (memref<2xi64>, memref<?xi32>) -> (f64, i1)
    // fv#1 is the overflow flag (true = |value| too large for a double).
    scf.if %fv#1 {
      func.call @__ly_long_raise_too_large_for_float() : () -> ()
    }
    %fh, %fb2 = func.call @__ly_float_format_core(%fv#0, %spec, %tname, %tname_len) : (f64, memref<?xi64>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %fh, %fb2 : memref<2xi64>, memref<?xi8>

  ^int_path:
    %z_on = arith.cmpi ne, %z_rec, %zero : i64
    scf.if %z_on {
      func.call @__ly_fmt_raise_z_int() : () -> ()
    }
    %has_prec = arith.cmpi ne, %prec_rec, %minus_one : i64
    scf.if %has_prec {
      func.call @__ly_fmt_raise_int_precision() : () -> ()
    }
    %cd = arith.constant 100 : i64
    %t_unset = arith.cmpi eq, %type_rec, %zero : i64
    %t = arith.select %t_unset, %cd, %type_rec : i64
    %cn = arith.constant 110 : i64
    %cb = arith.constant 98 : i64
    %co = arith.constant 111 : i64
    %cx = arith.constant 120 : i64
    %cX = arith.constant 88 : i64
    %cc = arith.constant 99 : i64
    %t_d = arith.cmpi eq, %t, %cd : i64
    %t_n = arith.cmpi eq, %t, %cn : i64
    %t_b = arith.cmpi eq, %t, %cb : i64
    %t_o = arith.cmpi eq, %t, %co : i64
    %t_x = arith.cmpi eq, %t, %cx : i64
    %t_X = arith.cmpi eq, %t, %cX : i64
    %t_c = arith.cmpi eq, %t, %cc : i64
    %va = arith.ori %t_d, %t_n : i1
    %vb = arith.ori %t_b, %t_o : i1
    %vc = arith.ori %t_x, %t_X : i1
    %vd = arith.ori %va, %vb : i1
    %ve = arith.ori %vc, %t_c : i1
    %t_valid = arith.ori %vd, %ve : i1
    %t_invalid = arith.xori %t_valid, %true_i : i1
    scf.if %t_invalid {
      func.call @__ly_fmt_raise_unknown_code(%t, %tname, %tname_len) : (i64, memref<?xi8>, i64) -> ()
    }
    // grouping legality: ',' only with 'd'; '_' with d/b/o/x/X
    %comma = arith.constant 44 : i64
    %under = arith.constant 95 : i64
    %g_comma = arith.cmpi eq, %group_rec, %comma : i64
    %g_under = arith.cmpi eq, %group_rec, %under : i64
    %not_d = arith.xori %t_d, %true_i : i1
    %bad_comma = arith.andi %g_comma, %not_d : i1
    %under_ok0 = arith.ori %t_d, %vb : i1
    %under_ok = arith.ori %under_ok0, %vc : i1
    %not_under_ok = arith.xori %under_ok, %true_i : i1
    %bad_under = arith.andi %g_under, %not_under_ok : i1
    %bad_group = arith.ori %bad_comma, %bad_under : i1
    scf.if %bad_group {
      func.call @__ly_fmt_raise_cannot_group(%group_rec, %t) : (i64, i64) -> ()
    }
    cf.cond_br %t_c, ^char_path, ^digits_path

  ^char_path:
    %sign_set = arith.cmpi ne, %sign_rec, %zero : i64
    scf.if %sign_set {
      func.call @__ly_fmt_raise_sign_c() : () -> ()
    }
    %alt_set = arith.cmpi ne, %alt_rec, %zero : i64
    scf.if %alt_set {
      func.call @__ly_fmt_raise_alt_c() : () -> ()
    }
    %fits = func.call @__ly_long_view_fits_i64(%meta, %digits) : (memref<2xi64>, memref<?xi32>) -> i1
    %cp_max = arith.constant 1114111 : i64
    %cval = scf.if %fits -> (i64) {
      %v = func.call @__ly_long_view_as_i64(%meta, %digits) : (memref<2xi64>, memref<?xi32>) -> i64
      scf.yield %v : i64
    } else {
      scf.yield %minus_one : i64
    }
    %too_low = arith.cmpi slt, %cval, %zero : i64
    %too_high = arith.cmpi sgt, %cval, %cp_max : i64
    %out_of_range = arith.ori %too_low, %too_high : i1
    scf.if %out_of_range {
      func.call @__ly_fmt_raise_c_range() : () -> ()
    }
    // A width past the allocator's reach is MemoryError before this makes
    // anything: what is made here is freed after the render, not when the
    // render raises (`__ly_fmt_render_number` checks again, for its own sake).
    // ⛔ Not once before the 'c'/digits branch: the 'c' path's own errors
    // come first, as CPython's do.
    // (No width is -1, which unsigned is the largest of all.)
    %code_unit_bytes = arith.constant 4 : i64
    %nothing_before = arith.constant 0 : i64
    %width_or_none = arith.maxsi %width_rec, %nothing_before : i64
    func.call @__ly_check_alloc_count(%width_or_none, %code_unit_bytes, %nothing_before) : (i64, i64, i64) -> ()
    %cbuf_cap = arith.constant 1 : index
    %cbuf = memref.alloc(%cbuf_cap) : memref<?xi32>
    %cval32 = arith.trunci %cval : i64 to i32
    memref.store %cval32, %cbuf[%s0] : memref<?xi32>
    cf.br ^render(%cbuf, %one, %one, %zero, %zero, %zero, %zero : memref<?xi32>, i64, i64, i64, i64, i64, i64)

  ^digits_path:
    %digit_unit_bytes = arith.constant 4 : i64
    %none_before = arith.constant 0 : i64
    %digits_width = arith.maxsi %width_rec, %none_before : i64
    func.call @__ly_check_alloc_count(%digits_width, %digit_unit_bytes, %none_before) : (i64, i64, i64) -> ()
    %sign_slot = arith.constant 0 : index
    %sgn = memref.load %meta[%sign_slot] : memref<2xi64>
    %negative = arith.cmpi slt, %sgn, %zero : i64
    %is_dec = arith.ori %t_d, %t_n : i1
    %nb:2 = scf.if %is_dec -> (memref<?xi32>, i64) {
      %rh, %rb = func.call @LyLong_Repr(%header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi8>)
      %c0i = arith.constant 0 : index
      %rdim = memref.dim %rb, %c0i : memref<?xi8>
      %rlen = arith.index_cast %rdim : index to i64
      %skip = arith.select %negative, %one, %zero : i64
      %ndig = arith.subi %rlen, %skip : i64
      %ndig_idx = arith.index_cast %ndig : i64 to index
      %buf = memref.alloc(%ndig_idx) : memref<?xi32>
      %skip_idx = arith.index_cast %skip : i64 to index
      %c1i = arith.constant 1 : index
      scf.for %i = %c0i to %ndig_idx step %c1i {
        %src = arith.addi %i, %skip_idx : index
        %b = memref.load %rb[%src] : memref<?xi8>
        %w = arith.extui %b : i8 to i32
        memref.store %w, %buf[%i] : memref<?xi32>
      }
      func.call @LyUnicode_DecRef(%rh) : (memref<2xi64>) -> ()
      scf.yield %buf, %ndig : memref<?xi32>, i64
    } else {
      %shift1 = arith.constant 1 : i64
      %shift3 = arith.constant 3 : i64
      %shift4 = arith.constant 4 : i64
      %sh0 = arith.select %t_b, %shift1, %shift4 : i64
      %sh = arith.select %t_o, %shift3, %sh0 : i64
      %bits = func.call @__ly_long_bit_length(%meta, %digits) : (memref<2xi64>, memref<?xi32>) -> i64
      %cap0 = arith.divui %bits, %sh : i64
      %cap1 = arith.constant 2 : i64
      %cap2 = arith.addi %cap0, %cap1 : i64
      %cap_idx = arith.index_cast %cap2 : i64 to index
      %buf = memref.alloc(%cap_idx) : memref<?xi32>
      %nd = func.call @__ly_fmt_int_base2k(%meta, %digits, %sh, %t_X, %buf) : (memref<2xi64>, memref<?xi32>, i64, i1, memref<?xi32>) -> i64
      scf.yield %buf, %nd : memref<?xi32>, i64
    }
    %plus = arith.constant 43 : i64
    %space = arith.constant 32 : i64
    %minus_cp = arith.constant 45 : i64
    %sp_plus = arith.cmpi eq, %sign_rec, %plus : i64
    %sp_space = arith.cmpi eq, %sign_rec, %space : i64
    %pos_sign0 = arith.select %sp_plus, %plus, %zero : i64
    %pos_sign = arith.select %sp_space, %space, %pos_sign0 : i64
    %sign_cp = arith.select %negative, %minus_cp, %pos_sign : i64
    %alt_on = arith.cmpi ne, %alt_rec, %zero : i64
    %need_prefix0 = arith.ori %vb, %vc : i1
    %need_prefix = arith.andi %alt_on, %need_prefix0 : i1
    %zero48 = arith.constant 48 : i64
    %pre0 = arith.select %need_prefix, %zero48, %zero : i64
    %pre1 = arith.select %need_prefix, %t, %zero : i64
    %gs4 = arith.constant 4 : i64
    %gs3 = arith.constant 3 : i64
    %gs = arith.select %is_dec, %gs3, %gs4 : i64
    cf.br ^render(%nb#0, %nb#1, %nb#1, %sign_cp, %pre0, %pre1, %gs : memref<?xi32>, i64, i64, i64, i64, i64, i64)

  ^render(%body_r: memref<?xi32>, %blen_r: i64, %ilen_r: i64, %sign_r: i64, %pre0_r: i64, %pre1_r: i64, %gs_r: i64):
    %gs_eff = arith.maxsi %gs_r, %one : i64
    %zero_flag = arith.cmpi ne, %zero_rec, %zero : i64
    %fill_unset = arith.cmpi eq, %fill_rec, %minus_one : i64
    %fill_zero = arith.constant 48 : i64
    %fill_space = arith.constant 32 : i64
    %fill_def = arith.select %zero_flag, %fill_zero, %fill_space : i64
    %fill_cp = arith.select %fill_unset, %fill_def, %fill_rec : i64
    %align_unset = arith.cmpi eq, %align_rec, %zero : i64
    %align_eq = arith.constant 61 : i64
    %align_gt = arith.constant 62 : i64
    %align_def = arith.select %zero_flag, %align_eq, %align_gt : i64
    %align_cp = arith.select %align_unset, %align_def, %align_rec : i64
    %group_eff = scf.if %t_c -> (i64) {
      scf.yield %zero : i64
    } else {
      scf.yield %group_rec : i64
    }
    %header_r, %bytes_r = func.call @__ly_fmt_render_number(%sign_r, %pre0_r, %pre1_r, %body_r, %blen_r, %ilen_r, %group_eff, %gs_eff, %fill_cp, %align_cp, %width_rec) : (i64, i64, i64, memref<?xi32>, i64, i64, i64, i64, i64, i64, i64) -> (memref<2xi64>, memref<?xi8>)
    memref.dealloc %body_r : memref<?xi32>
    func.return %header_r, %bytes_r : memref<2xi64>, memref<?xi8>
  }

  func.func @LyLong_Format(%header: memref<2xi64> {ly.ownership.object_header}, %spec_header: memref<2xi64> {ly.ownership.object_header}, %spec_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__format__", ly.runtime.result_contract = "builtins.str"} {
    %spec_store = memref.alloca() : memref<10xi64>
    %spec = memref.cast %spec_store : memref<10xi64> to memref<?xi64>
    %ok = func.call @__ly_fmt_parse_spec(%spec_header, %spec_bytes, %spec) : (memref<2xi64>, memref<?xi8>, memref<?xi64>) -> i1
    %true_lf = arith.constant true
    %bad = arith.xori %ok, %true_lf : i1
    scf.if %bad {
      %names = memref.get_global @__ly_fmt_msg_name_int : memref<3xi8>
      %name = memref.cast %names : memref<3xi8> to memref<?xi8>
      %nlen = arith.constant 3 : i64
      func.call @__ly_fmt_raise_invalid_spec(%spec_header, %spec_bytes, %name, %nlen) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> ()
    }
    %names2 = memref.get_global @__ly_fmt_msg_name_int : memref<3xi8>
    %name2 = memref.cast %names2 : memref<3xi8> to memref<?xi8>
    %nlen2 = arith.constant 3 : i64
    %h, %b = func.call @__ly_long_format_impl(%header, %spec, %name2, %nlen2) : (memref<2xi64>, memref<?xi64>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %h, %b : memref<2xi64>, memref<?xi8>
  }

  // int.__index__ is the identity (CPython long_long returns the receiver for
  // an exact int). A copy rather than a retain of the argument view: the
  // manifest method ABI owns result 0, and the incoming meta/digits are a
  // borrowed operand view, not an owned header.
  func.func @LyLong_Index(%header: memref<2xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__index__"} {
    %meta_raw, %digits_raw = func.call @__ly_long_parts(%header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %meta, %digits = func.call @__ly_long_operand_view(%meta_raw, %digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %h = func.call @__ly_long_copy(%meta, %digits) : (memref<2xi64>, memref<?xi32>) -> memref<2xi64>
    func.return %h : memref<2xi64>
  }

  func.func @LyLong_Abs(%header: memref<2xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__abs__"} {
    %meta_raw, %digits_raw = func.call @__ly_long_parts(%header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %meta, %digits = func.call @__ly_long_operand_view(%meta_raw, %digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %sign = memref.load %meta[%c0] : memref<2xi64>
    %negative = arith.cmpi slt, %sign, %zero : i64
    %new_sign = arith.select %negative, %one, %sign : i1, i64
    %h = func.call @__ly_long_copy_with_sign(%new_sign, %meta, %digits) : (i64, memref<2xi64>, memref<?xi32>) -> memref<2xi64>
    func.return %h : memref<2xi64>
  }

  // Power-of-two base rendering shared by hex/oct/bin: sign, "0<letter>",
  // then bl/bits digits pulled straight from the 30-bit limbs.
  func.func private @__ly_long_format_pow2(%meta: memref<2xi64>, %digits: memref<?xi32>, %bits: i64, %letter: i8) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %sign = memref.load %meta[%c0] : memref<2xi64>
    %count = memref.load %meta[%c1] : memref<2xi64>
    %is_zero = arith.cmpi eq, %sign, %zero : i64
    %bl_raw = func.call @__ly_long_bit_length(%meta, %digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %bl = arith.select %is_zero, %one, %bl_raw : i1, i64
    %bits_minus = arith.subi %bits, %one : i64
    %num = arith.addi %bl, %bits_minus : i64
    %ndigits = arith.divui %num, %bits : i64
    %negative = arith.cmpi slt, %sign, %zero : i64
    %neg_len = arith.select %negative, %one, %zero : i1, i64
    %two = arith.constant 2 : i64
    %prefix_total = arith.addi %neg_len, %two : i64
    %total = arith.addi %prefix_total, %ndigits : i64
    %total_index = arith.index_cast %total : i64 to index
    %buffer = memref.alloca(%total_index) : memref<?xi8>
    %minus_ch = arith.constant 45 : i8
    %zero_ch = arith.constant 48 : i8
    scf.if %negative {
      memref.store %minus_ch, %buffer[%c0] : memref<?xi8>
    }
    %neg_len_index = arith.index_cast %neg_len : i64 to index
    memref.store %zero_ch, %buffer[%neg_len_index] : memref<?xi8>
    %letter_pos = arith.addi %neg_len_index, %c1 : index
    memref.store %letter, %buffer[%letter_pos] : memref<?xi8>
    %digits_start = arith.addi %letter_pos, %c1 : index
    %ndigits_index = arith.index_cast %ndigits : i64 to index
    %mask_bit = arith.constant 1 : i64
    %mask0 = arith.shli %mask_bit, %bits : i64
    %mask = arith.subi %mask0, %one : i64
    %c30 = arith.constant 30 : i64
    scf.for %k = %c0 to %ndigits_index step %c1 {
      %k_i64 = arith.index_cast %k : index to i64
      %rev = arith.subi %ndigits, %k_i64 : i64
      %digit_pos = arith.subi %rev, %one : i64
      %bitpos = arith.muli %digit_pos, %bits : i64
      %q = arith.divui %bitpos, %c30 : i64
      %r = arith.remui %bitpos, %c30 : i64
      %q_index = arith.index_cast %q : i64 to index
      %q_in_range = arith.cmpi slt, %q, %count : i64
      %lo = scf.if %q_in_range -> (i64) {
        %limb_i32 = memref.load %digits[%q_index] : memref<?xi32>
        %limb = arith.extui %limb_i32 : i32 to i64
        %sh = arith.shrui %limb, %r : i64
        scf.yield %sh : i64
      } else {
        scf.yield %zero : i64
      }
      %q1 = arith.addi %q, %one : i64
      %spill = arith.addi %r, %bits : i64
      %spills = arith.cmpi sgt, %spill, %c30 : i64
      %q1_in_range = arith.cmpi slt, %q1, %count : i64
      %use_hi = arith.andi %spills, %q1_in_range : i1
      %hi = scf.if %use_hi -> (i64) {
        %q1_index = arith.index_cast %q1 : i64 to index
        %limb_i32 = memref.load %digits[%q1_index] : memref<?xi32>
        %limb = arith.extui %limb_i32 : i32 to i64
        %up_by = arith.subi %c30, %r : i64
        %sh = arith.shli %limb, %up_by : i64
        scf.yield %sh : i64
      } else {
        scf.yield %zero : i64
      }
      %merged = arith.ori %lo, %hi : i64
      %digit = arith.andi %merged, %mask : i64
      %ten = arith.constant 10 : i64
      %is_decimal = arith.cmpi ult, %digit, %ten : i64
      %dec_base = arith.constant 48 : i64
      %alpha_base = arith.constant 87 : i64
      %dec_ch = arith.addi %digit, %dec_base : i64
      %alpha_ch = arith.addi %digit, %alpha_base : i64
      %ch_i64 = arith.select %is_decimal, %dec_ch, %alpha_ch : i1, i64
      %ch = arith.trunci %ch_i64 : i64 to i8
      %dst = arith.addi %digits_start, %k : index
      memref.store %ch, %buffer[%dst] : memref<?xi8>
    }
    %h, %b = func.call @__ly_unicode_from_valid_utf8(%buffer, %c0, %total) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %h, %b : memref<2xi64>, memref<?xi8>
  }
}
