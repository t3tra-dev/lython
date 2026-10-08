// `float` -- CPython's Objects/floatobject.c, with float(str)
// (PyFloat_FromString) and the repr that PyOS_double_to_string spells.
//
// Signature source (1:1 correspondence target):
//   https://github.com/python/typeshed/blob/main/stdlib/builtins.pyi
//
// Deviations from CPython:
//   - float.__round__ takes ndigits as an `int`, not CPython's
//     `SupportsIndex`: `round(x, obj)` with an `__index__` object is refused
//     at compile time.

module attributes {
  ly.typing.manifest,
  ly.runtime.contracts = ["builtins.float"]
} {
  // ===== declared here, defined in another runtime file or built by the lowering =====
  func.func private @LyLong_DecRef(%header: memref<2xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.int", ly.runtime.deallocator}
  func.func private @LyLong_LShift(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_header: memref<2xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__lshift__"}
  func.func private @__ly_tuple_alloc(%length: i64) -> memref<5xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.tuple"], ly.ownership.owned_results = [0]}
  func.func private @__ly_tuple_items(%self: memref<5xi64>) -> memref<?xi64> attributes {ly.runtime.contract = "builtins.tuple", ly.runtime.interior_word, ly.runtime.primitive = "items_view"}
  func.func private @__ly_tuple_store_long(%items: memref<?xi64>, %slot: index, %h: memref<2xi64>)
  func.func private @__ly_unicode_from_valid_utf8(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]}
  func.func private @LyBaseException_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.BaseException", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"}
  func.func private @LyBaseException_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 5 : i64, ly.runtime.contract = "builtins.BaseException", ly.runtime.initializer = "__new__"}
  func.func private @LyEH_ThrowException(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "raise"}
  func.func private @LyLong_FromI64(%value: i64 {ly.runtime.default_i64 = 0 : i64}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 1 : i64, ly.runtime.contract = "builtins.int", ly.runtime.initializer = "__new__"}
  func.func private @LyObject_ReleaseStorageToZero(%storage: memref<?xi64>) -> i1 attributes {ly.runtime.contract = "builtins.object", ly.runtime.primitive = "release_to_zero"}
  func.func private @LyUnicode_Concat(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_bytes: memref<?xi8>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__add__"}
  func.func private @LyUnicode_DecRef(%header: memref<2xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.str", ly.runtime.deallocator}
  func.func private @LyUnicode_FromBytes(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 4 : i64, ly.runtime.contract = "builtins.str", ly.runtime.initializer = "__new__"}
  func.func private @LyUnicode_Repr(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"}
  func.func private @__ly_addresses_are_word_wide() -> i1
  func.func private @__ly_check_alloc_count(%count: i64, %element_bytes: i64, %extra: i64)
  func.func private @__ly_dtoa_counted(%abs: f64, %fixed_mode: i1, %req: i64, %digits_out: memref<?xi8>) -> (i64, i64)
  func.func private @__ly_dtoa_shortest(%abs: f64, %digits_out: memref<?xi8>) -> (i64, i64)
  func.func private @__ly_fmt_body_exp(%out: memref<?xi32>, %digits: memref<?xi8>, %count: i64, %decpt: i64, %mant_frac: i64, %force_dot: i1, %e_cp: i64) -> (i64, i64)
  func.func private @__ly_fmt_body_fixed(%out: memref<?xi32>, %digits: memref<?xi8>, %count: i64, %decpt: i64, %frac_len: i64, %min_frac: i64, %force_dot: i1) -> (i64, i64)
  memref.global "private" constant @__ly_fmt_msg_name_float : memref<5xi8>
  func.func private @__ly_fmt_parse_spec(%spec_h: memref<2xi64>, %spec_b: memref<?xi8>, %out: memref<?xi64>) -> i1
  func.func private @__ly_fmt_raise_cannot_group(%gcp: i64, %wcp: i64)
  func.func private @__ly_fmt_raise_invalid_spec(%spec_h: memref<2xi64>, %spec_b: memref<?xi8>, %name: memref<?xi8>, %name_len: i64)
  func.func private @__ly_fmt_raise_precision_too_big()
  func.func private @__ly_fmt_raise_unknown_code(%code: i64, %name: memref<?xi8>, %name_len: i64)
  func.func private @__ly_fmt_render_number(%sign_cp: i64, %pre0: i64, %pre1: i64, %body: memref<?xi32>, %body_len: i64, %int_len: i64, %group_cp: i64, %group_size: i64, %fill_cp: i64, %align_cp: i64, %width_in: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]}
  func.func private @__ly_global_view_i64(%pointer: i64, %size: i64) -> memref<?xi64>
  func.func private @__ly_handle_retain_raw(%entity: i64)
  func.func private @__ly_hash_fixup(%h: i64) -> i64
  func.func private @__ly_long_alloc_raw(%sign: i64, %capacity: i64) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0]}
  func.func private @__ly_long_cmp_f64(%meta_raw: memref<2xi64>, %digits_raw: memref<?xi32>, %d: f64) -> i64
  func.func private @__ly_long_normalize(%meta: memref<2xi64>, %digits: memref<?xi32>, %capacity: i64)
  func.func private @__ly_long_parts(%header: memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
  func.func private @__ly_long_raise_division_by_zero()
  func.func private @__ly_long_raise_float_infinity()
  func.func private @__ly_long_raise_float_nan()
  func.func private @__ly_long_raise_fractional_power_negative()
  func.func private @__ly_long_raise_zero_negative_power()
  func.func private @__ly_long_view_as_i64(%meta: memref<2xi64>, %digits: memref<?xi32>) -> i64
  func.func private @__ly_long_view_fits_i64(%meta: memref<2xi64>, %digits: memref<?xi32>) -> i1
  func.func private @__ly_raise_static_message(%class_id: i64, %message: memref<?xi8>, %length: i64)
  func.func private @__ly_slot_word_from_view_address(%address: i64) -> i64
  func.func private @__ly_slot_word_is_immediate(%word: i64) -> i1
  func.func private @__ly_unicode_count(%header: memref<2xi64>, %bytes: memref<?xi8>) -> i64
  func.func private @__ly_unicode_get(%bytes: memref<?xi8>, %width: i64, %i: index) -> i64
  func.func private @__ly_unicode_width(%header: memref<2xi64>) -> i64

  py.class @float attributes {
    base_names = ["object"], ly.typing.final,
    method_names = ["__new__", "__repr__", "__add__", "__sub__", "__mul__",
                    "__truediv__", "__floordiv__", "__mod__", "__float__",
                    "__bool__", "__round__", "__round__", "__lt__", "__le__",
                    "__gt__",
                    "__ge__", "__str__", "__eq__", "__ne__", "__pow__",
                    "__hash__", "__abs__", "__format__",
                    "__lt__", "__le__", "__gt__", "__ge__", "__eq__", "__ne__",
                    "__neg__", "__pos__", "conjugate", "is_integer", "as_integer_ratio", "hex", "fromhex"],
    method_contracts = [
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.float">>, !py.contract<"typing.SupportsFloat">] -> [!py.self]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">, !py.contract<"builtins.float">] -> [!py.contract<"builtins.float">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">, !py.contract<"builtins.float">] -> [!py.contract<"builtins.float">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">, !py.contract<"builtins.float">] -> [!py.contract<"builtins.float">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">, !py.contract<"builtins.float">] -> [!py.contract<"builtins.float">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">, !py.contract<"builtins.float">] -> [!py.contract<"builtins.float">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">, !py.contract<"builtins.float">] -> [!py.contract<"builtins.float">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">] -> [!py.contract<"builtins.float">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.float">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">, !py.contract<"builtins.float">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">, !py.contract<"builtins.float">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">, !py.contract<"builtins.float">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">, !py.contract<"builtins.float">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">, !py.contract<"builtins.float">] -> [!py.contract<"builtins.float">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">] -> [!py.contract<"builtins.float">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">] -> [!py.contract<"builtins.float">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">] -> [!py.contract<"builtins.float">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">] -> [!py.contract<"builtins.float">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">] -> [!py.contract<"builtins.tuple", [!py.contract<"builtins.int">, !py.contract<"builtins.int">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.float">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.float">>, !py.contract<"builtins.str">] -> [!py.contract<"builtins.float">]>
    ],
    method_kinds = ["classmethod", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance", "classmethod"]
  } {}

  // A float is immediate when the top three bits of its exponent are 011 or
  // 100 -- magnitudes in [2^-255, 2^256), every float most programs make --
  // or when it is +0.0. Rotating left by three brings sign and the two high
  // exponent bits to the bottom; those two bits are recoverable from the
  // third (bit 63 after the rotation), so they make room for the tag `10`.
  // This is Ruby's flonum. +0.0 takes the word of 0x3000000000000000 (the
  // one in-range pattern excluded), whose rotation is the word below.
  func.func private @__ly_float_immediate_fits(%bits: i64) -> i1 {
    %zero = arith.constant 0 : i64
    %sixty = arith.constant 60 : i64
    %seven = arith.constant 7 : i64
    %three_top = arith.constant 3 : i64
    %one = arith.constant 1 : i64
    %excluded = arith.constant 3458764513820540928 : i64
    %top = arith.shrui %bits, %sixty : i64
    %exp_top = arith.andi %top, %seven : i64
    %rebased = arith.subi %exp_top, %three_top : i64
    %in_range = arith.cmpi ule, %rebased, %one : i64
    %not_excluded = arith.cmpi ne, %bits, %excluded : i64
    %ranged = arith.andi %in_range, %not_excluded : i1
    %is_zero = arith.cmpi eq, %bits, %zero : i64
    %encodable = arith.ori %ranged, %is_zero : i1
    %wide = func.call @__ly_addresses_are_word_wide() : () -> i1
    %fits = arith.andi %encodable, %wide : i1
    func.return %fits : i1
  }

  func.func private @__ly_float_to_immediate(%bits: i64) -> i64 {
    %zero = arith.constant 0 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %sixty_one = arith.constant 61 : i64
    %low_clear = arith.constant -4 : i64
    %zero_word = arith.constant -9223372036854775806 : i64
    %high = arith.shli %bits, %three : i64
    %low = arith.shrui %bits, %sixty_one : i64
    %rotated = arith.ori %high, %low : i64
    %cleared = arith.andi %rotated, %low_clear : i64
    %tagged = arith.ori %cleared, %two : i64
    %is_zero = arith.cmpi eq, %bits, %zero : i64
    %word = arith.select %is_zero, %zero_word, %tagged : i1, i64
    func.return %word : i64
  }

  func.func private @__ly_float_from_immediate(%word: i64) -> i64 {
    %zero = arith.constant 0 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %sixty_one = arith.constant 61 : i64
    %sixty_three = arith.constant 63 : i64
    %low_clear = arith.constant -4 : i64
    %zero_word = arith.constant -9223372036854775806 : i64
    %b63 = arith.shrui %word, %sixty_three : i64
    %restored_low = arith.subi %two, %b63 : i64
    %cleared = arith.andi %word, %low_clear : i64
    %rotated = arith.ori %cleared, %restored_low : i64
    %low = arith.shrui %rotated, %three : i64
    %high = arith.shli %rotated, %sixty_one : i64
    %bits = arith.ori %low, %high : i64
    %is_zero = arith.cmpi eq, %word, %zero_word : i64
    %result = arith.select %is_zero, %zero, %bits : i1, i64
    func.return %result : i64
  }

  // "could not convert string to float: "
  memref.global "private" constant @__ly_float_msg_invalid_literal_prefix : memref<35xi8> = dense<[99, 111, 117, 108, 100, 32, 110, 111, 116, 32, 99, 111, 110, 118, 101, 114, 116, 32, 115, 116, 114, 105, 110, 103, 32, 116, 111, 32, 102, 108, 111, 97, 116, 58, 32]>

  func.func private @__ly_float_raise_invalid_literal(%subject_header: memref<2xi64> {ly.ownership.object_header}, %subject_bytes: memref<?xi8>) {
    %class_id = arith.constant 53 : i64
    %start = arith.constant 0 : index
    %prefix_length = arith.constant 35 : i64
    %prefix_static = memref.get_global @__ly_float_msg_invalid_literal_prefix : memref<35xi8>
    %prefix_bytes = memref.cast %prefix_static : memref<35xi8> to memref<?xi8>
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

  // float(str). Runtime-level __float__ on str, the twin of LyLong_FromStr's
  // __int__: CPython has no str.__float__ either, and only the emitter's
  // float(x) rewrite targets it.
  //
  // The digits go to strtod rather than to a hand-rolled accumulate, because
  // "2.5" has to round the way CPython rounds it and CPython's own dtoa is
  // correctly rounded -- an int mantissa divided by a power of ten rounds
  // twice and is off by an ulp on inputs this function must not be wrong on.
  // What is checked HERE is everything strtod is more permissive about than
  // float() is: a hex-float spelling ("0x1p3", which strtod accepts and
  // CPython rejects), trailing garbage, and underscore placement.
  func.func @LyFloat_FromStr(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> memref<3xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__float__", ly.runtime.result_contract = "builtins.float"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero64 = arith.constant 0 : i64
    %one64 = arith.constant 1 : i64
    %neg_one = arith.constant -1 : i64
    %true = arith.constant true
    %false = arith.constant false
    %ch_nul = arith.constant 0 : i8
    %cp_space = arith.constant 32 : i64
    %cp_tab = arith.constant 9 : i64
    %cp_cr = arith.constant 13 : i64
    %cp_underscore = arith.constant 95 : i64
    %cp_zero = arith.constant 48 : i64
    %cp_nine = arith.constant 57 : i64
    %cp_ascii_max = arith.constant 127 : i64

    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %len = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %len_index = arith.index_cast %len : i64 to index

    // Trim ASCII whitespace at both ends (strtod skips it in front only).
    %first = scf.for %i = %c0 to %len_index step %c1 iter_args(%found = %neg_one) -> (i64) {
      %cp = func.call @__ly_unicode_get(%bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
      %is_sp = arith.cmpi eq, %cp, %cp_space : i64
      %ge_tab = arith.cmpi sge, %cp, %cp_tab : i64
      %le_cr = arith.cmpi sle, %cp, %cp_cr : i64
      %is_ctl = arith.andi %ge_tab, %le_cr : i1
      %is_ws = arith.ori %is_sp, %is_ctl : i1
      %unset = arith.cmpi eq, %found, %neg_one : i64
      %i64v = arith.index_cast %i : index to i64
      %candidate = arith.select %is_ws, %found, %i64v : i1, i64
      %next = arith.select %unset, %candidate, %found : i1, i64
      scf.yield %next : i64
    }
    %last = scf.for %i = %c0 to %len_index step %c1 iter_args(%found = %neg_one) -> (i64) {
      %step = arith.addi %i, %c1 : index
      %rev = arith.subi %len_index, %step : index
      %cp = func.call @__ly_unicode_get(%bytes, %width, %rev) : (memref<?xi8>, i64, index) -> i64
      %is_sp = arith.cmpi eq, %cp, %cp_space : i64
      %ge_tab = arith.cmpi sge, %cp, %cp_tab : i64
      %le_cr = arith.cmpi sle, %cp, %cp_cr : i64
      %is_ctl = arith.andi %ge_tab, %le_cr : i1
      %is_ws = arith.ori %is_sp, %is_ctl : i1
      %unset = arith.cmpi eq, %found, %neg_one : i64
      %rev64 = arith.index_cast %rev : index to i64
      %candidate = arith.select %is_ws, %found, %rev64 : i1, i64
      %next = arith.select %unset, %candidate, %found : i1, i64
      scf.yield %next : i64
    }
    %empty = arith.cmpi eq, %first, %neg_one : i64
    cf.cond_br %empty, ^invalid, ^scan

  ^scan:
    %end = arith.addi %last, %one64 : i64
    %span = arith.subi %end, %first : i64
    %span_index = arith.index_cast %span : i64 to index
    %buffer_len = arith.addi %span, %one64 : i64
    %buffer_index = arith.index_cast %buffer_len : i64 to index
    %buffer = memref.alloc(%buffer_index) : memref<?xi8>
    %start_index = arith.index_cast %first : i64 to index
    %end_index = arith.index_cast %end : i64 to index

    // One pass: copy, drop underscores, and reject everything float() does
    // not accept -- a non-ASCII code point, an 'x'/'X' (the hex-float
    // spelling), and an underscore that is not between two digits.
    %scan_state:3 = scf.for %i = %start_index to %end_index step %c1 iter_args(%out = %c0, %ok = %true, %prev_digit = %false) -> (index, i1, i1) {
      %cp = func.call @__ly_unicode_get(%bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
      %too_wide = arith.cmpi sgt, %cp, %cp_ascii_max : i64
      %cp_x_lower = arith.constant 120 : i64
      %cp_x_upper = arith.constant 88 : i64
      %is_x_lower = arith.cmpi eq, %cp, %cp_x_lower : i64
      %is_x_upper = arith.cmpi eq, %cp, %cp_x_upper : i64
      %is_x = arith.ori %is_x_lower, %is_x_upper : i1
      %is_underscore = arith.cmpi eq, %cp, %cp_underscore : i64
      %ge_zero = arith.cmpi sge, %cp, %cp_zero : i64
      %le_nine = arith.cmpi sle, %cp, %cp_nine : i64
      %is_digit = arith.andi %ge_zero, %le_nine : i1
      // An underscore needs a digit before it and a digit after it; the
      // "after" half is the next character's own check, so a run of two
      // underscores fails on the second one and a trailing one fails below.
      %bad_underscore_pre = arith.xori %prev_digit, %true : i1
      %bad_underscore = arith.andi %is_underscore, %bad_underscore_pre : i1
      %bad_char = arith.ori %too_wide, %is_x : i1
      %bad = arith.ori %bad_char, %bad_underscore : i1
      %still_ok_pre = arith.xori %bad, %true : i1
      %still_ok = arith.andi %ok, %still_ok_pre : i1
      %keep = arith.xori %is_underscore, %true : i1
      %next_out = scf.if %keep -> (index) {
        %byte = arith.trunci %cp : i64 to i8
        memref.store %byte, %buffer[%out] : memref<?xi8>
        %bumped = arith.addi %out, %c1 : index
        scf.yield %bumped : index
      } else {
        scf.yield %out : index
      }
      // An underscore clears the flag as well as reading it, so `1__0` fails
      // on the second one the way CPython's parser does.
      scf.yield %next_out, %still_ok, %is_digit : index, i1, i1
    }
    // A trailing underscore leaves no digit after it.
    %last_index = arith.subi %end_index, %c1 : index
    %last_cp = func.call @__ly_unicode_get(%bytes, %width, %last_index) : (memref<?xi8>, i64, index) -> i64
    %ends_underscore = arith.cmpi eq, %last_cp, %cp_underscore : i64
    %trailing_ok = arith.xori %ends_underscore, %true : i1
    %clean = arith.andi %scan_state#1, %trailing_ok : i1
    %copied = arith.index_cast %scan_state#0 : index to i64
    %has_digits = arith.cmpi sgt, %copied, %zero64 : i64
    %parseable = arith.andi %clean, %has_digits : i1
    memref.store %ch_nul, %buffer[%scan_state#0] : memref<?xi8>
    cf.cond_br %parseable, ^parse, ^invalid_free

  ^invalid_free:
    memref.dealloc %buffer : memref<?xi8>
    cf.br ^invalid

  ^parse:
    %text_index = memref.extract_aligned_pointer_as_index %buffer : memref<?xi8> -> index
    %text_word = arith.index_cast %text_index : index to i64
    %text_ptr = llvm.inttoptr %text_word : i64 to !llvm.ptr
    %endptr_slot = memref.alloc() : memref<1xi64>
    memref.store %zero64, %endptr_slot[%c0] : memref<1xi64>
    %slot_index = memref.extract_aligned_pointer_as_index %endptr_slot : memref<1xi64> -> index
    %slot_word = arith.index_cast %slot_index : index to i64
    %slot_ptr = llvm.inttoptr %slot_word : i64 to !llvm.ptr
    %parsed = func.call @strtod(%text_ptr, %slot_ptr) : (!llvm.ptr, !llvm.ptr) -> f64
    %stop_word = memref.load %endptr_slot[%c0] : memref<1xi64>
    memref.dealloc %endptr_slot : memref<1xi64>
    %consumed = arith.subi %stop_word, %text_word : i64
    %all_consumed = arith.cmpi eq, %consumed, %copied : i64
    memref.dealloc %buffer : memref<?xi8>
    cf.cond_br %all_consumed, ^ok, ^invalid

  ^ok:
    %result = func.call @LyFloat_FromF64(%parsed) : (f64) -> memref<3xi64>
    func.return %result : memref<3xi64>

  ^invalid:
    func.call @__ly_float_raise_invalid_literal(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> ()
    %zero_f = arith.constant 0.0 : f64
    %unreached = func.call @LyFloat_FromF64(%zero_f) : (f64) -> memref<3xi64>
    func.return %unreached : memref<3xi64>
  }

  // Truncating float -> int conversion (runtime-level __int__ on float).
  // |x| < 2^63 goes through fptosi; larger finite magnitudes are exact:
  // every such double is the integer mantissa * 2^exponent, so the digits
  // are the 53 mantissa bits placed at bit offset `exponent` (CPython
  // PyLong_FromDouble). NaN and infinity raise with CPython's messages.
  func.func @LyFloat_Int(%header: memref<3xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.float", ly.runtime.method = "__int__", ly.runtime.result_contract = "builtins.int"} {
    %value_slot = arith.constant 2 : index
    %value_bits = memref.load %header[%value_slot] : memref<3xi64>
    %value = arith.bitcast %value_bits : i64 to f64
    %is_nan = arith.cmpf uno, %value, %value : f64
    cf.cond_br %is_nan, ^nan, ^check_inf

  ^nan:
    func.call @__ly_long_raise_float_nan() : () -> ()
    %zero_nan = arith.constant 0 : i64
    %nh = func.call @LyLong_FromI64(%zero_nan) : (i64) -> memref<2xi64>
    func.return %nh : memref<2xi64>

  ^check_inf:
    %bits = arith.bitcast %value : f64 to i64
    %exp_shift = arith.constant 52 : i64
    %exp_mask = arith.constant 2047 : i64
    %exp_raw_shifted = arith.shrui %bits, %exp_shift : i64
    %exp_raw = arith.andi %exp_raw_shifted, %exp_mask : i64
    %is_inf = arith.cmpi eq, %exp_raw, %exp_mask : i64
    cf.cond_br %is_inf, ^infinity, ^check_range

  ^infinity:
    func.call @__ly_long_raise_float_infinity() : () -> ()
    %zero_inf = arith.constant 0 : i64
    %ih = func.call @LyLong_FromI64(%zero_inf) : (i64) -> memref<2xi64>
    func.return %ih : memref<2xi64>

  ^check_range:
    %lower = arith.constant -9.2233720368547758E+18 : f64
    %upper = arith.constant 9.2233720368547758E+18 : f64
    %ge_lower = arith.cmpf oge, %value, %lower : f64
    %lt_upper = arith.cmpf olt, %value, %upper : f64
    %in_range = arith.andi %ge_lower, %lt_upper : i1
    cf.cond_br %in_range, ^convert, ^big

  ^big:
    // Finite with |x| >= 2^63: normal, exponent field >= 1086. The value is
    // (mantissa | 2^52) * 2^(exp_raw - 1075) with a positive binary exponent,
    // so each 30-bit limb is a window of the shifted 53-bit mantissa.
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %neg_one = arith.constant -1 : i64
    %thirty = arith.constant 30 : i64
    %mask30 = arith.constant 1073741823 : i64
    %mant_mask = arith.constant 4503599627370495 : i64
    %implicit_bit = arith.constant 4503599627370496 : i64
    %mant_low = arith.andi %bits, %mant_mask : i64
    %mantissa = arith.ori %mant_low, %implicit_bit : i64
    %exp_bias = arith.constant 1075 : i64
    %e = arith.subi %exp_raw, %exp_bias : i64
    %sign_negative = arith.cmpi slt, %bits, %zero : i64
    %sign = arith.select %sign_negative, %neg_one, %one : i1, i64
    %c53 = arith.constant 53 : i64
    %c29 = arith.constant 29 : i64
    %total_bits = arith.addi %e, %c53 : i64
    %rounded_up = arith.addi %total_bits, %c29 : i64
    %ndigits = arith.divui %rounded_up, %thirty : i64
    %h = func.call @__ly_long_alloc_raw(%sign, %ndigits) : (i64, i64) -> memref<2xi64>
    %m, %d = func.call @__ly_long_parts(%h) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %c0i = arith.constant 0 : index
    %c1i = arith.constant 1 : index
    %ndigits_index = arith.index_cast %ndigits : i64 to index
    scf.for %j = %c0i to %ndigits_index step %c1i {
      %j_i64 = arith.index_cast %j : index to i64
      %j30 = arith.muli %j_i64, %thirty : i64
      %s = arith.subi %j30, %e : i64
      // s >= 0: window is mantissa >> s (zero once s >= 53).
      // s in (-30, 0): window is (mantissa << -s) & mask.
      // s <= -30: below the mantissa, zero.
      %s_nonneg = arith.cmpi sge, %s, %zero : i64
      %s_lt53 = arith.cmpi slt, %s, %c53 : i64
      %right_ok = arith.andi %s_nonneg, %s_lt53 : i1
      %s_clamped = arith.select %right_ok, %s, %zero : i1, i64
      %right_shifted = arith.shrui %mantissa, %s_clamped : i64
      %right_part = arith.select %right_ok, %right_shifted, %zero : i1, i64
      %ns = arith.subi %zero, %s : i64
      %s_negative = arith.cmpi slt, %s, %zero : i64
      %ns_lt30 = arith.cmpi slt, %ns, %thirty : i64
      %left_ok = arith.andi %s_negative, %ns_lt30 : i1
      %ns_clamped = arith.select %left_ok, %ns, %zero : i1, i64
      %left_shifted = arith.shli %mantissa, %ns_clamped : i64
      %left_part = arith.select %left_ok, %left_shifted, %zero : i1, i64
      %part = arith.select %s_nonneg, %right_part, %left_part : i1, i64
      %digit_i64 = arith.andi %part, %mask30 : i64
      %digit = arith.trunci %digit_i64 : i64 to i32
      memref.store %digit, %d[%j] : memref<?xi32>
    }
    func.call @__ly_long_normalize(%m, %d, %ndigits) : (memref<2xi64>, memref<?xi32>, i64) -> ()
    func.return %h : memref<2xi64>

  ^convert:
    %truncated = arith.fptosi %value : f64 to i64
    %h2 = func.call @LyLong_FromI64(%truncated) : (i64) -> memref<2xi64>
    func.return %h2 : memref<2xi64>
  }

  func.func @LyFloat_LtLongBool(%lhs_header: memref<3xi64> {ly.ownership.object_header}, %rhs_header: memref<2xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.float", ly.runtime.method = "__lt__"} {
    %rhs_meta, %rhs_digits = func.call @__ly_long_parts(%rhs_header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %d = func.call @LyFloat_AsF64(%lhs_header) : (memref<3xi64>) -> f64
    %cmp = func.call @__ly_long_cmp_f64(%rhs_meta, %rhs_digits, %d) : (memref<2xi64>, memref<?xi32>, f64) -> i64
    %one = arith.constant 1 : i64
    %result = arith.cmpi eq, %cmp, %one : i64
    func.return %result : i1
  }

  func.func @LyFloat_LeLongBool(%lhs_header: memref<3xi64> {ly.ownership.object_header}, %rhs_header: memref<2xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.float", ly.runtime.method = "__le__"} {
    %rhs_meta, %rhs_digits = func.call @__ly_long_parts(%rhs_header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %d = func.call @LyFloat_AsF64(%lhs_header) : (memref<3xi64>) -> f64
    %cmp = func.call @__ly_long_cmp_f64(%rhs_meta, %rhs_digits, %d) : (memref<2xi64>, memref<?xi32>, f64) -> i64
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %is_ge = arith.cmpi sge, %cmp, %zero : i64
    %is_ordered = arith.cmpi sle, %cmp, %one : i64
    %result = arith.andi %is_ge, %is_ordered : i1
    func.return %result : i1
  }

  func.func @LyFloat_GtLongBool(%lhs_header: memref<3xi64> {ly.ownership.object_header}, %rhs_header: memref<2xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.float", ly.runtime.method = "__gt__"} {
    %rhs_meta, %rhs_digits = func.call @__ly_long_parts(%rhs_header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %d = func.call @LyFloat_AsF64(%lhs_header) : (memref<3xi64>) -> f64
    %cmp = func.call @__ly_long_cmp_f64(%rhs_meta, %rhs_digits, %d) : (memref<2xi64>, memref<?xi32>, f64) -> i64
    %neg_one = arith.constant -1 : i64
    %result = arith.cmpi eq, %cmp, %neg_one : i64
    func.return %result : i1
  }

  func.func @LyFloat_GeLongBool(%lhs_header: memref<3xi64> {ly.ownership.object_header}, %rhs_header: memref<2xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.float", ly.runtime.method = "__ge__"} {
    %rhs_meta, %rhs_digits = func.call @__ly_long_parts(%rhs_header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %d = func.call @LyFloat_AsF64(%lhs_header) : (memref<3xi64>) -> f64
    %cmp = func.call @__ly_long_cmp_f64(%rhs_meta, %rhs_digits, %d) : (memref<2xi64>, memref<?xi32>, f64) -> i64
    %zero = arith.constant 0 : i64
    %result = arith.cmpi sle, %cmp, %zero : i64
    func.return %result : i1
  }

  func.func @LyFloat_EqLongBool(%lhs_header: memref<3xi64> {ly.ownership.object_header}, %rhs_header: memref<2xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.float", ly.runtime.method = "__eq__"} {
    %rhs_meta, %rhs_digits = func.call @__ly_long_parts(%rhs_header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %d = func.call @LyFloat_AsF64(%lhs_header) : (memref<3xi64>) -> f64
    %cmp = func.call @__ly_long_cmp_f64(%rhs_meta, %rhs_digits, %d) : (memref<2xi64>, memref<?xi32>, f64) -> i64
    %zero = arith.constant 0 : i64
    %result = arith.cmpi eq, %cmp, %zero : i64
    func.return %result : i1
  }

  func.func @LyFloat_NeLongBool(%lhs_header: memref<3xi64> {ly.ownership.object_header}, %rhs_header: memref<2xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.float", ly.runtime.method = "__ne__"} {
    %rhs_meta, %rhs_digits = func.call @__ly_long_parts(%rhs_header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %d = func.call @LyFloat_AsF64(%lhs_header) : (memref<3xi64>) -> f64
    %cmp = func.call @__ly_long_cmp_f64(%rhs_meta, %rhs_digits, %d) : (memref<2xi64>, memref<?xi32>, f64) -> i64
    %zero = arith.constant 0 : i64
    %result = arith.cmpi ne, %cmp, %zero : i64
    func.return %result : i1
  }

  // Shortest round-trip repr text for any double, CPython 3.14 rules:
  // fixed notation iff -4 < decpt <= 16, otherwise exponent form with a
  // sign and at least two exponent digits; nan has no sign.
  func.func @LyUnicode_FromF64(%value: f64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]} {
    %buffer = memref.alloca() : memref<40xi8>
    %buffer_view = memref.cast %buffer : memref<40xi8> to memref<?xi8>
    %digit_store = memref.alloca() : memref<24xi8>
    %digit_buf = memref.cast %digit_store : memref<24xi8> to memref<?xi8>
    %len = func.call @__ly_float_repr_fill(%value, %buffer_view, %digit_buf) : (f64, memref<?xi8>, memref<?xi8>) -> i64
    %c0 = arith.constant 0 : index
    %header, %bytes = func.call @__ly_unicode_from_valid_utf8(%buffer_view, %c0, %len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %header, %bytes : memref<2xi64>, memref<?xi8>
  }

  // Writes the repr text into `out` (>= 32 bytes) from index 0 and returns
  // its length. `digit_buf` is 24-byte scratch for the shortest digits.
  func.func private @__ly_float_repr_fill(%value: f64, %out: memref<?xi8>, %digit_buf: memref<?xi8>) -> i64 {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %zero_f = arith.constant 0.0 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ch_minus = arith.constant 45 : i8
    %ch_dot = arith.constant 46 : i8
    %ch_zero = arith.constant 48 : i8
    %ch_e = arith.constant 101 : i8
    %ch_plus = arith.constant 43 : i8

    %isnan = arith.cmpf uno, %value, %value : f64
    cf.cond_br %isnan, ^nan, ^signed

  ^nan:
    %ch_n = arith.constant 110 : i8
    %ch_a = arith.constant 97 : i8
    memref.store %ch_n, %out[%c0] : memref<?xi8>
    memref.store %ch_a, %out[%c1] : memref<?xi8>
    %c2n = arith.constant 2 : index
    memref.store %ch_n, %out[%c2n] : memref<?xi8>
    %three = arith.constant 3 : i64
    func.return %three : i64

  ^signed:
    %bits = arith.bitcast %value : f64 to i64
    %negative = arith.cmpi slt, %bits, %zero : i64
    %prefix = arith.select %negative, %one, %zero : i64
    scf.if %negative {
      memref.store %ch_minus, %out[%c0] : memref<?xi8>
    }
    %pos0 = arith.index_cast %prefix : i64 to index
    %abs = math.absf %value : f64
    %inf = arith.constant 0x7FF0000000000000 : f64
    %isinf = arith.cmpf oeq, %abs, %inf : f64
    cf.cond_br %isinf, ^inf(%pos0 : index), ^finite(%pos0 : index)

  ^inf(%ipos: index):
    %ch_i = arith.constant 105 : i8
    %ch_nn = arith.constant 110 : i8
    %ch_f = arith.constant 102 : i8
    memref.store %ch_i, %out[%ipos] : memref<?xi8>
    %ipos1 = arith.addi %ipos, %c1 : index
    memref.store %ch_nn, %out[%ipos1] : memref<?xi8>
    %ipos2 = arith.addi %ipos1, %c1 : index
    memref.store %ch_f, %out[%ipos2] : memref<?xi8>
    %ilen_idx = arith.addi %ipos2, %c1 : index
    %ilen = arith.index_cast %ilen_idx : index to i64
    func.return %ilen : i64

  ^finite(%fpos: index):
    %iszero = arith.cmpf oeq, %abs, %zero_f : f64
    cf.cond_br %iszero, ^zero(%fpos : index), ^nonzero(%fpos : index)

  ^zero(%zpos: index):
    memref.store %ch_zero, %out[%zpos] : memref<?xi8>
    %zpos1 = arith.addi %zpos, %c1 : index
    memref.store %ch_dot, %out[%zpos1] : memref<?xi8>
    %zpos2 = arith.addi %zpos1, %c1 : index
    memref.store %ch_zero, %out[%zpos2] : memref<?xi8>
    %zlen_idx = arith.addi %zpos2, %c1 : index
    %zlen = arith.index_cast %zlen_idx : index to i64
    func.return %zlen : i64

  ^nonzero(%npos: index):
    %dt:2 = func.call @__ly_dtoa_shortest(%abs, %digit_buf) : (f64, memref<?xi8>) -> (i64, i64)
    %len = func.call @__ly_float_emit_repr_digits(%out, %npos, %digit_buf, %dt#0, %dt#1) : (memref<?xi8>, index, memref<?xi8>, i64, i64) -> i64
    func.return %len : i64
  }

  // Lay out shortest digits per CPython repr: fixed within [1e-4, 1e16),
  // exponential outside, minimum two exponent digits.
  func.func private @__ly_float_emit_repr_digits(%out: memref<?xi8>, %start: index, %digits: memref<?xi8>, %count: i64, %decpt: i64) -> i64 {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %ten = arith.constant 10 : i64
    %hundred = arith.constant 100 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ch_dot = arith.constant 46 : i8
    %ch_zero = arith.constant 48 : i8
    %ch_e = arith.constant 101 : i8
    %ch_plus = arith.constant 43 : i8
    %ch_minus = arith.constant 45 : i8
    %ascii_zero = arith.constant 48 : i64

    %low_bound = arith.constant -3 : i64
    %high_bound = arith.constant 16 : i64
    %ge_low = arith.cmpi sge, %decpt, %low_bound : i64
    %le_high = arith.cmpi sle, %decpt, %high_bound : i64
    %fixed = arith.andi %ge_low, %le_high : i1
    cf.cond_br %fixed, ^fixed_form, ^exp_form

  ^fixed_form:
    %dec_nonpos = arith.cmpi sle, %decpt, %zero : i64
    %pos_fixed = scf.if %dec_nonpos -> (index) {
      // 0.<zeros><digits>
      memref.store %ch_zero, %out[%start] : memref<?xi8>
      %p1 = arith.addi %start, %c1 : index
      memref.store %ch_dot, %out[%p1] : memref<?xi8>
      %p2 = arith.addi %p1, %c1 : index
      %zeros = arith.subi %zero, %decpt : i64
      %zeros_idx = arith.index_cast %zeros : i64 to index
      %p3 = scf.for %i = %c0 to %zeros_idx step %c1 iter_args(%p = %p2) -> (index) {
        memref.store %ch_zero, %out[%p] : memref<?xi8>
        %np = arith.addi %p, %c1 : index
        scf.yield %np : index
      }
      %count_idx = arith.index_cast %count : i64 to index
      %p4 = scf.for %i = %c0 to %count_idx step %c1 iter_args(%p = %p3) -> (index) {
        %ch = memref.load %digits[%i] : memref<?xi8>
        memref.store %ch, %out[%p] : memref<?xi8>
        %np = arith.addi %p, %c1 : index
        scf.yield %np : index
      }
      scf.yield %p4 : index
    } else {
      %dec_ge_count = arith.cmpi sge, %decpt, %count : i64
      %pf = scf.if %dec_ge_count -> (index) {
        // <digits><zeros>.0
        %count_idx = arith.index_cast %count : i64 to index
        %p1 = scf.for %i = %c0 to %count_idx step %c1 iter_args(%p = %start) -> (index) {
          %ch = memref.load %digits[%i] : memref<?xi8>
          memref.store %ch, %out[%p] : memref<?xi8>
          %np = arith.addi %p, %c1 : index
          scf.yield %np : index
        }
        %zeros = arith.subi %decpt, %count : i64
        %zeros_idx = arith.index_cast %zeros : i64 to index
        %p2 = scf.for %i = %c0 to %zeros_idx step %c1 iter_args(%p = %p1) -> (index) {
          memref.store %ch_zero, %out[%p] : memref<?xi8>
          %np = arith.addi %p, %c1 : index
          scf.yield %np : index
        }
        memref.store %ch_dot, %out[%p2] : memref<?xi8>
        %p3 = arith.addi %p2, %c1 : index
        memref.store %ch_zero, %out[%p3] : memref<?xi8>
        %p4 = arith.addi %p3, %c1 : index
        scf.yield %p4 : index
      } else {
        // <digits[:decpt]>.<digits[decpt:]>
        %dec_idx = arith.index_cast %decpt : i64 to index
        %p1 = scf.for %i = %c0 to %dec_idx step %c1 iter_args(%p = %start) -> (index) {
          %ch = memref.load %digits[%i] : memref<?xi8>
          memref.store %ch, %out[%p] : memref<?xi8>
          %np = arith.addi %p, %c1 : index
          scf.yield %np : index
        }
        memref.store %ch_dot, %out[%p1] : memref<?xi8>
        %p2 = arith.addi %p1, %c1 : index
        %count_idx = arith.index_cast %count : i64 to index
        %p3 = scf.for %i = %dec_idx to %count_idx step %c1 iter_args(%p = %p2) -> (index) {
          %ch = memref.load %digits[%i] : memref<?xi8>
          memref.store %ch, %out[%p] : memref<?xi8>
          %np = arith.addi %p, %c1 : index
          scf.yield %np : index
        }
        scf.yield %p3 : index
      }
      scf.yield %pf : index
    }
    %len_fixed = arith.index_cast %pos_fixed : index to i64
    func.return %len_fixed : i64

  ^exp_form:
    // d1[.d2...]e<sign><exponent, at least two digits>
    %first = memref.load %digits[%c0] : memref<?xi8>
    memref.store %first, %out[%start] : memref<?xi8>
    %p1 = arith.addi %start, %c1 : index
    %multi = arith.cmpi sgt, %count, %one : i64
    %p2 = scf.if %multi -> (index) {
      memref.store %ch_dot, %out[%p1] : memref<?xi8>
      %pd = arith.addi %p1, %c1 : index
      %count_idx = arith.index_cast %count : i64 to index
      %pe = scf.for %i = %c1 to %count_idx step %c1 iter_args(%p = %pd) -> (index) {
        %ch = memref.load %digits[%i] : memref<?xi8>
        memref.store %ch, %out[%p] : memref<?xi8>
        %np = arith.addi %p, %c1 : index
        scf.yield %np : index
      }
      scf.yield %pe : index
    } else {
      scf.yield %p1 : index
    }
    memref.store %ch_e, %out[%p2] : memref<?xi8>
    %p3 = arith.addi %p2, %c1 : index
    %exp = arith.subi %decpt, %one : i64
    %exp_neg = arith.cmpi slt, %exp, %zero : i64
    %exp_sign = arith.select %exp_neg, %ch_minus, %ch_plus : i8
    memref.store %exp_sign, %out[%p3] : memref<?xi8>
    %p4 = arith.addi %p3, %c1 : index
    %exp_negated = arith.subi %zero, %exp : i64
    %exp_abs = arith.select %exp_neg, %exp_negated, %exp : i64
    %ge100 = arith.cmpi sge, %exp_abs, %hundred : i64
    %p5 = scf.if %ge100 -> (index) {
      %h = arith.divui %exp_abs, %hundred : i64
      %h_ch_i64 = arith.addi %h, %ascii_zero : i64
      %h_ch = arith.trunci %h_ch_i64 : i64 to i8
      memref.store %h_ch, %out[%p4] : memref<?xi8>
      %np = arith.addi %p4, %c1 : index
      scf.yield %np : index
    } else {
      scf.yield %p4 : index
    }
    %rem100 = arith.remui %exp_abs, %hundred : i64
    %tens = arith.divui %rem100, %ten : i64
    %tens_ch_i64 = arith.addi %tens, %ascii_zero : i64
    %tens_ch = arith.trunci %tens_ch_i64 : i64 to i8
    memref.store %tens_ch, %out[%p5] : memref<?xi8>
    %p6 = arith.addi %p5, %c1 : index
    %units = arith.remui %rem100, %ten : i64
    %units_ch_i64 = arith.addi %units, %ascii_zero : i64
    %units_ch = arith.trunci %units_ch_i64 : i64 to i8
    memref.store %units_ch, %out[%p6] : memref<?xi8>
    %p7 = arith.addi %p6, %c1 : index
    %len_exp = arith.index_cast %p7 : index to i64
    func.return %len_exp : i64
  }

  // The shared float presentation core (types e/E/f/F/g/G/n/%/none over a
  // parsed spec record); int and bool delegate here for float type codes.
  func.func private @__ly_float_format_core(%value: f64, %spec: memref<?xi64>, %tname: memref<?xi8>, %tname_len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %minus_one = arith.constant -1 : i64
    %six = arith.constant 6 : i64
    %zero_f = arith.constant 0.0 : f64
    %true_core = arith.constant true
    %c16_core = arith.constant 16 : i64
    %s0 = arith.constant 0 : index
    %s1 = arith.constant 1 : index
    %s2 = arith.constant 2 : index
    %s3 = arith.constant 3 : index
    %s4_core = arith.constant 4 : index
    %s5 = arith.constant 5 : index
    %s6 = arith.constant 6 : index
    %s7 = arith.constant 7 : index
    %s8 = arith.constant 8 : index
    %s9 = arith.constant 9 : index
    %fill_rec = memref.load %spec[%s0] : memref<?xi64>
    %align_rec = memref.load %spec[%s1] : memref<?xi64>
    %sign_rec = memref.load %spec[%s2] : memref<?xi64>
    %alt_rec = memref.load %spec[%s3] : memref<?xi64>
    %zero_rec = memref.load %spec[%s4_core] : memref<?xi64>
    %width_rec = memref.load %spec[%s5] : memref<?xi64>
    %group_rec = memref.load %spec[%s6] : memref<?xi64>
    %prec_rec = memref.load %spec[%s7] : memref<?xi64>
    %type_rec = memref.load %spec[%s8] : memref<?xi64>
    %z_rec = memref.load %spec[%s9] : memref<?xi64>

    // type resolution
    %cE = arith.constant 69 : i64
    %cF = arith.constant 70 : i64
    %cG = arith.constant 71 : i64
    %ce = arith.constant 101 : i64
    %cf = arith.constant 102 : i64
    %cg = arith.constant 103 : i64
    %cn = arith.constant 110 : i64
    %cr = arith.constant 114 : i64
    %cpct = arith.constant 37 : i64
    %is_E = arith.cmpi eq, %type_rec, %cE : i64
    %is_F = arith.cmpi eq, %type_rec, %cF : i64
    %is_G = arith.cmpi eq, %type_rec, %cG : i64
    %up0 = arith.ori %is_E, %is_F : i1
    %upper = arith.ori %up0, %is_G : i1
    %c32add = arith.constant 32 : i64
    %lowered = arith.addi %type_rec, %c32add : i64
    %tl0 = arith.select %upper, %lowered, %type_rec : i64
    %is_pct = arith.cmpi eq, %tl0, %cpct : i64
    %tl1 = arith.select %is_pct, %cf, %tl0 : i64
    %is_n = arith.cmpi eq, %tl1, %cn : i64
    %tl2 = arith.select %is_n, %cg, %tl1 : i64
    %no_type = arith.cmpi eq, %tl2, %zero : i64
    %no_prec = arith.cmpi eq, %prec_rec, %minus_one : i64
    %repr_mode0 = arith.andi %no_type, %no_prec : i1
    %gdot_mode = scf.if %no_type -> (i1) {
      %has_prec = arith.cmpi ne, %prec_rec, %minus_one : i64
      scf.yield %has_prec : i1
    } else {
      %f0 = arith.constant false
      scf.yield %f0 : i1
    }
    %tl3 = arith.select %repr_mode0, %cr, %tl2 : i64
    %tl = arith.select %gdot_mode, %cg, %tl3 : i64
    %is_e_t = arith.cmpi eq, %tl, %ce : i64
    %is_f_t = arith.cmpi eq, %tl, %cf : i64
    %is_g_t = arith.cmpi eq, %tl, %cg : i64
    %is_r_t = arith.cmpi eq, %tl, %cr : i64
    %v0 = arith.ori %is_e_t, %is_f_t : i1
    %v1 = arith.ori %v0, %is_g_t : i1
    %valid = arith.ori %v1, %is_r_t : i1
    %invalid = arith.xori %valid, %true_core : i1
    scf.if %invalid {
      func.call @__ly_fmt_raise_unknown_code(%type_rec, %tname, %tname_len) : (i64, memref<?xi8>, i64) -> ()
    }
    // ',' or '_' with 'n' is rejected like CPython
    %grouped_in = arith.cmpi ne, %group_rec, %zero : i64
    %n_and_group = arith.andi %is_n, %grouped_in : i1
    scf.if %n_and_group {
      func.call @__ly_fmt_raise_cannot_group(%group_rec, %cn) : (i64, i64) -> ()
    }

    %hundred_f = arith.constant 100.0 : f64
    %scaled = arith.mulf %value, %hundred_f : f64
    %value2 = arith.select %is_pct, %scaled, %value : f64
    %bits = arith.bitcast %value2 : f64 to i64
    %isnan = arith.cmpf uno, %value2, %value2 : f64
    %neg_bit = arith.cmpi slt, %bits, %zero : i64
    %not_nan = arith.xori %isnan, %true_core : i1
    %negative0 = arith.andi %neg_bit, %not_nan : i1
    %abs = math.absf %value2 : f64
    %inf_c = arith.constant 0x7FF0000000000000 : f64
    %isinf = arith.cmpf oeq, %abs, %inf_c : f64
    %special = arith.ori %isnan, %isinf : i1

    // precision resolution
    %prec_or6 = arith.select %no_prec, %six, %prec_rec : i64
    %int_max = arith.constant 2147483647 : i64
    %prec_too_big = arith.cmpi sgt, %prec_or6, %int_max : i64
    scf.if %prec_too_big {
      func.call @__ly_fmt_raise_precision_too_big() : () -> ()
    }
    %g_p = arith.maxsi %prec_or6, %one : i64
    %gdot_dec = arith.select %gdot_mode, %one, %zero : i64
    %g_thr = arith.subi %g_p, %gdot_dec : i64

    // A width past the allocator's reach is MemoryError before either path
    // makes anything (see `__ly_long_format_impl`).
    // (No width is -1, which unsigned is the largest of all.)
    %code_unit_bytes = arith.constant 4 : i64
    %nothing_before = arith.constant 0 : i64
    %width_or_none = arith.maxsi %width_rec, %nothing_before : i64
    func.call @__ly_check_alloc_count(%width_or_none, %code_unit_bytes, %nothing_before) : (i64, i64, i64) -> ()
    cf.cond_br %special, ^special_case, ^finite_case

  ^special_case:
    // 3 letters + optional '%'
    %sp_cap = arith.constant 4 : index
    %sp = memref.alloc(%sp_cap) : memref<?xi32>
    %l_n = arith.constant 110 : i32
    %l_a = arith.constant 97 : i32
    %l_i = arith.constant 105 : i32
    %l_f = arith.constant 102 : i32
    %u_n = arith.constant 78 : i32
    %u_a = arith.constant 65 : i32
    %u_i = arith.constant 73 : i32
    %u_f = arith.constant 70 : i32
    %pct32 = arith.constant 37 : i32
    %ch0 = scf.if %isnan -> (i32) {
      %r = arith.select %upper, %u_n, %l_n : i32
      scf.yield %r : i32
    } else {
      %r = arith.select %upper, %u_i, %l_i : i32
      scf.yield %r : i32
    }
    %ch1 = scf.if %isnan -> (i32) {
      %r = arith.select %upper, %u_a, %l_a : i32
      scf.yield %r : i32
    } else {
      %r = arith.select %upper, %u_n, %l_n : i32
      scf.yield %r : i32
    }
    %ch2 = scf.if %isnan -> (i32) {
      %r = arith.select %upper, %u_n, %l_n : i32
      scf.yield %r : i32
    } else {
      %r = arith.select %upper, %u_f, %l_f : i32
      scf.yield %r : i32
    }
    memref.store %ch0, %sp[%s0] : memref<?xi32>
    memref.store %ch1, %sp[%s1] : memref<?xi32>
    memref.store %ch2, %sp[%s2] : memref<?xi32>
    %three_sp = arith.constant 3 : i64
    %four_sp = arith.constant 4 : i64
    %sp_len = scf.if %is_pct -> (i64) {
      memref.store %pct32, %sp[%s3] : memref<?xi32>
      scf.yield %four_sp : i64
    } else {
      scf.yield %three_sp : i64
    }
    // nan never shows a bit sign; inf keeps it
    %neg_sp = arith.andi %negative0, %isinf : i1
    cf.br ^assemble(%sp, %sp_len, %zero, %neg_sp : memref<?xi32>, i64, i64, i1)

  ^finite_case:
    %dig_cap_idx = arith.constant 800 : index
    %dig_store = memref.alloc(%dig_cap_idx) : memref<?xi8>
    %is_zero_v = arith.cmpf oeq, %abs, %zero_f : f64
    %dtoa:2 = scf.if %is_zero_v -> (i64, i64) {
      %zc8 = arith.constant 48 : i8
      memref.store %zc8, %dig_store[%s0] : memref<?xi8>
      scf.yield %one, %one : i64, i64
    } else {
      %r:2 = scf.if %is_r_t -> (i64, i64) {
        %sh:2 = func.call @__ly_dtoa_shortest(%abs, %dig_store) : (f64, memref<?xi8>) -> (i64, i64)
        scf.yield %sh#0, %sh#1 : i64, i64
      } else {
        %fixed_req = arith.select %is_f_t, %prec_or6, %zero : i64
        %e_req = arith.addi %prec_or6, %one : i64
        %sig_req0 = arith.select %is_e_t, %e_req, %g_p : i64
        %req = arith.select %is_f_t, %fixed_req, %sig_req0 : i64
        %ct:2 = func.call @__ly_dtoa_counted(%abs, %is_f_t, %req, %dig_store) : (f64, i1, i64, memref<?xi8>) -> (i64, i64)
        scf.yield %ct#0, %ct#1 : i64, i64
      }
      scf.yield %r#0, %r#1 : i64, i64
    }

    // 'z': a rounded-to-zero result drops the negative sign
    %z_on = arith.cmpi ne, %z_rec, %zero : i64
    %count_zero = arith.cmpi eq, %dtoa#0, %zero : i64
    %single = arith.cmpi eq, %dtoa#0, %one : i64
    %zc8b = arith.constant 48 : i8
    %first_ch = memref.load %dig_store[%s0] : memref<?xi8>
    %first_zero = arith.cmpi eq, %first_ch, %zc8b : i8
    %single_zero = arith.andi %single, %first_zero : i1
    %is_zero_res = arith.ori %count_zero, %single_zero : i1
    %z_apply = arith.andi %z_on, %is_zero_res : i1
    %not_z = arith.xori %z_apply, %true_core : i1
    %negative = arith.andi %negative0, %not_z : i1

    // body sizing: integer digits + dot + fraction + exponent + '%'
    %dec_c = arith.maxsi %dtoa#1, %one : i64
    %neg_dec = arith.subi %zero, %dtoa#1 : i64
    %neg_dec_c = arith.maxsi %neg_dec, %zero : i64
    %margin = arith.constant 40 : i64
    %cap0 = arith.addi %dec_c, %dtoa#0 : i64
    %cap1 = arith.addi %cap0, %prec_or6 : i64
    %cap2 = arith.addi %cap1, %neg_dec_c : i64
    %body_cap = arith.addi %cap2, %margin : i64
    %body_unit = arith.constant 4 : i64
    func.call @__ly_check_alloc_count(%body_cap, %body_unit, %zero) : (i64, i64, i64) -> ()
    %body_cap_idx = arith.index_cast %body_cap : i64 to index
    %body = memref.alloc(%body_cap_idx) : memref<?xi32>

    %alt_on = arith.cmpi ne, %alt_rec, %zero : i64
    %e_upper = arith.constant 69 : i64
    %e_lower = arith.constant 101 : i64
    %e_cp = arith.select %upper, %e_upper, %e_lower : i64

    %bl:2 = scf.if %is_f_t -> (i64, i64) {
      %r:2 = func.call @__ly_fmt_body_fixed(%body, %dig_store, %dtoa#0, %dtoa#1, %prec_or6, %zero, %alt_on) : (memref<?xi32>, memref<?xi8>, i64, i64, i64, i64, i1) -> (i64, i64)
      scf.yield %r#0, %r#1 : i64, i64
    } else {
      %r2:2 = scf.if %is_e_t -> (i64, i64) {
        %r:2 = func.call @__ly_fmt_body_exp(%body, %dig_store, %dtoa#0, %dtoa#1, %prec_or6, %alt_on, %e_cp) : (memref<?xi32>, memref<?xi8>, i64, i64, i64, i1, i64) -> (i64, i64)
        scf.yield %r#0, %r#1 : i64, i64
      } else {
        // g and repr: pick the form from the decimal exponent
        %thr = arith.select %is_r_t, %c16_core, %g_thr : i64
        %low_thr = arith.constant -3 : i64
        %too_small = arith.cmpi slt, %dtoa#1, %low_thr : i64
        %too_big = arith.cmpi sgt, %dtoa#1, %thr : i64
        %use_exp = arith.ori %too_small, %too_big : i1
        %min_frac = scf.if %is_r_t -> (i64) {
          scf.yield %one : i64
        } else {
          %gm = scf.if %gdot_mode -> (i64) {
            scf.yield %one : i64
          } else {
            scf.yield %zero : i64
          }
          scf.yield %gm : i64
        }
        %r3:2 = scf.if %use_exp -> (i64, i64) {
          %mant_nat = arith.constant -1 : i64
          %alt_mant = arith.subi %g_p, %one : i64
          %use_alt_mant = arith.andi %alt_on, %is_g_t : i1
          %mant = arith.select %use_alt_mant, %alt_mant, %mant_nat : i64
          %force = arith.andi %alt_on, %true_core : i1
          %r:2 = func.call @__ly_fmt_body_exp(%body, %dig_store, %dtoa#0, %dtoa#1, %mant, %force, %e_cp) : (memref<?xi32>, memref<?xi8>, i64, i64, i64, i1, i64) -> (i64, i64)
          scf.yield %r#0, %r#1 : i64, i64
        } else {
          %nat = arith.constant -1 : i64
          %alt_frac = arith.subi %g_p, %dtoa#1 : i64
          %alt_frac_c = arith.maxsi %alt_frac, %zero : i64
          %use_alt_frac = arith.andi %alt_on, %is_g_t : i1
          %frac = arith.select %use_alt_frac, %alt_frac_c, %nat : i64
          %force = arith.andi %alt_on, %true_core : i1
          %r:2 = func.call @__ly_fmt_body_fixed(%body, %dig_store, %dtoa#0, %dtoa#1, %frac, %min_frac, %force) : (memref<?xi32>, memref<?xi8>, i64, i64, i64, i64, i1) -> (i64, i64)
          scf.yield %r#0, %r#1 : i64, i64
        }
        scf.yield %r3#0, %r3#1 : i64, i64
      }
      scf.yield %r2#0, %r2#1 : i64, i64
    }
    %bl_pct = scf.if %is_pct -> (i64) {
      %pos = arith.index_cast %bl#0 : i64 to index
      %pc32 = arith.constant 37 : i32
      memref.store %pc32, %body[%pos] : memref<?xi32>
      %nl = arith.addi %bl#0, %one : i64
      scf.yield %nl : i64
    } else {
      scf.yield %bl#0 : i64
    }
    memref.dealloc %dig_store : memref<?xi8>
    cf.br ^assemble(%body, %bl_pct, %bl#1, %negative : memref<?xi32>, i64, i64, i1)

  ^assemble(%body_a: memref<?xi32>, %body_len_a: i64, %int_len_a: i64, %neg_a: i1):
    %plus_c = arith.constant 43 : i64
    %minus_c = arith.constant 45 : i64
    %space_c = arith.constant 32 : i64
    %is_plus_s = arith.cmpi eq, %sign_rec, %plus_c : i64
    %is_space_s = arith.cmpi eq, %sign_rec, %space_c : i64
    %pos_sign0 = arith.select %is_plus_s, %plus_c, %zero : i64
    %pos_sign = scf.if %is_space_s -> (i64) {
      scf.yield %space_c : i64
    } else {
      scf.yield %pos_sign0 : i64
    }
    %sign_cp = arith.select %neg_a, %minus_c, %pos_sign : i64
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
    %gs3 = arith.constant 3 : i64
    %header, %bytes = func.call @__ly_fmt_render_number(%sign_cp, %zero, %zero, %body_a, %body_len_a, %int_len_a, %group_rec, %gs3, %fill_cp, %align_cp, %width_rec) : (i64, i64, i64, memref<?xi32>, i64, i64, i64, i64, i64, i64, i64) -> (memref<2xi64>, memref<?xi8>)
    memref.dealloc %body_a : memref<?xi32>
    func.return %header, %bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyFloat_Format(%header: memref<3xi64> {ly.ownership.object_header}, %spec_header: memref<2xi64> {ly.ownership.object_header}, %spec_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.float", ly.runtime.method = "__format__", ly.runtime.result_contract = "builtins.str"} {
    %value = func.call @LyFloat_AsF64(%header) : (memref<3xi64>) -> f64
    %spec_store = memref.alloca() : memref<10xi64>
    %spec = memref.cast %spec_store : memref<10xi64> to memref<?xi64>
    %ok = func.call @__ly_fmt_parse_spec(%spec_header, %spec_bytes, %spec) : (memref<2xi64>, memref<?xi8>, memref<?xi64>) -> i1
    %true_ff = arith.constant true
    %bad = arith.xori %ok, %true_ff : i1
    scf.if %bad {
      %names = memref.get_global @__ly_fmt_msg_name_float : memref<5xi8>
      %name = memref.cast %names : memref<5xi8> to memref<?xi8>
      %nlen = arith.constant 5 : i64
      func.call @__ly_fmt_raise_invalid_spec(%spec_header, %spec_bytes, %name, %nlen) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> ()
    }
    %names2 = memref.get_global @__ly_fmt_msg_name_float : memref<5xi8>
    %name2 = memref.cast %names2 : memref<5xi8> to memref<?xi8>
    %nlen2 = arith.constant 5 : i64
    %h, %b = func.call @__ly_float_format_core(%value, %spec, %name2, %nlen2) : (f64, memref<?xi64>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %h, %b : memref<2xi64>, memref<?xi8>
  }

  // ===== impls: float =====
  // Handle words: 0 refcount, 1 layout/destructor family id, 2 the double's
  // bit pattern. Three is unlike either lane width the two-lane form used
  // (memref<2xi64> header, memref<1xf64> payload), which is what keeps
  // verifyReceiverShape's prefix comparison a complete migration gate. There is
  // no C++ mirror of these offsets: unlike a container, a fixed-width inline
  // payload is never addressed from the lowering side -- only through
  // LyFloat_FromF64 / LyFloat_AsF64.
  func.func private @LyFloat_Shape() -> memref<3xi64> attributes {ly.runtime.contract = "builtins.float", ly.runtime.shape}

  func.func @LyFloat_FromF64(%value: f64 {ly.runtime.default_f64 = 0.0 : f64}) -> memref<3xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 2 : i64, ly.runtime.contract = "builtins.float", ly.runtime.initializer = "__new__"} {
    // One entity, one handle: word 2 carries the double as its bit pattern.
    // Why the bits and not an f64-typed lane: the payload is INSIDE the handle,
    // so there is no pointer word for __ly_global_view_f64 to build a
    // descriptor from, and the manifest may not spell an inline cast. The
    // bitcast is a register-level no-op, so the retyping costs nothing.
    %block_bytes = arith.constant 24 : index
    %block = memref.alloc(%block_bytes) {alignment = 16 : i64} : memref<?xi8>
    %self_offset = arith.constant 0 : index
    %header = memref.view %block[%self_offset][] {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<?xi8> to memref<3xi64>
    %one = arith.constant 1 : i64
    %layout_float = arith.constant 2 : i64
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %value_slot = arith.constant 2 : index
    %value_bits = arith.bitcast %value : f64 to i64
    memref.store %one, %header[%refcount_slot] : memref<3xi64>
    memref.store %layout_float, %header[%layout_slot] : memref<3xi64>
    memref.store %value_bits, %header[%value_slot] : memref<3xi64>
    func.return %header : memref<3xi64>
  }

  func.func @LyFloat_DecRef(%header: memref<3xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.float", ly.runtime.deallocator} {
    %storage = memref.cast %header : memref<3xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    memref.dealloc %header : memref<3xi64>
    cf.br ^done

  ^done:
    func.return
  }

  func.func @LyFloat_Init(%header: memref<3xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "builtins.float", ly.runtime.method = "__init__"} {
    func.return
  }

  func.func @LyFloat_AsF64(%header: memref<3xi64> {ly.ownership.object_header}) -> f64 attributes {ly.runtime.contract = "builtins.float", ly.runtime.method = "__float__", ly.runtime.primitive = "unbox.f64"} {
    %value_slot = arith.constant 2 : index
    %value_bits = memref.load %header[%value_slot] : memref<3xi64>
    %value = arith.bitcast %value_bits : i64 to f64
    func.return %value : f64
  }

  // The float a slot's entity word names, as an owned object.
  func.func @LyFloat_FromSlotWord(%slot_view: memref<3xi64>) -> memref<3xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.float"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.float", ly.runtime.primitive = "from_slot_word"} {
    %word_idx = memref.extract_aligned_pointer_as_index %slot_view : memref<3xi64> -> index
    %address = arith.index_cast %word_idx : index to i64
    %word = func.call @__ly_slot_word_from_view_address(%address) : (i64) -> i64
    %immediate = func.call @__ly_slot_word_is_immediate(%word) : (i64) -> i1
    %header = scf.if %immediate -> (memref<3xi64>) {
      %bits = func.call @__ly_float_from_immediate(%word) : (i64) -> i64
      %value = arith.bitcast %bits : i64 to f64
      %fresh = func.call @LyFloat_FromF64(%value) : (f64) -> memref<3xi64>
      scf.yield %fresh : memref<3xi64>
    } else {
      func.call @__ly_handle_retain_raw(%word) : (i64) -> ()
      %three = arith.constant 3 : i64
      %view = func.call @__ly_global_view_i64(%word, %three) : (i64, i64) -> memref<?xi64>
      %held = memref.cast %view : memref<?xi64> to memref<3xi64>
      scf.yield %held : memref<3xi64>
    }
    func.return %header : memref<3xi64>
  }

  // The f64 a slot's entity word names.
  func.func @LyFloat_SlotWordAsF64(%word: i64) -> f64 attributes {ly.runtime.contract = "builtins.float", ly.runtime.primitive = "slot_word_as_f64"} {
    %immediate = func.call @__ly_slot_word_is_immediate(%word) : (i64) -> i1
    %bits = scf.if %immediate -> (i64) {
      %b = func.call @__ly_float_from_immediate(%word) : (i64) -> i64
      scf.yield %b : i64
    } else {
      %ptr = llvm.inttoptr %word : i64 to !llvm.ptr
      %slot = llvm.getelementptr %ptr[2] : (!llvm.ptr) -> !llvm.ptr, i64
      %b = llvm.load %slot : !llvm.ptr -> i64
      scf.yield %b : i64
    }
    %value = arith.bitcast %bits : i64 to f64
    func.return %value : f64
  }

  // The entity word a slot stores for a float it is handed as an f64.
  func.func @LyFloat_SlotWordFromF64(%value: f64) -> i64 attributes {ly.runtime.contract = "builtins.float", ly.runtime.primitive = "slot_word_from_f64"} {
    %bits = arith.bitcast %value : f64 to i64
    %fits = func.call @__ly_float_immediate_fits(%bits) : (i64) -> i1
    %word = scf.if %fits -> (i64) {
      %w = func.call @__ly_float_to_immediate(%bits) : (i64) -> i64
      scf.yield %w : i64
    } else {
      %header = func.call @LyFloat_FromF64(%value) : (f64) -> memref<3xi64>
      %idx = memref.extract_aligned_pointer_as_index %header : memref<3xi64> -> index
      %w = arith.index_cast %idx : index to i64
      scf.yield %w : i64
    }
    func.return %word : i64
  }

  // The entity word a slot stores for a float OBJECT whose reference the
  // slot has just been given (the lowering's aggregate retain): the immediate
  // when the value has one -- and then that reference is dropped again, since
  // the slot holds no object -- else the object's address.
  // ⛔ Called only after the retain: before it, the drop could free an object
  // the frame still holds.
  // The f64 a slot names, read through the view a slot read builds (see
  // `LyFloat_FromSlotWord`) -- the lane of a read that makes no object.
  func.func @LyFloat_ReadSlotF64(%slot_view: memref<3xi64>) -> f64 attributes {ly.runtime.contract = "builtins.float", ly.runtime.primitive = "read_slot_f64"} {
    %word_idx = memref.extract_aligned_pointer_as_index %slot_view : memref<3xi64> -> index
    %address = arith.index_cast %word_idx : index to i64
    %word = func.call @__ly_slot_word_from_view_address(%address) : (i64) -> i64
    %value = func.call @LyFloat_SlotWordAsF64(%word) : (i64) -> f64
    func.return %value : f64
  }

  func.func @LyFloat_SlotWordTakingRef(%header: memref<3xi64> {ly.ownership.object_header}) -> i64 attributes {ly.runtime.contract = "builtins.float", ly.runtime.primitive = "slot_word_taking_ref"} {
    %value_slot = arith.constant 2 : index
    %bits = memref.load %header[%value_slot] : memref<3xi64>
    %fits = func.call @__ly_float_immediate_fits(%bits) : (i64) -> i1
    %word = scf.if %fits -> (i64) {
      func.call @LyFloat_DecRef(%header) : (memref<3xi64>) -> ()
      %w = func.call @__ly_float_to_immediate(%bits) : (i64) -> i64
      scf.yield %w : i64
    } else {
      %idx = memref.extract_aligned_pointer_as_index %header : memref<3xi64> -> index
      %w = arith.index_cast %idx : index to i64
      scf.yield %w : i64
    }
    func.return %word : i64
  }

  func.func @LyFloat_Add(%lhs_header: memref<3xi64> {ly.ownership.object_header}, %rhs_header: memref<3xi64> {ly.ownership.object_header}) -> memref<3xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.float"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.float", ly.runtime.method = "__add__"} {
    %lhs = func.call @LyFloat_AsF64(%lhs_header) : (memref<3xi64>) -> f64
    %rhs = func.call @LyFloat_AsF64(%rhs_header) : (memref<3xi64>) -> f64
    %value = arith.addf %lhs, %rhs : f64
    %out_header = func.call @LyFloat_FromF64(%value) : (f64) -> memref<3xi64>
    func.return %out_header : memref<3xi64>
  }

  func.func @LyFloat_Sub(%lhs_header: memref<3xi64> {ly.ownership.object_header}, %rhs_header: memref<3xi64> {ly.ownership.object_header}) -> memref<3xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.float"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.float", ly.runtime.method = "__sub__"} {
    %lhs = func.call @LyFloat_AsF64(%lhs_header) : (memref<3xi64>) -> f64
    %rhs = func.call @LyFloat_AsF64(%rhs_header) : (memref<3xi64>) -> f64
    %value = arith.subf %lhs, %rhs : f64
    %out_header = func.call @LyFloat_FromF64(%value) : (f64) -> memref<3xi64>
    func.return %out_header : memref<3xi64>
  }

  func.func @LyFloat_Mul(%lhs_header: memref<3xi64> {ly.ownership.object_header}, %rhs_header: memref<3xi64> {ly.ownership.object_header}) -> memref<3xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.float"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.float", ly.runtime.method = "__mul__"} {
    %lhs = func.call @LyFloat_AsF64(%lhs_header) : (memref<3xi64>) -> f64
    %rhs = func.call @LyFloat_AsF64(%rhs_header) : (memref<3xi64>) -> f64
    %value = arith.mulf %lhs, %rhs : f64
    %out_header = func.call @LyFloat_FromF64(%value) : (f64) -> memref<3xi64>
    func.return %out_header : memref<3xi64>
  }

  func.func @LyFloat_TrueDiv(%lhs_header: memref<3xi64> {ly.ownership.object_header}, %rhs_header: memref<3xi64> {ly.ownership.object_header}) -> memref<3xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.float"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.float", ly.runtime.method = "__truediv__"} {
    %lhs = func.call @LyFloat_AsF64(%lhs_header) : (memref<3xi64>) -> f64
    %rhs = func.call @LyFloat_AsF64(%rhs_header) : (memref<3xi64>) -> f64
    %value = func.call @LyFloat_DivF64(%lhs, %rhs) : (f64, f64) -> f64
    %out_header = func.call @LyFloat_FromF64(%value) : (f64) -> memref<3xi64>
    func.return %out_header : memref<3xi64>
  }

  // float_div on two doubles: the lane's `/`, which needs no object for
  // either operand to raise.
  func.func @LyFloat_DivF64(%lhs: f64, %rhs: f64) -> f64 attributes {ly.runtime.contract = "builtins.float", ly.runtime.primitive = "truediv.f64"} {
    // Not arith.divf alone: IEEE would yield inf/nan for a zero divisor,
    // where CPython float_div raises. Returning inf is the one outcome the
    // project forbids -- a wrong value with no diagnostic.
    %zero = arith.constant 0.0 : f64
    %divisor_zero = arith.cmpf oeq, %rhs, %zero : f64
    cf.cond_br %divisor_zero, ^by_zero, ^divide

  ^by_zero:
    func.call @__ly_long_raise_division_by_zero() : () -> ()
    cf.br ^divide

  ^divide:
    %value = arith.divf %lhs, %rhs : f64
    func.return %value : f64
  }

  // C fmod, the only piece of CPython's float_divmod that arith/math cannot
  // express: x - trunc(x/y)*y loses the exactness fmod guarantees once the
  // quotient exceeds the significand, and MLIR's math dialect has no fmod.
  func.func private @fmod(%x: f64, %y: f64) -> f64

  // CPython float_divmod (Objects/floatobject.c) verbatim: fmod for the
  // remainder, then the sign fixup that makes the remainder follow the
  // divisor, then the quotient snap. Both __floordiv__ and __mod__ read this
  // one helper because CPython derives float_floor_div and float_rem from the
  // same computation -- splitting them would let the pair disagree.
  func.func private @__ly_float_divmod(%vx: f64, %wx: f64) -> (f64, f64) {
    %zero = arith.constant 0.0 : f64
    %one = arith.constant 1.0 : f64
    %half = arith.constant 0.5 : f64

    %mod_raw = func.call @fmod(%vx, %wx) : (f64, f64) -> f64
    %div_raw = arith.subf %vx, %mod_raw : f64
    %div_scaled = arith.divf %div_raw, %wx : f64

    %mod_nonzero = arith.cmpf one, %mod_raw, %zero : f64
    %divisor_negative = arith.cmpf olt, %wx, %zero : f64
    %mod_negative = arith.cmpf olt, %mod_raw, %zero : f64
    %signs_differ = arith.xori %divisor_negative, %mod_negative : i1
    %needs_fixup = arith.andi %mod_nonzero, %signs_differ : i1

    %mod_fixed = arith.addf %mod_raw, %wx : f64
    %div_fixed = arith.subf %div_scaled, %one : f64
    // A zero remainder's sign is platform-dependent out of fmod; CPython
    // forces the divisor's sign so that -0.0 % 2.0 matches.
    %mod_zero_signed = math.copysign %zero, %wx : f64
    %mod_maybe = arith.select %needs_fixup, %mod_fixed, %mod_raw : f64
    %mod = arith.select %mod_nonzero, %mod_maybe, %mod_zero_signed : f64
    %div = arith.select %needs_fixup, %div_fixed, %div_scaled : f64

    %div_nonzero = arith.cmpf one, %div, %zero : f64
    %floor = math.floor %div : f64
    %fraction = arith.subf %div, %floor : f64
    %past_half = arith.cmpf ogt, %fraction, %half : f64
    %floor_up = arith.addf %floor, %one : f64
    %floordiv_nonzero = arith.select %past_half, %floor_up, %floor : f64
    %quotient = arith.divf %vx, %wx : f64
    %floordiv_zero = math.copysign %zero, %quotient : f64
    %floordiv = arith.select %div_nonzero, %floordiv_nonzero, %floordiv_zero : f64

    func.return %floordiv, %mod : f64, f64
  }

  func.func @LyFloat_FloorDiv(%lhs_header: memref<3xi64> {ly.ownership.object_header}, %rhs_header: memref<3xi64> {ly.ownership.object_header}) -> memref<3xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.float", ly.runtime.method = "__floordiv__", ly.runtime.result_contract = "builtins.float"} {
    %lhs = func.call @LyFloat_AsF64(%lhs_header) : (memref<3xi64>) -> f64
    %rhs = func.call @LyFloat_AsF64(%rhs_header) : (memref<3xi64>) -> f64
    %zero = arith.constant 0.0 : f64
    %divisor_zero = arith.cmpf oeq, %rhs, %zero : f64
    cf.cond_br %divisor_zero, ^by_zero, ^divide

  ^by_zero:
    func.call @__ly_long_raise_division_by_zero() : () -> ()
    cf.br ^divide

  ^divide:
    %floordiv, %mod = func.call @__ly_float_divmod(%lhs, %rhs) : (f64, f64) -> (f64, f64)
    %out_header = func.call @LyFloat_FromF64(%floordiv) : (f64) -> memref<3xi64>
    func.return %out_header : memref<3xi64>
  }

  func.func @LyFloat_Mod(%lhs_header: memref<3xi64> {ly.ownership.object_header}, %rhs_header: memref<3xi64> {ly.ownership.object_header}) -> memref<3xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.float", ly.runtime.method = "__mod__", ly.runtime.result_contract = "builtins.float"} {
    %lhs = func.call @LyFloat_AsF64(%lhs_header) : (memref<3xi64>) -> f64
    %rhs = func.call @LyFloat_AsF64(%rhs_header) : (memref<3xi64>) -> f64
    %zero = arith.constant 0.0 : f64
    %divisor_zero = arith.cmpf oeq, %rhs, %zero : f64
    cf.cond_br %divisor_zero, ^by_zero, ^divide

  ^by_zero:
    func.call @__ly_long_raise_division_by_zero() : () -> ()
    cf.br ^divide

  ^divide:
    %floordiv, %mod = func.call @__ly_float_divmod(%lhs, %rhs) : (f64, f64) -> (f64, f64)
    %out_header = func.call @LyFloat_FromF64(%mod) : (f64) -> memref<3xi64>
    func.return %out_header : memref<3xi64>
  }

  // round(x) and round(x, n) are separate manifest overloads because CPython
  // returns different *types*: float___round___impl narrows to int when
  // ndigits is absent and stays float otherwise. One declaration with a
  // default, as int.__round__ uses, cannot express that split.
  func.func @LyFloat_RoundToInt(%header: memref<3xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.float", ly.runtime.method = "__round__", ly.runtime.result_contract = "builtins.int"} {
    %value = func.call @LyFloat_AsF64(%header) : (memref<3xi64>) -> f64
    // round-half-to-even, then reuse float.__int__ for the narrowing so the
    // nan/inf and >2^63 paths raise exactly what int(x) raises. The temporary
    // exists only because __int__ takes an object, not an f64.
    %rounded = math.roundeven %value : f64
    %tmp_header = func.call @LyFloat_FromF64(%rounded) : (f64) -> memref<3xi64>
    %h = func.call @LyFloat_Int(%tmp_header) : (memref<3xi64>) -> memref<2xi64>
    func.call @LyFloat_DecRef(%tmp_header) : (memref<3xi64>) -> ()
    func.return %h : memref<2xi64>
  }

  func.func @LyFloat_RoundNdigits(%header: memref<3xi64> {ly.ownership.object_header}, %ndigits_header: memref<2xi64> {ly.ownership.object_header}) -> memref<3xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.float", ly.runtime.method = "__round__", ly.runtime.result_contract = "builtins.float"} {
    %ndigits_meta, %ndigits_digits = func.call @__ly_long_parts(%ndigits_header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %value = func.call @LyFloat_AsF64(%header) : (memref<3xi64>) -> f64
    // WHY NOT LyLong_AsI64: it raises on a wider value, which is right for
    // int(x) and wrong here. float___round___impl reads ndigits with
    // PyNumber_AsSsize_t(o_ndigits, NULL), and the NULL says CLIP rather than
    // raise, so round(1.5, 10**30) is round(1.5, SSIZE_MAX) -- 1.5 -- and the
    // NDIGITS_MAX/MIN guards below turn every clipped value into a passthrough.
    %ndigits_fits = func.call @__ly_long_view_fits_i64(%ndigits_meta, %ndigits_digits) : (memref<2xi64>, memref<?xi32>) -> i1
    %ndigits_exact = func.call @__ly_long_view_as_i64(%ndigits_meta, %ndigits_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %ndigits_sign_slot = arith.constant 0 : index
    %ndigits_zero = arith.constant 0 : i64
    %ndigits_sign = memref.load %ndigits_meta[%ndigits_sign_slot] : memref<2xi64>
    %ndigits_negative = arith.cmpi slt, %ndigits_sign, %ndigits_zero : i64
    %ndigits_max = arith.constant 9223372036854775807 : i64
    %ndigits_min = arith.constant -9223372036854775808 : i64
    %ndigits_clipped = arith.select %ndigits_negative, %ndigits_min, %ndigits_max : i64
    %ndigits = arith.select %ndigits_fits, %ndigits_exact, %ndigits_clipped : i64
    %rounded = func.call @__ly_float_round_ndigits(%value, %ndigits) : (f64, i64) -> f64
    %out_header = func.call @LyFloat_FromF64(%rounded) : (f64) -> memref<3xi64>
    func.return %out_header : memref<3xi64>
  }

  // CPython's float___round___impl + double_round (Objects/floatobject.c,
  // Python/pystrtod.c): round the value to a DECIMAL string with `ndigits`
  // places, then parse that string back. Both halves already exist here --
  // @__ly_dtoa_counted with fixed_mode IS _Py_dg_dtoa mode 3 (it is what
  // format(x, '.Nf') calls), and the parse is the one direction libc does
  // correctly-roundedly on every host this targets.
  //
  // WHY NOT keep scaling by 10**ndigits: the multiply commits to a digit
  // before the rounding rule is consulted, and that is one cause with three
  // symptoms. It manufactures ties the number does not have (2.675 * 100 IS
  // 267.5 in binary64, so half-to-even went up to 2.68 where CPython says
  // 2.67 -- the fma-residual override this block used to carry existed only
  // to paper over that one shape). It loses the last digit wherever
  // x * 10**n / 10**n is not the identity (round(234743633112.0, 8) gave
  // 234743633112.00003). And it overflows at exponents CPython never reaches
  // (round(1e300, 15) gave inf). No guard on the binary path removes them;
  // the decimal round-trip removes all three, and the residual override with
  // them.
  //
  // WHY NOT snprintf for the text, as the earlier note here proposed:
  // "%s0%se%d" is variadic and func.func cannot express varargs, and the
  // digits are already in hand from the dtoa above. The emitted text carries
  // no radix character, so LC_NUMERIC cannot reach strtod either.
  memref.global "private" constant @__ly_float_msg_round_overflow : memref<36xi8> = dense<[114, 111, 117, 110, 100, 101, 100, 32, 118, 97, 108, 117, 101, 32, 116, 111, 111, 32, 108, 97, 114, 103, 101, 32, 116, 111, 32, 114, 101, 112, 114, 101, 115, 101, 110, 116]>

  func.func private @__ly_float_raise_round_overflow() {
    %class_id = arith.constant 104 : i64
    %length = arith.constant 36 : i64
    %message_static = memref.get_global @__ly_float_msg_round_overflow : memref<36xi8>
    %message = memref.cast %message_static : memref<36xi8> to memref<?xi8>
    func.call @__ly_raise_static_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @strtod(%text: !llvm.ptr, %end: !llvm.ptr) -> f64

  func.func private @__ly_float_round_ndigits(%x: f64, %ndigits: i64) -> f64 {
    %zero_f = arith.constant 0.0 : f64
    %inf = arith.constant 0x7FF0000000000000 : f64
    %zero64 = arith.constant 0 : i64
    %ten = arith.constant 10 : i64
    %hundred = arith.constant 100 : i64
    %ascii_zero = arith.constant 48 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %ch_minus = arith.constant 45 : i8
    %ch_zero = arith.constant 48 : i8
    %ch_e = arith.constant 101 : i8
    %ch_nul = arith.constant 0 : i8

    %is_nan = arith.cmpf uno, %x, %x : f64
    %abs_x = math.absf %x : f64
    %is_inf = arith.cmpf oeq, %abs_x, %inf : f64
    %is_zero = arith.cmpf oeq, %abs_x, %zero_f : f64
    // NDIGITS_MAX = (DBL_MANT_DIG - DBL_MIN_EXP) * 0.30103 == 323: past it the
    // cutoff sits below the last bit of every double, so x rounds to itself.
    // Zero is excluded here because @__ly_dtoa_counted estimates its first
    // digit with log10, which has no answer at 0; CPython's dtoa special-cases
    // it to the same passthrough.
    %hi_digits = arith.constant 323 : i64
    %above = arith.cmpi sgt, %ndigits, %hi_digits : i64
    %special = arith.ori %is_nan, %is_inf : i1
    %special_or_zero = arith.ori %special, %is_zero : i1
    %passthrough = arith.ori %special_or_zero, %above : i1
    cf.cond_br %passthrough, ^pass, ^check_low

  ^pass:
    func.return %x : f64

  ^check_low:
    // NDIGITS_MIN = -(DBL_MAX_EXP + 1) * 0.30103 == -308: below it every
    // finite x is under half the cutoff, so the answer is a zero that keeps
    // x's sign -- `0.0 * x`, as float___round___impl writes it. Returning x
    // here (what this function used to do) is the same defect at the other
    // end of the range.
    %lo_digits = arith.constant -308 : i64
    %below = arith.cmpi slt, %ndigits, %lo_digits : i64
    cf.cond_br %below, ^to_zero, ^decimal

  ^to_zero:
    %signed_zero = arith.mulf %zero_f, %x : f64
    func.return %signed_zero : f64

  ^decimal:
    // 800 bytes matches the format path's digit buffer. The reachable bound
    // is decpt + ndigits <= 309 + 323, and the exact expansion runs out well
    // before that, so the cap inside @__ly_dtoa_counted is never the limit.
    %digit_store = memref.alloca() : memref<800xi8>
    %digits = memref.cast %digit_store : memref<800xi8> to memref<?xi8>
    %dt:2 = func.call @__ly_dtoa_counted(%abs_x, %true, %ndigits, %digits) : (f64, i1, i64, memref<?xi8>) -> (i64, i64)

    // "[-]0<digits>e[-]<three digits>" is CPython's "%s0%se%d" with the
    // exponent zero-padded: |decpt - count| <= 323 on every path that gets
    // here, and strtod does not care about the width. A zero result arrives
    // as count == 0 and needs no special case: "-0e-002" parses to -0.0.
    %text_store = memref.alloca() : memref<800xi8>
    %bits = arith.bitcast %x : f64 to i64
    %negative = arith.cmpi slt, %bits, %zero64 : i64
    %after_sign = scf.if %negative -> (index) {
      memref.store %ch_minus, %text_store[%c0] : memref<800xi8>
      scf.yield %c1 : index
    } else {
      scf.yield %c0 : index
    }
    memref.store %ch_zero, %text_store[%after_sign] : memref<800xi8>
    %after_zero = arith.addi %after_sign, %c1 : index
    %count_idx = arith.index_cast %dt#0 : i64 to index
    %after_digits = scf.for %i = %c0 to %count_idx step %c1 iter_args(%p = %after_zero) -> (index) {
      %ch = memref.load %digits[%i] : memref<?xi8>
      memref.store %ch, %text_store[%p] : memref<800xi8>
      %np = arith.addi %p, %c1 : index
      scf.yield %np : index
    }
    memref.store %ch_e, %text_store[%after_digits] : memref<800xi8>
    %after_e = arith.addi %after_digits, %c1 : index
    %exp10 = arith.subi %dt#1, %dt#0 : i64
    %exp_neg = arith.cmpi slt, %exp10, %zero64 : i64
    %exp_flip = arith.subi %zero64, %exp10 : i64
    %exp_abs = arith.select %exp_neg, %exp_flip, %exp10 : i64
    %exp_start = scf.if %exp_neg -> (index) {
      memref.store %ch_minus, %text_store[%after_e] : memref<800xi8>
      %np = arith.addi %after_e, %c1 : index
      scf.yield %np : index
    } else {
      scf.yield %after_e : index
    }
    %huns = arith.divui %exp_abs, %hundred : i64
    %rest = arith.remui %exp_abs, %hundred : i64
    %tens = arith.divui %rest, %ten : i64
    %ones = arith.remui %rest, %ten : i64
    %huns_i = arith.addi %huns, %ascii_zero : i64
    %huns_ch = arith.trunci %huns_i : i64 to i8
    memref.store %huns_ch, %text_store[%exp_start] : memref<800xi8>
    %tens_at = arith.addi %exp_start, %c1 : index
    %tens_i = arith.addi %tens, %ascii_zero : i64
    %tens_ch = arith.trunci %tens_i : i64 to i8
    memref.store %tens_ch, %text_store[%tens_at] : memref<800xi8>
    %ones_at = arith.addi %tens_at, %c1 : index
    %ones_i = arith.addi %ones, %ascii_zero : i64
    %ones_ch = arith.trunci %ones_i : i64 to i8
    memref.store %ones_ch, %text_store[%ones_at] : memref<800xi8>
    %nul_at = arith.addi %ones_at, %c1 : index
    memref.store %ch_nul, %text_store[%nul_at] : memref<800xi8>

    %text_index = memref.extract_aligned_pointer_as_index %text_store : memref<800xi8> -> index
    %text_word = arith.index_cast %text_index : index to i64
    %text_ptr = llvm.inttoptr %text_word : i64 to !llvm.ptr
    %null_ptr = llvm.inttoptr %zero64 : i64 to !llvm.ptr
    %parsed = func.call @strtod(%text_ptr, %null_ptr) : (!llvm.ptr, !llvm.ptr) -> f64

    // CPython raises on `errno == ERANGE && fabs(rounded) >= 1.0`. WHY NOT
    // read errno: it needs a per-platform accessor (__error / __errno_location)
    // and the test is exactly equivalent without it -- the text handed to
    // strtod is a finite decimal, so an infinity out of it can only be the
    // overflow that condition names. ERANGE on underflow stays invisible here
    // the same way it does in CPython.
    %parsed_abs = math.absf %parsed : f64
    %overflowed = arith.cmpf oeq, %parsed_abs, %inf : f64
    cf.cond_br %overflowed, ^too_large, ^done

  ^too_large:
    func.call @__ly_float_raise_round_overflow() : () -> ()
    cf.br ^done

  ^done:
    func.return %parsed : f64
  }

  // float ** float via libm pow (CPython float_pow also defers to C pow).
  // 0.0 ** negative raises ZeroDivisionError like CPython; a negative base
  // with a non-integral exponent yields a complex in CPython, which is not
  // implemented, so it raises instead (deviation until complex lands, R6).
  func.func @LyFloat_Pow(%lhs_header: memref<3xi64> {ly.ownership.object_header}, %rhs_header: memref<3xi64> {ly.ownership.object_header}) -> memref<3xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.float"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.float", ly.runtime.method = "__pow__"} {
    %base = func.call @LyFloat_AsF64(%lhs_header) : (memref<3xi64>) -> f64
    %exponent = func.call @LyFloat_AsF64(%rhs_header) : (memref<3xi64>) -> f64
    %zero = arith.constant 0.0 : f64
    %base_zero = arith.cmpf oeq, %base, %zero : f64
    %exp_negative = arith.cmpf olt, %exponent, %zero : f64
    %zero_negative = arith.andi %base_zero, %exp_negative : i1
    cf.cond_br %zero_negative, ^zero_to_negative, ^check_fractional

  ^zero_to_negative:
    func.call @__ly_long_raise_zero_negative_power() : () -> ()
    %dummy0 = arith.constant 0.0 : f64
    %zh = func.call @LyFloat_FromF64(%dummy0) : (f64) -> memref<3xi64>
    func.return %zh : memref<3xi64>

  ^check_fractional:
    %base_negative = arith.cmpf olt, %base, %zero : f64
    %exp_trunc = math.trunc %exponent : f64
    %exp_fractional = arith.cmpf one, %exp_trunc, %exponent : f64
    %complex_result = arith.andi %base_negative, %exp_fractional : i1
    cf.cond_br %complex_result, ^fractional_negative, ^pow

  ^fractional_negative:
    func.call @__ly_long_raise_fractional_power_negative() : () -> ()
    %dummy1 = arith.constant 0.0 : f64
    %fh = func.call @LyFloat_FromF64(%dummy1) : (f64) -> memref<3xi64>
    func.return %fh : memref<3xi64>

  ^pow:
    %value = math.powf %base, %exponent : f64
    %out_header = func.call @LyFloat_FromF64(%value) : (f64) -> memref<3xi64>
    func.return %out_header : memref<3xi64>
  }

  // Comparisons follow IEEE ordered semantics (NaN compares false), matching
  // CPython's float comparisons.
  func.func @LyFloat_EqBool(%lhs_header: memref<3xi64> {ly.ownership.object_header}, %rhs_header: memref<3xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.float", ly.runtime.method = "__eq__"} {
    %lhs = func.call @LyFloat_AsF64(%lhs_header) : (memref<3xi64>) -> f64
    %rhs = func.call @LyFloat_AsF64(%rhs_header) : (memref<3xi64>) -> f64
    %result = arith.cmpf oeq, %lhs, %rhs : f64
    func.return %result : i1
  }

  func.func @LyFloat_NeBool(%lhs_header: memref<3xi64> {ly.ownership.object_header}, %rhs_header: memref<3xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.float", ly.runtime.method = "__ne__"} {
    %lhs = func.call @LyFloat_AsF64(%lhs_header) : (memref<3xi64>) -> f64
    %rhs = func.call @LyFloat_AsF64(%rhs_header) : (memref<3xi64>) -> f64
    %result = arith.cmpf une, %lhs, %rhs : f64
    func.return %result : i1
  }

  func.func @LyFloat_LtBool(%lhs_header: memref<3xi64> {ly.ownership.object_header}, %rhs_header: memref<3xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.float", ly.runtime.method = "__lt__"} {
    %lhs = func.call @LyFloat_AsF64(%lhs_header) : (memref<3xi64>) -> f64
    %rhs = func.call @LyFloat_AsF64(%rhs_header) : (memref<3xi64>) -> f64
    %result = arith.cmpf olt, %lhs, %rhs : f64
    func.return %result : i1
  }

  func.func @LyFloat_LeBool(%lhs_header: memref<3xi64> {ly.ownership.object_header}, %rhs_header: memref<3xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.float", ly.runtime.method = "__le__"} {
    %lhs = func.call @LyFloat_AsF64(%lhs_header) : (memref<3xi64>) -> f64
    %rhs = func.call @LyFloat_AsF64(%rhs_header) : (memref<3xi64>) -> f64
    %result = arith.cmpf ole, %lhs, %rhs : f64
    func.return %result : i1
  }

  func.func @LyFloat_GtBool(%lhs_header: memref<3xi64> {ly.ownership.object_header}, %rhs_header: memref<3xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.float", ly.runtime.method = "__gt__"} {
    %lhs = func.call @LyFloat_AsF64(%lhs_header) : (memref<3xi64>) -> f64
    %rhs = func.call @LyFloat_AsF64(%rhs_header) : (memref<3xi64>) -> f64
    %result = arith.cmpf ogt, %lhs, %rhs : f64
    func.return %result : i1
  }

  func.func @LyFloat_GeBool(%lhs_header: memref<3xi64> {ly.ownership.object_header}, %rhs_header: memref<3xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.float", ly.runtime.method = "__ge__"} {
    %lhs = func.call @LyFloat_AsF64(%lhs_header) : (memref<3xi64>) -> f64
    %rhs = func.call @LyFloat_AsF64(%rhs_header) : (memref<3xi64>) -> f64
    %result = arith.cmpf oge, %lhs, %rhs : f64
    func.return %result : i1
  }

  func.func @LyFloat_Bool(%header: memref<3xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.float", ly.runtime.method = "__bool__"} {
    %value = func.call @LyFloat_AsF64(%header) : (memref<3xi64>) -> f64
    %zero = arith.constant 0.0 : f64
    %truth = arith.cmpf une, %value, %zero : f64
    func.return %truth : i1
  }

  // str(float) == repr(float) in CPython; delegate.
  func.func @LyFloat_Str(%header: memref<3xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.float", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %str_header, %str_bytes = func.call @LyFloat_Repr(%header) : (memref<3xi64>) -> (memref<2xi64>, memref<?xi8>)
    func.return %str_header, %str_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyFloat_Repr(%header: memref<3xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.float", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %value = func.call @LyFloat_AsF64(%header) : (memref<3xi64>) -> f64
    %str_header, %str_bytes = func.call @LyUnicode_FromF64(%value) : (f64) -> (memref<2xi64>, memref<?xi8>)
    func.return %str_header, %str_bytes : memref<2xi64>, memref<?xi8>
  }

  // float.__hash__: CPython's modular reduction over the Mersenne prime
  // 2^61-1, computed on the IEEE-754 fields directly (value = mant * 2^exp;
  // multiplying by 2^k mod 2^61-1 is a 61-bit rotation), so hash(1.0) ==
  // hash(1) holds against LyLong_Hash's digit rotation.
  //
  // ⛔ TAKEN AS RAW BITS PLUS AN IDENTITY WORD rather than as a float object,
  // because complex hashes its two components with the SAME rule and CPython's
  // hash(complex(2, 0)) == hash(2) depends on it being the same rule. The
  // identity word is only read for a NaN, which CPython hashes by object.
  func.func private @__ly_float_hash_bits(%bits: i64, %ident: i64) -> i64 {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0 : i64
    %true_ident = arith.constant true
    %c52 = arith.constant 52 : i64
    %c63 = arith.constant 63 : i64
    %exp_mask = arith.constant 2047 : i64
    %mant_mask = arith.constant 4503599627370495 : i64
    %implicit_bit = arith.constant 4503599627370496 : i64
    %exp_field = arith.shrui %bits, %c52 : i64
    %exp_bits = arith.andi %exp_field, %exp_mask : i64
    %mant_bits = arith.andi %bits, %mant_mask : i64
    %sign_bit = arith.shrui %bits, %c63 : i64
    %is_special = arith.cmpi eq, %exp_bits, %exp_mask : i64
    %result = scf.if %is_special -> (i64) {
      %is_nan = arith.cmpi ne, %mant_bits, %zero : i64
      %special = scf.if %is_nan -> (i64) {
        // NaN: identity hash of the boxed object (CPython 3.10+).
        %p = arith.select %true_ident, %ident, %ident : i1, i64
        %c4 = arith.constant 4 : i64
        %c60 = arith.constant 60 : i64
        %lo = arith.shrui %p, %c4 : i64
        %hi = arith.shli %p, %c60 : i64
        %rot = arith.ori %lo, %hi : i64
        scf.yield %rot : i64
      } else {
        %inf_hash = arith.constant 314159 : i64
        %neg_inf_hash = arith.constant -314159 : i64
        %is_neg = arith.cmpi ne, %sign_bit, %zero : i64
        %sel = arith.select %is_neg, %neg_inf_hash, %inf_hash : i1, i64
        scf.yield %sel : i64
      }
      scf.yield %special : i64
    } else {
      %is_zero_mag = arith.cmpi eq, %exp_bits, %zero : i64
      %mant_zero = arith.cmpi eq, %mant_bits, %zero : i64
      %is_zero = arith.andi %is_zero_mag, %mant_zero : i1
      %finite = scf.if %is_zero -> (i64) {
        scf.yield %zero : i64
      } else {
        %one = arith.constant 1 : i64
        %bias = arith.constant 1075 : i64
        %subnormal = arith.cmpi eq, %exp_bits, %zero : i64
        %norm_mant = arith.ori %mant_bits, %implicit_bit : i64
        %mant = arith.select %subnormal, %mant_bits, %norm_mant : i1, i64
        %norm_exp = arith.subi %exp_bits, %bias : i64
        %sub_exp = arith.subi %one, %bias : i64
        %exp = arith.select %subnormal, %sub_exp, %norm_exp : i1, i64
        // rot = exp mod 61 (Euclidean).
        %c61 = arith.constant 61 : i64
        %rem = arith.remsi %exp, %c61 : i64
        %rem_neg = arith.cmpi slt, %rem, %zero : i64
        %rem_adj = arith.addi %rem, %c61 : i64
        %rot = arith.select %rem_neg, %rem_adj, %rem : i1, i64
        %modulus = arith.constant 2305843009213693951 : i64
        %no_rot = arith.cmpi eq, %rot, %zero : i64
        %rotated = scf.if %no_rot -> (i64) {
          scf.yield %mant : i64
        } else {
          %up = arith.shli %mant, %rot : i64
          %up_masked = arith.andi %up, %modulus : i64
          %down_by = arith.subi %c61, %rot : i64
          %down = arith.shrui %mant, %down_by : i64
          %r = arith.ori %up_masked, %down : i64
          // One conditional reduction keeps the value inside [0, 2^61-1).
          %needs = arith.cmpi uge, %r, %modulus : i64
          %reduced = arith.subi %r, %modulus : i64
          %out = arith.select %needs, %reduced, %r : i1, i64
          scf.yield %out : i64
        }
        %negated = arith.subi %zero, %rotated : i64
        %is_neg = arith.cmpi ne, %sign_bit, %zero : i64
        %signed = arith.select %is_neg, %negated, %rotated : i1, i64
        scf.yield %signed : i64
      }
      scf.yield %finite : i64
    }
    %fixed = func.call @__ly_hash_fixup(%result) : (i64) -> i64
    func.return %fixed : i64
  }

  // float.__hash__ over the object: the bits are word 2 and the identity word
  // is the handle's own address.
  func.func @LyFloat_Hash(%header: memref<3xi64> {ly.ownership.object_header}) -> i64 attributes {ly.runtime.contract = "builtins.float", ly.runtime.method = "__hash__"} {
    %value_slot = arith.constant 2 : index
    %bits = memref.load %header[%value_slot] : memref<3xi64>
    %ptr_index = memref.extract_aligned_pointer_as_index %header : memref<3xi64> -> index
    %p = arith.index_cast %ptr_index : index to i64
    %c4 = arith.constant 4 : i64
    %c60 = arith.constant 60 : i64
    %lo = arith.shrui %p, %c4 : i64
    %hi = arith.shli %p, %c60 : i64
    %ident = arith.ori %lo, %hi : i64
    %hashed = func.call @__ly_float_hash_bits(%bits, %ident) : (i64, i64) -> i64
    func.return %hashed : i64
  }

  func.func @LyFloat_Abs(%header: memref<3xi64> {ly.ownership.object_header}) -> memref<3xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.float", ly.runtime.method = "__abs__", ly.runtime.result_contract = "builtins.float"} {
    %value_slot = arith.constant 2 : index
    %value_bits = memref.load %header[%value_slot] : memref<3xi64>
    %value = arith.bitcast %value_bits : i64 to f64
    %abs = math.absf %value : f64
    %h = func.call @LyFloat_FromF64(%abs) : (f64) -> memref<3xi64>
    func.return %h : memref<3xi64>
  }

  func.func @LyFloat_Neg(%header: memref<3xi64> {ly.ownership.object_header}) -> memref<3xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.float", ly.runtime.method = "__neg__", ly.runtime.result_contract = "builtins.float"} {
    %value_slot = arith.constant 2 : index
    %value_bits = memref.load %header[%value_slot] : memref<3xi64>
    %value = arith.bitcast %value_bits : i64 to f64
    %neg = arith.negf %value : f64
    %h = func.call @LyFloat_FromF64(%neg) : (f64) -> memref<3xi64>
    func.return %h : memref<3xi64>
  }

  func.func @LyFloat_Pos(%header: memref<3xi64> {ly.ownership.object_header}) -> memref<3xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.float", ly.runtime.method = "__pos__", ly.runtime.result_contract = "builtins.float"} {
    %value_slot = arith.constant 2 : index
    %value_bits = memref.load %header[%value_slot] : memref<3xi64>
    %value = arith.bitcast %value_bits : i64 to f64
    %h = func.call @LyFloat_FromF64(%value) : (f64) -> memref<3xi64>
    func.return %h : memref<3xi64>
  }

  memref.global "private" constant @__ly_float_text_ratio_nan : memref<35xi8> = dense<[99, 97, 110, 110, 111, 116, 32, 99, 111, 110, 118, 101, 114, 116, 32, 78, 97, 78, 32, 116, 111, 32, 105, 110, 116, 101, 103, 101, 114, 32, 114, 97, 116, 105, 111]>
  memref.global "private" constant @__ly_float_text_ratio_inf : memref<40xi8> = dense<[99, 97, 110, 110, 111, 116, 32, 99, 111, 110, 118, 101, 114, 116, 32, 73, 110, 102, 105, 110, 105, 116, 121, 32, 116, 111, 32, 105, 110, 116, 101, 103, 101, 114, 32, 114, 97, 116, 105, 111]>
  memref.global "private" constant @__ly_float_text_hex_digits : memref<16xi8> = dense<[48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 97, 98, 99, 100, 101, 102]>
  memref.global "private" constant @__ly_float_text_nan : memref<3xi8> = dense<[110, 97, 110]>
  memref.global "private" constant @__ly_float_text_inf : memref<3xi8> = dense<[105, 110, 102]>
  memref.global "private" constant @__ly_float_text_neg_inf : memref<4xi8> = dense<[45, 105, 110, 102]>
  memref.global "private" constant @__ly_float_text_zero : memref<8xi8> = dense<[48, 120, 48, 46, 48, 112, 43, 48]>
  memref.global "private" constant @__ly_float_text_neg_zero : memref<9xi8> = dense<[45, 48, 120, 48, 46, 48, 112, 43, 48]>
  memref.global "private" constant @__ly_float_text_infinity : memref<8xi8> = dense<[105, 110, 102, 105, 110, 105, 116, 121]>
  memref.global "private" constant @__ly_float_text_bad_hex : memref<41xi8> = dense<[105, 110, 118, 97, 108, 105, 100, 32, 104, 101, 120, 97, 100, 101, 99, 105, 109, 97, 108, 32, 102, 108, 111, 97, 116, 105, 110, 103, 45, 112, 111, 105, 110, 116, 32, 115, 116, 114, 105, 110, 103]>
  memref.global "private" constant @__ly_float_text_hex_too_large : memref<51xi8> = dense<[104, 101, 120, 97, 100, 101, 99, 105, 109, 97, 108, 32, 118, 97, 108, 117, 101, 32, 116, 111, 111, 32, 108, 97, 114, 103, 101, 32, 116, 111, 32, 114, 101, 112, 114, 101, 115, 101, 110, 116, 32, 97, 115, 32, 97, 32, 102, 108, 111, 97, 116]>

  // float.conjugate(): the float itself.
  func.func @LyFloat_Conjugate(%header: memref<3xi64> {ly.ownership.object_header}) -> memref<3xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.float", ly.runtime.method = "conjugate", ly.runtime.result_contract = "builtins.float"} {
    %h = func.call @LyFloat_Pos(%header) : (memref<3xi64>) -> memref<3xi64>
    func.return %h : memref<3xi64>
  }

  // float.is_integer(): finite and equal to its floor.
  func.func @LyFloat_IsInteger(%header: memref<3xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.float", ly.runtime.method = "is_integer"} {
    %value_slot = arith.constant 2 : index
    %value_bits = memref.load %header[%value_slot] : memref<3xi64>
    %value = arith.bitcast %value_bits : i64 to f64
    %inf = arith.constant 0x7FF0000000000000 : f64
    %magnitude = math.absf %value : f64
    %finite = arith.cmpf olt, %magnitude, %inf : f64
    %floor = math.floor %value : f64
    %whole = arith.cmpf oeq, %floor, %value : f64
    %result = arith.andi %finite, %whole : i1
    func.return %result : i1
  }

  // float.as_integer_ratio() (float_as_integer_ratio): the mantissa with its
  // trailing zero bits moved into the exponent, which then scales the
  // numerator or the denominator. Read from the bits, so no frexp loop.
  func.func @LyFloat_AsIntegerRatio(%header: memref<3xi64> {ly.ownership.object_header}) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.float", ly.runtime.method = "as_integer_ratio", ly.runtime.result_contract = "builtins.tuple"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %value_slot = arith.constant 2 : index
    %bits = memref.load %header[%value_slot] : memref<3xi64>
    %exp_shift = arith.constant 52 : i64
    %exp_mask = arith.constant 2047 : i64
    %mant_mask = arith.constant 4503599627370495 : i64
    %implicit_bit = arith.constant 4503599627370496 : i64
    %exp_shifted = arith.shrui %bits, %exp_shift : i64
    %exp_raw = arith.andi %exp_shifted, %exp_mask : i64
    %mant = arith.andi %bits, %mant_mask : i64
    %special = arith.cmpi eq, %exp_raw, %exp_mask : i64
    scf.if %special {
      %is_nan = arith.cmpi ne, %mant, %zero : i64
      scf.if %is_nan {
        %msg_static = memref.get_global @__ly_float_text_ratio_nan : memref<35xi8>
        %msg = memref.cast %msg_static : memref<35xi8> to memref<?xi8>
        %len = arith.constant 35 : i64
        %cls = arith.constant 53 : i64
        func.call @__ly_raise_static_message(%cls, %msg, %len) : (i64, memref<?xi8>, i64) -> ()
      } else {
        %msg_static = memref.get_global @__ly_float_text_ratio_inf : memref<40xi8>
        %msg = memref.cast %msg_static : memref<40xi8> to memref<?xi8>
        %len = arith.constant 40 : i64
        %cls = arith.constant 104 : i64
        func.call @__ly_raise_static_message(%cls, %msg, %len) : (i64, memref<?xi8>, i64) -> ()
      }
    }
    %subnormal = arith.cmpi eq, %exp_raw, %zero : i64
    %normal_mant = arith.ori %mant, %implicit_bit : i64
    %m0 = arith.select %subnormal, %mant, %normal_mant : i64
    %bias = arith.constant 1075 : i64
    %normal_e = arith.subi %exp_raw, %bias : i64
    %subnormal_e = arith.constant -1074 : i64
    %e0 = arith.select %subnormal, %subnormal_e, %normal_e : i64
    %is_zero = arith.cmpi eq, %m0, %zero : i64
    %tz_raw = math.cttz %m0 : i64
    %tz = arith.select %is_zero, %zero, %tz_raw : i64
    %m = arith.shrui %m0, %tz : i64
    %e_shifted = arith.addi %e0, %tz : i64
    %e = arith.select %is_zero, %zero, %e_shifted : i64
    %negative = arith.cmpi slt, %bits, %zero : i64
    %neg_m = arith.subi %zero, %m : i64
    %signed_m = arith.select %negative, %neg_m, %m : i64
    %up = arith.maxsi %e, %zero : i64
    %neg_e = arith.subi %zero, %e : i64
    %down = arith.maxsi %neg_e, %zero : i64
    %mantissa = func.call @LyLong_FromI64(%signed_m) : (i64) -> memref<2xi64>
    %up_count = func.call @LyLong_FromI64(%up) : (i64) -> memref<2xi64>
    %numerator = func.call @LyLong_LShift(%mantissa, %up_count) : (memref<2xi64>, memref<2xi64>) -> memref<2xi64>
    %unit = func.call @LyLong_FromI64(%one) : (i64) -> memref<2xi64>
    %down_count = func.call @LyLong_FromI64(%down) : (i64) -> memref<2xi64>
    %denominator = func.call @LyLong_LShift(%unit, %down_count) : (memref<2xi64>, memref<2xi64>) -> memref<2xi64>
    // ⛔ A native body is not given releases: the four operands of the
    // shifts are dropped here, or every value past the small ints leaks.
    func.call @LyLong_DecRef(%mantissa) : (memref<2xi64>) -> ()
    func.call @LyLong_DecRef(%up_count) : (memref<2xi64>) -> ()
    func.call @LyLong_DecRef(%unit) : (memref<2xi64>) -> ()
    func.call @LyLong_DecRef(%down_count) : (memref<2xi64>) -> ()
    %self = func.call @__ly_tuple_alloc(%two) : (i64) -> memref<5xi64>
    %items = func.call @__ly_tuple_items(%self) : (memref<5xi64>) -> memref<?xi64>
    func.call @__ly_tuple_store_long(%items, %c0, %numerator) : (memref<?xi64>, index, memref<2xi64>) -> ()
    func.call @__ly_tuple_store_long(%items, %c1, %denominator) : (memref<?xi64>, index, memref<2xi64>) -> ()
    func.return %self : memref<5xi64>
  }

  // float.hex() (float_hex): a leading 1 (0 for a subnormal), the 52
  // mantissa bits as 13 hex digits, and the binary exponent -- read from the
  // bits, which is what CPython's frexp/ldexp steps compute.
  func.func @LyFloat_Hex(%header: memref<3xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.float", ly.runtime.method = "hex", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %value_slot = arith.constant 2 : index
    %bits = memref.load %header[%value_slot] : memref<3xi64>
    %exp_shift = arith.constant 52 : i64
    %exp_mask = arith.constant 2047 : i64
    %mant_mask = arith.constant 4503599627370495 : i64
    %exp_shifted = arith.shrui %bits, %exp_shift : i64
    %exp_raw = arith.andi %exp_shifted, %exp_mask : i64
    %mant = arith.andi %bits, %mant_mask : i64
    %negative = arith.cmpi slt, %bits, %zero : i64
    %special = arith.cmpi eq, %exp_raw, %exp_mask : i64
    cf.cond_br %special, ^special, ^finite

  ^special:
    %is_nan = arith.cmpi ne, %mant, %zero : i64
    cf.cond_br %is_nan, ^nan, ^infinite

  ^infinite:
    cf.cond_br %negative, ^neg_inf, ^inf

  ^nan:
    %t_nan_static = memref.get_global @__ly_float_text_nan : memref<3xi8>
    %t_nan = memref.cast %t_nan_static : memref<3xi8> to memref<?xi8>
    %nan_c0 = arith.constant 0 : index
    %nan_len = arith.constant 3 : i64
    %nan_h, %nan_b = func.call @LyUnicode_FromBytes(%t_nan, %nan_c0, %nan_len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %nan_h, %nan_b : memref<2xi64>, memref<?xi8>

  ^inf:
    %t_inf_static = memref.get_global @__ly_float_text_inf : memref<3xi8>
    %t_inf = memref.cast %t_inf_static : memref<3xi8> to memref<?xi8>
    %inf_c0 = arith.constant 0 : index
    %inf_len = arith.constant 3 : i64
    %inf_h, %inf_b = func.call @LyUnicode_FromBytes(%t_inf, %inf_c0, %inf_len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %inf_h, %inf_b : memref<2xi64>, memref<?xi8>

  ^neg_inf:
    %t_neg_inf_static = memref.get_global @__ly_float_text_neg_inf : memref<4xi8>
    %t_neg_inf = memref.cast %t_neg_inf_static : memref<4xi8> to memref<?xi8>
    %neg_inf_c0 = arith.constant 0 : index
    %neg_inf_len = arith.constant 4 : i64
    %neg_inf_h, %neg_inf_b = func.call @LyUnicode_FromBytes(%t_neg_inf, %neg_inf_c0, %neg_inf_len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %neg_inf_h, %neg_inf_b : memref<2xi64>, memref<?xi8>

  ^finite:
    %subnormal = arith.cmpi eq, %exp_raw, %zero : i64
    %mant_zero = arith.cmpi eq, %mant, %zero : i64
    %is_zero = arith.andi %subnormal, %mant_zero : i1
    cf.cond_br %is_zero, ^zero_text, ^digits

  ^zero_text:
    cf.cond_br %negative, ^neg_zero, ^zero

  ^zero:
    %t_zero_static = memref.get_global @__ly_float_text_zero : memref<8xi8>
    %t_zero = memref.cast %t_zero_static : memref<8xi8> to memref<?xi8>
    %zero_c0 = arith.constant 0 : index
    %zero_len = arith.constant 8 : i64
    %zero_h, %zero_b = func.call @LyUnicode_FromBytes(%t_zero, %zero_c0, %zero_len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %zero_h, %zero_b : memref<2xi64>, memref<?xi8>

  ^neg_zero:
    %t_neg_zero_static = memref.get_global @__ly_float_text_neg_zero : memref<9xi8>
    %t_neg_zero = memref.cast %t_neg_zero_static : memref<9xi8> to memref<?xi8>
    %neg_zero_c0 = arith.constant 0 : index
    %neg_zero_len = arith.constant 9 : i64
    %neg_zero_h, %neg_zero_b = func.call @LyUnicode_FromBytes(%t_neg_zero, %neg_zero_c0, %neg_zero_len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %neg_zero_h, %neg_zero_b : memref<2xi64>, memref<?xi8>

  ^digits:
    %hex_static = memref.get_global @__ly_float_text_hex_digits : memref<16xi8>
    %hex = memref.cast %hex_static : memref<16xi8> to memref<?xi8>
    %buffer_size = arith.constant 32 : index
    %buffer = memref.alloc(%buffer_size) : memref<?xi8>
    %minus = arith.constant 45 : i8
    %ch_0 = arith.constant 48 : i8
    %ch_1 = arith.constant 49 : i8
    %ch_x = arith.constant 120 : i8
    %ch_dot = arith.constant 46 : i8
    %ch_p = arith.constant 112 : i8
    %ch_plus = arith.constant 43 : i8
    %start = arith.select %negative, %c1, %c0 : index
    scf.if %negative {
      memref.store %minus, %buffer[%c0] : memref<?xi8>
    }
    %p1 = arith.addi %start, %c1 : index
    memref.store %ch_0, %buffer[%start] : memref<?xi8>
    memref.store %ch_x, %buffer[%p1] : memref<?xi8>
    %p2 = arith.addi %p1, %c1 : index
    %lead = arith.select %subnormal, %ch_0, %ch_1 : i8
    memref.store %lead, %buffer[%p2] : memref<?xi8>
    %p3 = arith.addi %p2, %c1 : index
    memref.store %ch_dot, %buffer[%p3] : memref<?xi8>
    %p4 = arith.addi %p3, %c1 : index
    %c13 = arith.constant 13 : index
    %four = arith.constant 4 : i64
    %fifteen = arith.constant 15 : i64
    %top_shift = arith.constant 48 : i64
    scf.for %i = %c0 to %c13 step %c1 {
      %i64v = arith.index_cast %i : index to i64
      %drop = arith.muli %i64v, %four : i64
      %shift = arith.subi %top_shift, %drop : i64
      %nibble_word = arith.shrui %mant, %shift : i64
      %nibble = arith.andi %nibble_word, %fifteen : i64
      %nibble_index = arith.index_cast %nibble : i64 to index
      %ch = memref.load %hex[%nibble_index] : memref<?xi8>
      %slot = arith.addi %p4, %i : index
      memref.store %ch, %buffer[%slot] : memref<?xi8>
    }
    %p5 = arith.addi %p4, %c13 : index
    memref.store %ch_p, %buffer[%p5] : memref<?xi8>
    %p6 = arith.addi %p5, %c1 : index
    %bias = arith.constant 1023 : i64
    %normal_e = arith.subi %exp_raw, %bias : i64
    %subnormal_e = arith.constant -1022 : i64
    %e = arith.select %subnormal, %subnormal_e, %normal_e : i64
    %e_negative = arith.cmpi slt, %e, %zero : i64
    %e_sign = arith.select %e_negative, %minus, %ch_plus : i8
    memref.store %e_sign, %buffer[%p6] : memref<?xi8>
    %p7 = arith.addi %p6, %c1 : index
    %neg_e = arith.subi %zero, %e : i64
    %magnitude = arith.select %e_negative, %neg_e, %e : i64
    %c4 = arith.constant 4 : index
    %ten = arith.constant 10 : i64
    %thousand = arith.constant 1000 : i64
    %c3 = arith.constant 3 : index
    %false = arith.constant false
    %written:3 = scf.for %k = %c0 to %c4 step %c1 iter_args(%pos = %p7, %divisor = %thousand, %started = %false) -> (index, i64, i1) {
      %q = arith.divui %magnitude, %divisor : i64
      %digit = arith.remui %q, %ten : i64
      %nonzero = arith.cmpi ne, %digit, %zero : i64
      %is_last = arith.cmpi eq, %k, %c3 : index
      %some = arith.ori %started, %nonzero : i1
      %emit = arith.ori %some, %is_last : i1
      %next_pos = scf.if %emit -> (index) {
        %digit8 = arith.trunci %digit : i64 to i8
        %ch = arith.addi %digit8, %ch_0 : i8
        memref.store %ch, %buffer[%pos] : memref<?xi8>
        %bumped = arith.addi %pos, %c1 : index
        scf.yield %bumped : index
      } else {
        scf.yield %pos : index
      }
      %next_divisor = arith.divui %divisor, %ten : i64
      scf.yield %next_pos, %next_divisor, %emit : index, i64, i1
    }
    %length = arith.index_cast %written#0 : index to i64
    %h, %b = func.call @LyUnicode_FromBytes(%buffer, %c0, %length) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    memref.dealloc %buffer : memref<?xi8>
    func.return %h, %b : memref<2xi64>, memref<?xi8>
  }

  // The double nearest x * 2**e, for an x already rounded to fit: the bits
  // are assembled, so no ldexp is needed and nothing rounds twice.
  func.func private @__ly_float_exact_scaled(%x: i64, %e: i64) -> f64 {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %c52 = arith.constant 52 : i64
    %c53 = arith.constant 53 : i64
    %c64 = arith.constant 64 : i64
    %bias = arith.constant 1023 : i64
    %min_exp = arith.constant -1022 : i64
    %subnormal_shift_base = arith.constant 1074 : i64
    %mant_mask = arith.constant 4503599627370495 : i64
    %is_zero = arith.cmpi eq, %x, %zero : i64
    %lz = math.ctlz %x : i64
    %bitlen = arith.subi %c64, %lz : i64
    %top = arith.addi %e, %bitlen : i64
    %top_exp = arith.subi %top, %one : i64
    %normal = arith.cmpi sge, %top_exp, %min_exp : i64
    // Normal: the 53 significant bits, the leading one dropped.
    %excess = arith.subi %bitlen, %c53 : i64
    %too_long = arith.cmpi sgt, %excess, %zero : i64
    %neg_excess = arith.subi %zero, %excess : i64
    %right = arith.shrui %x, %excess : i64
    %left = arith.shli %x, %neg_excess : i64
    %mant53 = arith.select %too_long, %right, %left : i64
    %fraction = arith.andi %mant53, %mant_mask : i64
    %biased = arith.addi %top_exp, %bias : i64
    %exp_field = arith.shli %biased, %c52 : i64
    %normal_bits = arith.ori %exp_field, %fraction : i64
    // Subnormal: x * 2**(e + 1074) is the field itself.
    %sub_shift = arith.addi %e, %subnormal_shift_base : i64
    %sub_right_count = arith.subi %zero, %sub_shift : i64
    %sub_left = arith.cmpi sge, %sub_shift, %zero : i64
    %sub_l = arith.shli %x, %sub_shift : i64
    %sub_r = arith.shrui %x, %sub_right_count : i64
    %subnormal_bits = arith.select %sub_left, %sub_l, %sub_r : i64
    %chosen = arith.select %normal, %normal_bits, %subnormal_bits : i64
    %bits = arith.select %is_zero, %zero, %chosen : i64
    %result = arith.bitcast %bits : i64 to f64
    func.return %result : f64
  }

  // CPython's float_fromhex rounding over `total` hex digit values (most
  // significant first) scaled by 2**exp: round-half-even at the double's
  // last bit, subnormals included. The i1 is the overflow CPython reports.
  func.func private @__ly_float_round_hex_digits(%digits: memref<?xi8>, %total: i64, %exp: i64) -> (f64, i1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %four = arith.constant 4 : i64
    %eight = arith.constant 8 : i64
    %sixteen = arith.constant 16 : i64
    %c53 = arith.constant 53 : i64
    %c64 = arith.constant 64 : i64
    %min_exp = arith.constant -1021 : i64
    %max_exp = arith.constant 1024 : i64
    %underflow_exp = arith.constant -1074 : i64
    %c0_i8 = arith.constant 0 : i8
    %false = arith.constant false
    %true = arith.constant true
    %zero_f = arith.constant 0.0 : f64
    %total_index = arith.index_cast %total : i64 to index
    // Leading zeros carry nothing.
    %lead = scf.for %i = %c0 to %total_index step %c1 iter_args(%found = %total) -> (i64) {
      %unset = arith.cmpi eq, %found, %total : i64
      %d8 = memref.load %digits[%i] : memref<?xi8>
      %nonzero = arith.cmpi ne, %d8, %c0_i8 : i8
      %take = arith.andi %unset, %nonzero : i1
      %i64v = arith.index_cast %i : index to i64
      %next = arith.select %take, %i64v, %found : i64
      scf.yield %next : i64
    }
    %nd = arith.subi %total, %lead : i64
    %no_digits = arith.cmpi eq, %nd, %zero : i64
    cf.cond_br %no_digits, ^zero_result, ^scale

  ^zero_result:
    func.return %zero_f, %false : f64, i1

  ^scale:
    // HEX_DIGIT(j), j counted from the least significant digit, is
    // digits[total - 1 - j].
    %last = arith.subi %total, %one : i64
    %lead_index = arith.index_cast %lead : i64 to index
    %top_digit8 = memref.load %digits[%lead_index] : memref<?xi8>
    %top_digit = arith.extui %top_digit8 : i8 to i64
    %top_lz = math.ctlz %top_digit : i64
    %top_bits = arith.subi %c64, %top_lz : i64
    %nd_less = arith.subi %nd, %one : i64
    %whole_digits_bits = arith.muli %nd_less, %four : i64
    %top_base = arith.addi %exp, %whole_digits_bits : i64
    %top_exp = arith.addi %top_base, %top_bits : i64
    %underflows = arith.cmpi slt, %top_exp, %underflow_exp : i64
    cf.cond_br %underflows, ^zero_result, ^check_overflow

  ^check_overflow:
    %overflows = arith.cmpi sgt, %top_exp, %max_exp : i64
    cf.cond_br %overflows, ^overflow, ^round

  ^overflow:
    func.return %zero_f, %true : f64, i1

  ^round:
    %floor_exp = arith.maxsi %top_exp, %min_exp : i64
    %lsb = arith.subi %floor_exp, %c53 : i64
    %exact = arith.cmpi sge, %exp, %lsb : i64
    cf.cond_br %exact, ^exact_digits, ^rounded_digits

  ^exact_digits:
    %nd_index = arith.index_cast %nd : i64 to index
    %all = scf.for %k = %c0 to %nd_index step %c1 iter_args(%x = %zero) -> (i64) {
      %k64 = arith.index_cast %k : index to i64
      %at64 = arith.addi %lead, %k64 : i64
      %at = arith.index_cast %at64 : i64 to index
      %d8 = memref.load %digits[%at] : memref<?xi8>
      %d = arith.extui %d8 : i8 to i64
      %shifted = arith.muli %x, %sixteen : i64
      %next = arith.addi %shifted, %d : i64
      scf.yield %next : i64
    }
    %exact_value = func.call @__ly_float_exact_scaled(%all, %exp) : (i64, i64) -> f64
    func.return %exact_value, %false : f64, i1

  ^rounded_digits:
    %gap = arith.subi %lsb, %exp : i64
    %gap_less = arith.subi %gap, %one : i64
    %half_shift = arith.remsi %gap_less, %four : i64
    %half_eps = arith.shli %one, %half_shift : i64
    %key = arith.divsi %gap_less, %four : i64
    // Digits above the key digit, most significant first: positions
    // lead .. total - 2 - key.
    %key_pos = arith.subi %last, %key : i64
    %lead_index_r = arith.index_cast %lead : i64 to index
    %key_index = arith.index_cast %key_pos : i64 to index
    %above = scf.for %at = %lead_index_r to %key_index step %c1 iter_args(%x = %zero) -> (i64) {
      %d8 = memref.load %digits[%at] : memref<?xi8>
      %d = arith.extui %d8 : i8 to i64
      %shifted = arith.muli %x, %sixteen : i64
      %next = arith.addi %shifted, %d : i64
      scf.yield %next : i64
    }
    %key8 = memref.load %digits[%key_index] : memref<?xi8>
    %key_digit = arith.extui %key8 : i8 to i64
    %two_half = arith.muli %half_eps, %two : i64
    %keep_mask = arith.subi %sixteen, %two_half : i64
    %kept = arith.andi %key_digit, %keep_mask : i64
    %above16 = arith.muli %above, %sixteen : i64
    %x0 = arith.addi %above16, %kept : i64
    %half_bit = arith.andi %key_digit, %half_eps : i64
    %at_half = arith.cmpi ne, %half_bit, %zero : i64
    // Round up past the half, or at it when the kept part is odd.
    %three_half = arith.muli %half_eps, %three : i64
    %below_mask = arith.subi %three_half, %one : i64
    %below_bits = arith.andi %key_digit, %below_mask : i64
    %below_or_odd = arith.cmpi ne, %below_bits, %zero : i64
    %half_is_eight = arith.cmpi eq, %half_eps, %eight : i64
    %key_up = arith.addi %key, %one : i64
    %has_next = arith.cmpi slt, %key_up, %nd : i64
    %eight_case = arith.andi %half_is_eight, %has_next : i1
    %next_odd = scf.if %eight_case -> (i1) {
      %prev_pos = arith.subi %key_index, %c1 : index
      %p8 = memref.load %digits[%prev_pos] : memref<?xi8>
      %p = arith.extui %p8 : i8 to i64
      %p_low = arith.andi %p, %one : i64
      %odd = arith.cmpi ne, %p_low, %zero : i64
      scf.yield %odd : i1
    } else {
      scf.yield %false : i1
    }
    %decided = arith.ori %below_or_odd, %next_odd : i1
    %after_key = arith.addi %key_index, %c1 : index
    %sticky = scf.for %at = %after_key to %total_index step %c1 iter_args(%any = %false) -> (i1) {
      %d8 = memref.load %digits[%at] : memref<?xi8>
      %nonzero = arith.cmpi ne, %d8, %c0_i8 : i8
      %next = arith.ori %any, %nonzero : i1
      scf.yield %next : i1
    }
    %up_reason = arith.ori %decided, %sticky : i1
    %round_up = arith.andi %at_half, %up_reason : i1
    %bumped = arith.addi %x0, %two_half : i64
    %x = arith.select %round_up, %bumped, %x0 : i64
    %at_max = arith.cmpi eq, %top_exp, %max_exp : i64
    %limit = arith.shli %two_half, %c53 : i64
    %reached = arith.cmpi eq, %x, %limit : i64
    %rounded_over = arith.andi %at_max, %reached : i1
    %round_over_up = arith.andi %rounded_over, %round_up : i1
    %key_bits = arith.muli %key, %four : i64
    %scale = arith.addi %exp, %key_bits : i64
    %rounded_value = func.call @__ly_float_exact_scaled(%x, %scale) : (i64, i64) -> f64
    func.return %rounded_value, %round_over_up : f64, i1
  }

  // Whether code points [start, start + len) of a str spell `word`, ASCII
  // case ignored.
  func.func private @__ly_float_span_is(%bytes: memref<?xi8>, %width: i64, %start: index, %len: index, %word: memref<?xi8>) -> i1 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %word_len = memref.dim %word, %c0 : memref<?xi8>
    %same_len = arith.cmpi eq, %len, %word_len : index
    %result = scf.if %same_len -> (i1) {
      %all = scf.for %i = %c0 to %len step %c1 iter_args(%acc = %true) -> (i1) {
        %at = arith.addi %start, %i : index
        %cp = func.call @__ly_unicode_get(%bytes, %width, %at) : (memref<?xi8>, i64, index) -> i64
        %upper_a = arith.constant 65 : i64
        %upper_z = arith.constant 90 : i64
        %ge = arith.cmpi sge, %cp, %upper_a : i64
        %le = arith.cmpi sle, %cp, %upper_z : i64
        %is_upper = arith.andi %ge, %le : i1
        %case_bit = arith.constant 32 : i64
        %lowered = arith.ori %cp, %case_bit : i64
        %folded = arith.select %is_upper, %lowered, %cp : i64
        %w8 = memref.load %word[%i] : memref<?xi8>
        %w = arith.extui %w8 : i8 to i64
        %eq = arith.cmpi eq, %folded, %w : i64
        %next = arith.andi %acc, %eq : i1
        scf.yield %next : i1
      }
      scf.yield %all : i1
    } else {
      %no = arith.constant false
      scf.yield %no : i1
    }
    func.return %result : i1
  }

  // float.fromhex (float_fromhex): an optional "0x", hex digits with at most
  // one point, an optional binary exponent; the digits are rounded by
  // __ly_float_round_hex_digits.
  //
  // ⛔ Not strtod over "0x" + the text: wasi-libc's answered
  // "0x1.fffffffffffff7p1023" with an overflow where CPython and the host
  // libc round down to the largest double -- a target-dependent answer.
  func.func @LyFloat_FromHex(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> memref<3xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.float", ly.runtime.initializer = "fromhex"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %neg_one = arith.constant -1 : i64
    %true = arith.constant true
    %false = arith.constant false
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %count = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %count_index = arith.index_cast %count : i64 to index
    %cp_space = arith.constant 32 : i64
    %cp_tab = arith.constant 9 : i64
    %cp_cr = arith.constant 13 : i64
    %first = scf.for %i = %c0 to %count_index step %c1 iter_args(%found = %neg_one) -> (i64) {
      %cp = func.call @__ly_unicode_get(%bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
      %is_sp = arith.cmpi eq, %cp, %cp_space : i64
      %ge_tab = arith.cmpi sge, %cp, %cp_tab : i64
      %le_cr = arith.cmpi sle, %cp, %cp_cr : i64
      %is_ctl = arith.andi %ge_tab, %le_cr : i1
      %is_ws = arith.ori %is_sp, %is_ctl : i1
      %unset = arith.cmpi eq, %found, %neg_one : i64
      %i64v = arith.index_cast %i : index to i64
      %candidate = arith.select %is_ws, %found, %i64v : i1, i64
      %next = arith.select %unset, %candidate, %found : i1, i64
      scf.yield %next : i64
    }
    %last = scf.for %i = %c0 to %count_index step %c1 iter_args(%found = %neg_one) -> (i64) {
      %step = arith.addi %i, %c1 : index
      %rev = arith.subi %count_index, %step : index
      %cp = func.call @__ly_unicode_get(%bytes, %width, %rev) : (memref<?xi8>, i64, index) -> i64
      %is_sp = arith.cmpi eq, %cp, %cp_space : i64
      %ge_tab = arith.cmpi sge, %cp, %cp_tab : i64
      %le_cr = arith.cmpi sle, %cp, %cp_cr : i64
      %is_ctl = arith.andi %ge_tab, %le_cr : i1
      %is_ws = arith.ori %is_sp, %is_ctl : i1
      %unset = arith.cmpi eq, %found, %neg_one : i64
      %rev64 = arith.index_cast %rev : index to i64
      %candidate = arith.select %is_ws, %found, %rev64 : i1, i64
      %next = arith.select %unset, %candidate, %found : i1, i64
      scf.yield %next : i64
    }
    %empty = arith.cmpi eq, %first, %neg_one : i64
    cf.cond_br %empty, ^invalid, ^sign

  ^sign:
    %first_index = arith.index_cast %first : i64 to index
    %end64 = arith.addi %last, %one : i64
    %end = arith.index_cast %end64 : i64 to index
    %lead = func.call @__ly_unicode_get(%bytes, %width, %first_index) : (memref<?xi8>, i64, index) -> i64
    %cp_minus = arith.constant 45 : i64
    %cp_plus = arith.constant 43 : i64
    %is_minus = arith.cmpi eq, %lead, %cp_minus : i64
    %is_plus = arith.cmpi eq, %lead, %cp_plus : i64
    %has_sign = arith.ori %is_minus, %is_plus : i1
    %after_sign = arith.addi %first_index, %c1 : index
    %body = arith.select %has_sign, %after_sign, %first_index : index
    %body_len = arith.subi %end, %body : index
    %w_inf_static = memref.get_global @__ly_float_text_inf : memref<3xi8>
    %w_inf = memref.cast %w_inf_static : memref<3xi8> to memref<?xi8>
    %w_infinity_static = memref.get_global @__ly_float_text_infinity : memref<8xi8>
    %w_infinity = memref.cast %w_infinity_static : memref<8xi8> to memref<?xi8>
    %w_nan_static = memref.get_global @__ly_float_text_nan : memref<3xi8>
    %w_nan = memref.cast %w_nan_static : memref<3xi8> to memref<?xi8>
    %is_inf = func.call @__ly_float_span_is(%bytes, %width, %body, %body_len, %w_inf) : (memref<?xi8>, i64, index, index, memref<?xi8>) -> i1
    %is_infinity = func.call @__ly_float_span_is(%bytes, %width, %body, %body_len, %w_infinity) : (memref<?xi8>, i64, index, index, memref<?xi8>) -> i1
    %is_nan = func.call @__ly_float_span_is(%bytes, %width, %body, %body_len, %w_nan) : (memref<?xi8>, i64, index, index, memref<?xi8>) -> i1
    %infinite = arith.ori %is_inf, %is_infinity : i1
    cf.cond_br %infinite, ^infinite, ^not_infinite

  ^infinite:
    %pos_inf = arith.constant 0x7FF0000000000000 : f64
    %neg_inf = arith.constant 0xFFF0000000000000 : f64
    %inf_value = arith.select %is_minus, %neg_inf, %pos_inf : f64
    %inf_h = func.call @LyFloat_FromF64(%inf_value) : (f64) -> memref<3xi64>
    func.return %inf_h : memref<3xi64>

  ^not_infinite:
    cf.cond_br %is_nan, ^nan, ^hex

  ^nan:
    %nan_value = arith.constant 0x7FF8000000000000 : f64
    %nan_h = func.call @LyFloat_FromF64(%nan_value) : (f64) -> memref<3xi64>
    func.return %nan_h : memref<3xi64>

  ^hex:
    // An optional "0x" / "0X".
    %cp_zero = arith.constant 48 : i64
    %cp_x = arith.constant 120 : i64
    %case_bit = arith.constant 32 : i64
    %room_for_prefix = arith.cmpi sge, %body_len, %c2 : index
    %prefixed = scf.if %room_for_prefix -> (i1) {
      %a = func.call @__ly_unicode_get(%bytes, %width, %body) : (memref<?xi8>, i64, index) -> i64
      %second = arith.addi %body, %c1 : index
      %b = func.call @__ly_unicode_get(%bytes, %width, %second) : (memref<?xi8>, i64, index) -> i64
      %b_lower = arith.ori %b, %case_bit : i64
      %a_ok = arith.cmpi eq, %a, %cp_zero : i64
      %b_ok = arith.cmpi eq, %b_lower, %cp_x : i64
      %both = arith.andi %a_ok, %b_ok : i1
      scf.yield %both : i1
    } else {
      scf.yield %false : i1
    }
    %after_prefix = arith.addi %body, %c2 : index
    %digits_start = arith.select %prefixed, %after_prefix, %body : index
    %rest = arith.subi %end, %digits_start : index
    %digits_size = arith.addi %rest, %c1 : index
    %digits = memref.alloc(%digits_size) : memref<?xi8>
    // States: 0 integer digits, 1 fraction digits, 2 just after 'p',
    // 3 after the exponent's sign, 4 exponent digits. The exponent saturates
    // far past any double, as CPython's strtol does past a long.
    %s0 = arith.constant 0 : i64
    %s1 = arith.constant 1 : i64
    %s2 = arith.constant 2 : i64
    %s3 = arith.constant 3 : i64
    %s4 = arith.constant 4 : i64
    %ten = arith.constant 10 : i64
    %exp_cap = arith.constant 1099511627776 : i64
    %scan:6 = scf.for %i = %digits_start to %end step %c1 iter_args(%state = %s0, %ok = %true, %total = %zero, %fdigits = %zero, %exp_abs = %zero, %exp_neg = %false) -> (i64, i1, i64, i64, i64, i1) {
      %cp = func.call @__ly_unicode_get(%bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
      %lower = arith.ori %cp, %case_bit : i64
      %ge_0 = arith.cmpi sge, %cp, %cp_zero : i64
      %cp_nine = arith.constant 57 : i64
      %le_9 = arith.cmpi sle, %cp, %cp_nine : i64
      %is_dec = arith.andi %ge_0, %le_9 : i1
      %cp_a = arith.constant 97 : i64
      %cp_f = arith.constant 102 : i64
      %ge_a = arith.cmpi sge, %lower, %cp_a : i64
      %le_f = arith.cmpi sle, %lower, %cp_f : i64
      %is_letter = arith.andi %ge_a, %le_f : i1
      %is_hex = arith.ori %is_dec, %is_letter : i1
      %cp_dot = arith.constant 46 : i64
      %cp_p = arith.constant 112 : i64
      %is_dot = arith.cmpi eq, %cp, %cp_dot : i64
      %is_p = arith.cmpi eq, %lower, %cp_p : i64
      %is_sign_ch_m = arith.cmpi eq, %cp, %cp_minus : i64
      %is_sign_ch_p = arith.cmpi eq, %cp, %cp_plus : i64
      %is_sign_ch = arith.ori %is_sign_ch_m, %is_sign_ch_p : i1
      %in0 = arith.cmpi eq, %state, %s0 : i64
      %in1 = arith.cmpi eq, %state, %s1 : i64
      %in2 = arith.cmpi eq, %state, %s2 : i64
      %in3 = arith.cmpi eq, %state, %s3 : i64
      %in4 = arith.cmpi eq, %state, %s4 : i64
      %in_mantissa = arith.ori %in0, %in1 : i1
      %in_exp_head = arith.ori %in2, %in3 : i1
      %in_exp = arith.ori %in_exp_head, %in4 : i1
      %mant_digit = arith.andi %in_mantissa, %is_hex : i1
      %point = arith.andi %in0, %is_dot : i1
      %exp_mark = arith.andi %in_mantissa, %is_p : i1
      %exp_sign = arith.andi %in2, %is_sign_ch : i1
      %exp_digit = arith.andi %in_exp, %is_dec : i1
      %t1 = arith.ori %mant_digit, %point : i1
      %t2 = arith.ori %t1, %exp_mark : i1
      %t3 = arith.ori %t2, %exp_sign : i1
      %accepted = arith.ori %t3, %exp_digit : i1
      %next_ok = arith.andi %ok, %accepted : i1
      %st_point = arith.select %point, %s1, %state : i64
      %st_mark = arith.select %exp_mark, %s2, %st_point : i64
      %st_sign = arith.select %exp_sign, %s3, %st_mark : i64
      %next_state = arith.select %exp_digit, %s4, %st_sign : i64
      scf.if %mant_digit {
        %letter_value = arith.constant 87 : i64
        %dec_value = arith.subi %cp, %cp_zero : i64
        %hex_value = arith.subi %lower, %letter_value : i64
        %value = arith.select %is_dec, %dec_value, %hex_value : i64
        %value8 = arith.trunci %value : i64 to i8
        %slot = arith.index_cast %total : i64 to index
        memref.store %value8, %digits[%slot] : memref<?xi8>
      }
      %bump = arith.select %mant_digit, %one, %zero : i64
      %next_total = arith.addi %total, %bump : i64
      %fraction_digit = arith.andi %mant_digit, %in1 : i1
      %fbump = arith.select %fraction_digit, %one, %zero : i64
      %next_fdigits = arith.addi %fdigits, %fbump : i64
      %dec = arith.subi %cp, %cp_zero : i64
      %scaled = arith.muli %exp_abs, %ten : i64
      %grown = arith.addi %scaled, %dec : i64
      %capped = arith.minsi %grown, %exp_cap : i64
      %next_exp_abs = arith.select %exp_digit, %capped, %exp_abs : i64
      %negates = arith.andi %exp_sign, %is_sign_ch_m : i1
      %next_exp_neg = arith.ori %exp_neg, %negates : i1
      scf.yield %next_state, %next_ok, %next_total, %next_fdigits, %next_exp_abs, %next_exp_neg : i64, i1, i64, i64, i64, i1
    }
    %has_digits = arith.cmpi sgt, %scan#2, %zero : i64
    %ends_head = arith.cmpi eq, %scan#0, %s2 : i64
    %ends_sign = arith.cmpi eq, %scan#0, %s3 : i64
    %ends_open = arith.ori %ends_head, %ends_sign : i1
    %ends_closed = arith.xori %ends_open, %true : i1
    %grammar_ok = arith.andi %scan#1, %has_digits : i1
    %valid = arith.andi %grammar_ok, %ends_closed : i1
    cf.cond_br %valid, ^parse, ^invalid_free

  ^invalid_free:
    memref.dealloc %digits : memref<?xi8>
    cf.br ^invalid

  ^parse:
    %four = arith.constant 4 : i64
    %neg_exp_abs = arith.subi %zero, %scan#4 : i64
    %parsed_exp = arith.select %scan#5, %neg_exp_abs, %scan#4 : i64
    %fraction_bits = arith.muli %scan#3, %four : i64
    %exp = arith.subi %parsed_exp, %fraction_bits : i64
    %magnitude, %overflow = func.call @__ly_float_round_hex_digits(%digits, %scan#2, %exp) : (memref<?xi8>, i64, i64) -> (f64, i1)
    memref.dealloc %digits : memref<?xi8>
    scf.if %overflow {
      %msg_static = memref.get_global @__ly_float_text_hex_too_large : memref<51xi8>
      %msg = memref.cast %msg_static : memref<51xi8> to memref<?xi8>
      %len = arith.constant 51 : i64
      %cls = arith.constant 104 : i64
      func.call @__ly_raise_static_message(%cls, %msg, %len) : (i64, memref<?xi8>, i64) -> ()
    }
    %negated = arith.negf %magnitude : f64
    %value = arith.select %is_minus, %negated, %magnitude : f64
    %h = func.call @LyFloat_FromF64(%value) : (f64) -> memref<3xi64>
    func.return %h : memref<3xi64>

  ^invalid:
    %msg_static = memref.get_global @__ly_float_text_bad_hex : memref<41xi8>
    %msg = memref.cast %msg_static : memref<41xi8> to memref<?xi8>
    %len = arith.constant 41 : i64
    %cls = arith.constant 53 : i64
    func.call @__ly_raise_static_message(%cls, %msg, %len) : (i64, memref<?xi8>, i64) -> ()
    %zero_f = arith.constant 0.0 : f64
    %unreached = func.call @LyFloat_FromF64(%zero_f) : (f64) -> memref<3xi64>
    func.return %unreached : memref<3xi64>
  }
}
