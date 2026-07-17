// Contract manifest AND runtime implementation for the statically supported
// `unicodedata` surface (CPython's Modules/unicodedata.c counterpart), plus
// the UCD-driven str case-mapping and classification runtime methods that
// Wave 1 exposes on py.class @str. The two-stage tables and their accessors
// live in the generated _ucd.mlir (tools/gen_unicode_tables.py).
//
// Deviations from CPython's unicodedata (per project convention, recorded
// here at the top of the module):
//   * only category / numeric / decimal / digit are exported; name, lookup,
//     normalize, bidirectional, combining, mirrored, east_asian_width and
//     the ucd_3_2_0 snapshot are not implemented yet;
//   * the optional `default` parameter of numeric/decimal/digit is not
//     supported: a character without the property always raises ValueError;
//   * numeric values come from UnicodeData.txt field 8 only -- CJK numerals
//     sourced from the Unihan database (e.g. U+4E94) have no numeric value.
module attributes {
  ly.typing.module = "unicodedata",
  ly.typing.callable_exports = [
    "unicodedata.category",
    "unicodedata.numeric",
    "unicodedata.decimal",
    "unicodedata.digit"
  ],
  ly.typing.function_names = [
    "unicodedata.category",
    "unicodedata.numeric",
    "unicodedata.decimal",
    "unicodedata.digit"
  ],
  ly.typing.function_contracts = [
    !py.callable<[!py.contract<"builtins.str">], arg_names = ["chr"], arg_defaults = [false], returns = [!py.contract<"builtins.str">]>,
    !py.callable<[!py.contract<"builtins.str">], arg_names = ["chr"], arg_defaults = [false], returns = [!py.contract<"builtins.float">]>,
    !py.callable<[!py.contract<"builtins.str">], arg_names = ["chr"], arg_defaults = [false], returns = [!py.contract<"builtins.int">]>,
    !py.callable<[!py.contract<"builtins.str">], arg_names = ["chr"], arg_defaults = [false], returns = [!py.contract<"builtins.int">]>
  ]
} {
  // ---- declarations resolved by the runtime import merge ----
  func.func private @__ly_unicode_alloc(%count: i64, %width: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0]}
  func.func private @__ly_unicode_width(%header: memref<2xi64>) -> i64
  func.func private @__ly_unicode_count(%header: memref<2xi64>, %bytes: memref<?xi8>) -> i64
  func.func private @__ly_unicode_get(%bytes: memref<?xi8>, %width: i64, %i: index) -> i64
  func.func private @__ly_unicode_put(%bytes: memref<?xi8>, %width: i64, %i: index, %cp: i64)
  func.func private @__ly_unicode_width_for(%cp: i64) -> i64
  func.func private @__ly_ucd_ctype(%cp: i64) -> (i64, i64, i64, i64, i64, i64)
  func.func private @__ly_ucd_info(%cp: i64) -> (i64, i64)
  func.func private @__ly_ucd_ext_cp(%packed: i64, %j: i64) -> i64
  func.func private @__ly_ucd_numeric_value(%idx: i64) -> f64
  func.func private @__ly_ucd_category_char(%cat: i64, %j: i64) -> i64
  func.func private @LyUnicode_FromBytes(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0]}
  func.func private @LyLong_FromI64(%value: i64 {ly.runtime.default_i64 = 0 : i64}) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 1 : i64, ly.runtime.contract = "builtins.int", ly.runtime.initializer = "__new__"}
  func.func private @LyFloat_FromF64(%value: f64 {ly.runtime.default_f64 = 0.0 : f64}) -> (memref<2xi64>, memref<1xf64>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 2 : i64, ly.runtime.contract = "builtins.float", ly.runtime.initializer = "__new__"}
  func.func private @LyBaseException_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 5 : i64, ly.runtime.contract = "builtins.BaseException", ly.runtime.initializer = "__new__"}
  func.func private @LyBaseException_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.BaseException", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"}
  func.func private @LyEH_ThrowException(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "raise"}

  // Flag bits, kept in sync with tools/gen_unicode_tables.py:
  // 1 ALPHA, 2 LOWER, 4 UPPER, 8 TITLE, 16 SPACE, 32 CASED,
  // 64 CASE_IGNORABLE, 128 EXT_UPPER, 256 EXT_LOWER, 512 EXT_FOLD.

  // "argument must be a unicode character"
  memref.global "private" constant @__ly_ucd_msg_not_single : memref<36xi8> = dense<[97, 114, 103, 117, 109, 101, 110, 116, 32, 109, 117, 115, 116, 32, 98, 101, 32, 97, 32, 117, 110, 105, 99, 111, 100, 101, 32, 99, 104, 97, 114, 97, 99, 116, 101, 114]>
  // "not a numeric character"
  memref.global "private" constant @__ly_ucd_msg_not_numeric : memref<23xi8> = dense<[110, 111, 116, 32, 97, 32, 110, 117, 109, 101, 114, 105, 99, 32, 99, 104, 97, 114, 97, 99, 116, 101, 114]>
  // "not a decimal"
  memref.global "private" constant @__ly_ucd_msg_not_decimal : memref<13xi8> = dense<[110, 111, 116, 32, 97, 32, 100, 101, 99, 105, 109, 97, 108]>
  // "not a digit"
  memref.global "private" constant @__ly_ucd_msg_not_digit : memref<11xi8> = dense<[110, 111, 116, 32, 97, 32, 100, 105, 103, 105, 116]>

  func.func private @__ly_ucd_raise(%class_id: i64, %message: memref<?xi8>, %length: i64) {
    %start = arith.constant 0 : index
    %exception:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    %message_header, %message_bytes = func.call @LyUnicode_FromBytes(%message, %start, %length) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %initialized:3 = func.call @LyBaseException_Init(%exception#0, %exception#1, %exception#2, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.call @LyEH_ThrowException(%initialized#0, %initialized#1, %initialized#2) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  // The single code point of a one-character str; TypeError otherwise
  // (CPython's unicodedata argument contract).
  func.func private @__ly_ucd_single_cp(%header: memref<2xi64>, %bytes: memref<?xi8>) -> i64 {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1 : i64
    %count = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %single = arith.cmpi eq, %count, %one : i64
    %true_bit = arith.constant true
    %bad = arith.xori %single, %true_bit : i1
    scf.if %bad {
      %class_id = arith.constant 52 : i64
      %length = arith.constant 36 : i64
      %static = memref.get_global @__ly_ucd_msg_not_single : memref<36xi8>
      %message = memref.cast %static : memref<36xi8> to memref<?xi8>
      func.call @__ly_ucd_raise(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    }
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %cp = func.call @__ly_unicode_get(%bytes, %width, %c0) : (memref<?xi8>, i64, index) -> i64
    func.return %cp : i64
  }

  // ---- public unicodedata functions ----

  func.func @LyUnicodeData_Category(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "unicodedata.category", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.str", ly.runtime.primitive = "ucd_category", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %cp = func.call @__ly_ucd_single_cp(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %cat, %numeric = func.call @__ly_ucd_info(%cp) : (i64) -> (i64, i64)
    %out_header, %out_bytes = func.call @__ly_unicode_alloc(%two, %one) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)
    %ch0 = func.call @__ly_ucd_category_char(%cat, %zero) : (i64, i64) -> i64
    %ch1 = func.call @__ly_ucd_category_char(%cat, %one) : (i64, i64) -> i64
    func.call @__ly_unicode_put(%out_bytes, %one, %c0, %ch0) : (memref<?xi8>, i64, index, i64) -> ()
    func.call @__ly_unicode_put(%out_bytes, %one, %c1, %ch1) : (memref<?xi8>, i64, index, i64) -> ()
    func.return %out_header, %out_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicodeData_Numeric(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<1xf64>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "unicodedata.numeric", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.str", ly.runtime.primitive = "ucd_numeric", ly.runtime.result_contract = "builtins.float"} {
    %zero = arith.constant 0 : i64
    %minus_one = arith.constant -1 : i64
    %cp = func.call @__ly_ucd_single_cp(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %cat, %numeric = func.call @__ly_ucd_info(%cp) : (i64) -> (i64, i64)
    %missing = arith.cmpi eq, %numeric, %minus_one : i64
    scf.if %missing {
      %class_id = arith.constant 53 : i64
      %length = arith.constant 23 : i64
      %static = memref.get_global @__ly_ucd_msg_not_numeric : memref<23xi8>
      %message = memref.cast %static : memref<23xi8> to memref<?xi8>
      func.call @__ly_ucd_raise(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    }
    // The raise above unwinds; the select keeps the fallthrough load in
    // bounds for the verifier.
    %safe_index = arith.select %missing, %zero, %numeric : i64
    %value = func.call @__ly_ucd_numeric_value(%safe_index) : (i64) -> f64
    %out_header, %out_payload = func.call @LyFloat_FromF64(%value) : (f64) -> (memref<2xi64>, memref<1xf64>)
    func.return %out_header, %out_payload : memref<2xi64>, memref<1xf64>
  }

  func.func @LyUnicodeData_Decimal(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "unicodedata.decimal", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.str", ly.runtime.primitive = "ucd_decimal", ly.runtime.result_contract = "builtins.int"} {
    %minus_one = arith.constant -1 : i64
    %cp = func.call @__ly_ucd_single_cp(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %upper, %lower, %fold, %decimal, %digit, %flags = func.call @__ly_ucd_ctype(%cp) : (i64) -> (i64, i64, i64, i64, i64, i64)
    %missing = arith.cmpi eq, %decimal, %minus_one : i64
    scf.if %missing {
      %class_id = arith.constant 53 : i64
      %length = arith.constant 13 : i64
      %static = memref.get_global @__ly_ucd_msg_not_decimal : memref<13xi8>
      %message = memref.cast %static : memref<13xi8> to memref<?xi8>
      func.call @__ly_ucd_raise(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    }
    %result:3 = func.call @LyLong_FromI64(%decimal) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyUnicodeData_Digit(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "unicodedata.digit", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.str", ly.runtime.primitive = "ucd_digit", ly.runtime.result_contract = "builtins.int"} {
    %minus_one = arith.constant -1 : i64
    %cp = func.call @__ly_ucd_single_cp(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %upper, %lower, %fold, %decimal, %digit, %flags = func.call @__ly_ucd_ctype(%cp) : (i64) -> (i64, i64, i64, i64, i64, i64)
    %missing = arith.cmpi eq, %digit, %minus_one : i64
    scf.if %missing {
      %class_id = arith.constant 53 : i64
      %length = arith.constant 11 : i64
      %static = memref.get_global @__ly_ucd_msg_not_digit : memref<11xi8>
      %message = memref.cast %static : memref<11xi8> to memref<?xi8>
      func.call @__ly_ucd_raise(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    }
    %result:3 = func.call @LyLong_FromI64(%digit) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  // ---- full case mapping helpers ----

  // (field value, is-extended) of the mapping selected by %which:
  // 0 = uppercase, 1 = lowercase, 2 = case fold.
  func.func private @__ly_ucd_select_map(%which: i64, %cp: i64) -> (i64, i1) {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %ext_upper = arith.constant 128 : i64
    %ext_lower = arith.constant 256 : i64
    %ext_fold = arith.constant 512 : i64
    %upper, %lower, %fold, %decimal, %digit, %flags = func.call @__ly_ucd_ctype(%cp) : (i64) -> (i64, i64, i64, i64, i64, i64)
    %is_upper = arith.cmpi eq, %which, %zero : i64
    %is_lower = arith.cmpi eq, %which, %one : i64
    %lower_or_fold = arith.select %is_lower, %lower, %fold : i64
    %value = arith.select %is_upper, %upper, %lower_or_fold : i64
    %lower_or_fold_bit = arith.select %is_lower, %ext_lower, %ext_fold : i64
    %bit = arith.select %is_upper, %ext_upper, %lower_or_fold_bit : i64
    %masked = arith.andi %flags, %bit : i64
    %is_ext = arith.cmpi ne, %masked, %zero : i64
    func.return %value, %is_ext : i64, i1
  }

  func.func private @__ly_ucd_map_len(%value: i64, %is_ext: i1) -> i64 {
    %one = arith.constant 1 : i64
    %mask = arith.constant 255 : i64
    %ext_len = arith.andi %value, %mask : i64
    %len = arith.select %is_ext, %ext_len, %one : i64
    func.return %len : i64
  }

  func.func private @__ly_ucd_map_cp(%cp: i64, %value: i64, %is_ext: i1, %j: i64) -> i64 {
    %result = scf.if %is_ext -> (i64) {
      %ext = func.call @__ly_ucd_ext_cp(%value, %j) : (i64, i64) -> i64
      scf.yield %ext : i64
    } else {
      %mapped = arith.addi %cp, %value : i64
      scf.yield %mapped : i64
    }
    func.return %result : i64
  }

  // Unicode Default Case Algorithm Final_Sigma: the position is preceded by
  // a cased character (after skipping case-ignorables) and not followed by
  // one (before the next non-case-ignorable).
  func.func private @__ly_ucd_sigma_final(%bytes: memref<?xi8>, %width: i64, %count: index, %i: index) -> i1 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %cased_bit = arith.constant 32 : i64
    %ignorable_bit = arith.constant 64 : i64
    %true_bit = arith.constant true
    %false_bit = arith.constant false

    %back:3 = scf.while (%j = %i, %found = %false_bit, %cased = %false_bit) : (index, i1, i1) -> (index, i1, i1) {
      %not_found = arith.xori %found, %true_bit : i1
      %has_prev = arith.cmpi ugt, %j, %c0 : index
      %continue = arith.andi %not_found, %has_prev : i1
      scf.condition(%continue) %j, %found, %cased : index, i1, i1
    } do {
    ^bb0(%j: index, %found: i1, %cased: i1):
      %prev = arith.subi %j, %c1 : index
      %cp = func.call @__ly_unicode_get(%bytes, %width, %prev) : (memref<?xi8>, i64, index) -> i64
      %upper, %lower, %fold, %decimal, %digit, %flags = func.call @__ly_ucd_ctype(%cp) : (i64) -> (i64, i64, i64, i64, i64, i64)
      %ig_masked = arith.andi %flags, %ignorable_bit : i64
      %ignorable = arith.cmpi ne, %ig_masked, %zero : i64
      %cased_masked = arith.andi %flags, %cased_bit : i64
      %is_cased = arith.cmpi ne, %cased_masked, %zero : i64
      %stop = arith.xori %ignorable, %true_bit : i1
      %next_cased = arith.select %ignorable, %false_bit, %is_cased : i1
      scf.yield %prev, %stop, %next_cased : index, i1, i1
    }
    %preceded = arith.andi %back#1, %back#2 : i1

    %start = arith.addi %i, %c1 : index
    %fwd:3 = scf.while (%j = %start, %found = %false_bit, %cased = %false_bit) : (index, i1, i1) -> (index, i1, i1) {
      %not_found = arith.xori %found, %true_bit : i1
      %has_next = arith.cmpi ult, %j, %count : index
      %continue = arith.andi %not_found, %has_next : i1
      scf.condition(%continue) %j, %found, %cased : index, i1, i1
    } do {
    ^bb0(%j: index, %found: i1, %cased: i1):
      %cp = func.call @__ly_unicode_get(%bytes, %width, %j) : (memref<?xi8>, i64, index) -> i64
      %upper, %lower, %fold, %decimal, %digit, %flags = func.call @__ly_ucd_ctype(%cp) : (i64) -> (i64, i64, i64, i64, i64, i64)
      %ig_masked = arith.andi %flags, %ignorable_bit : i64
      %ignorable = arith.cmpi ne, %ig_masked, %zero : i64
      %cased_masked = arith.andi %flags, %cased_bit : i64
      %is_cased = arith.cmpi ne, %cased_masked, %zero : i64
      %stop = arith.xori %ignorable, %true_bit : i1
      %next_cased = arith.select %ignorable, %false_bit, %is_cased : i1
      %next_j = arith.addi %j, %c1 : index
      scf.yield %next_j, %stop, %next_cased : index, i1, i1
    }
    %followed = arith.andi %fwd#1, %fwd#2 : i1
    %not_followed = arith.xori %followed, %true_bit : i1
    %final = arith.andi %preceded, %not_followed : i1
    func.return %final : i1
  }

  // Shared two-pass full case transform: %which selects the mapping
  // (0 = upper, 1 = lower with the Final_Sigma rule, 2 = casefold). Pass 1
  // measures the mapped length and the widest mapped code point so the
  // output re-canonicalizes to the smallest fitting width.
  func.func private @__ly_unicode_case_transform(%header: memref<2xi64>, %bytes: memref<?xi8>, %which: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %sigma = arith.constant 931 : i64
    %small_sigma = arith.constant 963 : i64
    %final_sigma = arith.constant 962 : i64
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %count = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %count_index = arith.index_cast %count : i64 to index
    %is_lower_mode = arith.cmpi eq, %which, %one : i64

    %scan:2 = scf.for %i = %c0 to %count_index step %c1 iter_args(%total = %zero, %maxcp = %zero) -> (i64, i64) {
      %cp = func.call @__ly_unicode_get(%bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
      %is_sigma_cp = arith.cmpi eq, %cp, %sigma : i64
      %is_sigma = arith.andi %is_sigma_cp, %is_lower_mode : i1
      %step:2 = scf.if %is_sigma -> (i64, i64) {
        scf.yield %one, %small_sigma : i64, i64
      } else {
        %value, %is_ext = func.call @__ly_ucd_select_map(%which, %cp) : (i64, i64) -> (i64, i1)
        %len = func.call @__ly_ucd_map_len(%value, %is_ext) : (i64, i1) -> i64
        %len_index = arith.index_cast %len : i64 to index
        %widest = scf.for %j = %c0 to %len_index step %c1 iter_args(%acc = %zero) -> (i64) {
          %j_i64 = arith.index_cast %j : index to i64
          %mapped = func.call @__ly_ucd_map_cp(%cp, %value, %is_ext, %j_i64) : (i64, i64, i1, i64) -> i64
          %bigger = arith.cmpi ugt, %mapped, %acc : i64
          %next = arith.select %bigger, %mapped, %acc : i64
          scf.yield %next : i64
        }
        scf.yield %len, %widest : i64, i64
      }
      %new_total = arith.addi %total, %step#0 : i64
      %bigger = arith.cmpi ugt, %step#1, %maxcp : i64
      %new_max = arith.select %bigger, %step#1, %maxcp : i64
      scf.yield %new_total, %new_max : i64, i64
    }

    %out_width = func.call @__ly_unicode_width_for(%scan#1) : (i64) -> i64
    %out_header, %out_bytes = func.call @__ly_unicode_alloc(%scan#0, %out_width) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)

    scf.for %i = %c0 to %count_index step %c1 iter_args(%pos = %c0) -> (index) {
      %cp = func.call @__ly_unicode_get(%bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
      %is_sigma_cp = arith.cmpi eq, %cp, %sigma : i64
      %is_sigma = arith.andi %is_sigma_cp, %is_lower_mode : i1
      %next_pos = scf.if %is_sigma -> (index) {
        %final = func.call @__ly_ucd_sigma_final(%bytes, %width, %count_index, %i) : (memref<?xi8>, i64, index, index) -> i1
        %mapped = arith.select %final, %final_sigma, %small_sigma : i64
        func.call @__ly_unicode_put(%out_bytes, %out_width, %pos, %mapped) : (memref<?xi8>, i64, index, i64) -> ()
        %advanced = arith.addi %pos, %c1 : index
        scf.yield %advanced : index
      } else {
        %value, %is_ext = func.call @__ly_ucd_select_map(%which, %cp) : (i64, i64) -> (i64, i1)
        %len = func.call @__ly_ucd_map_len(%value, %is_ext) : (i64, i1) -> i64
        %len_index = arith.index_cast %len : i64 to index
        scf.for %j = %c0 to %len_index step %c1 {
          %j_i64 = arith.index_cast %j : index to i64
          %mapped = func.call @__ly_ucd_map_cp(%cp, %value, %is_ext, %j_i64) : (i64, i64, i1, i64) -> i64
          %dst = arith.addi %pos, %j : index
          func.call @__ly_unicode_put(%out_bytes, %out_width, %dst, %mapped) : (memref<?xi8>, i64, index, i64) -> ()
        }
        %advanced = arith.addi %pos, %len_index : index
        scf.yield %advanced : index
      }
      scf.yield %next_pos : index
    }
    func.return %out_header, %out_bytes : memref<2xi64>, memref<?xi8>
  }

  // ---- str case-mapping runtime methods (typing exposure lands in Wave 1) ----

  func.func @LyUnicode_Upper(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "upper", ly.runtime.result_contract = "builtins.str"} {
    %which = arith.constant 0 : i64
    %result:2 = func.call @__ly_unicode_case_transform(%header, %bytes, %which) : (memref<2xi64>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_Lower(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "lower", ly.runtime.result_contract = "builtins.str"} {
    %which = arith.constant 1 : i64
    %result:2 = func.call @__ly_unicode_case_transform(%header, %bytes, %which) : (memref<2xi64>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_CaseFold(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "casefold", ly.runtime.result_contract = "builtins.str"} {
    %which = arith.constant 2 : i64
    %result:2 = func.call @__ly_unicode_case_transform(%header, %bytes, %which) : (memref<2xi64>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  // ---- str classification runtime methods ----

  func.func private @__ly_unicode_all_flagged(%header: memref<2xi64>, %bytes: memref<?xi8>, %mask: i64) -> i1 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %true_bit = arith.constant true
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %count = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %count_index = arith.index_cast %count : i64 to index
    %all = scf.for %i = %c0 to %count_index step %c1 iter_args(%acc = %true_bit) -> (i1) {
      %cp = func.call @__ly_unicode_get(%bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
      %upper, %lower, %fold, %decimal, %digit, %flags = func.call @__ly_ucd_ctype(%cp) : (i64) -> (i64, i64, i64, i64, i64, i64)
      %masked = arith.andi %flags, %mask : i64
      %has = arith.cmpi ne, %masked, %zero : i64
      %next = arith.andi %acc, %has : i1
      scf.yield %next : i1
    }
    %nonempty = arith.cmpi sgt, %count, %zero : i64
    %result = arith.andi %nonempty, %all : i1
    func.return %result : i1
  }

  func.func @LyUnicode_IsAlpha(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> i1 attributes {ly.runtime.contract = "builtins.str", ly.runtime.method = "isalpha"} {
    %mask = arith.constant 1 : i64
    %result = func.call @__ly_unicode_all_flagged(%header, %bytes, %mask) : (memref<2xi64>, memref<?xi8>, i64) -> i1
    func.return %result : i1
  }

  func.func @LyUnicode_IsSpace(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> i1 attributes {ly.runtime.contract = "builtins.str", ly.runtime.method = "isspace"} {
    %mask = arith.constant 16 : i64
    %result = func.call @__ly_unicode_all_flagged(%header, %bytes, %mask) : (memref<2xi64>, memref<?xi8>, i64) -> i1
    func.return %result : i1
  }

  // All characters carry the ctype field selected by %which (0 = decimal,
  // 1 = digit) and the string is nonempty.
  func.func private @__ly_unicode_all_valued(%header: memref<2xi64>, %bytes: memref<?xi8>, %which: i64) -> i1 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %minus_one = arith.constant -1 : i64
    %true_bit = arith.constant true
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %count = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %count_index = arith.index_cast %count : i64 to index
    %is_decimal_mode = arith.cmpi eq, %which, %zero : i64
    %all = scf.for %i = %c0 to %count_index step %c1 iter_args(%acc = %true_bit) -> (i1) {
      %cp = func.call @__ly_unicode_get(%bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
      %upper, %lower, %fold, %decimal, %digit, %flags = func.call @__ly_ucd_ctype(%cp) : (i64) -> (i64, i64, i64, i64, i64, i64)
      %value = arith.select %is_decimal_mode, %decimal, %digit : i64
      %has = arith.cmpi ne, %value, %minus_one : i64
      %next = arith.andi %acc, %has : i1
      scf.yield %next : i1
    }
    %nonempty = arith.cmpi sgt, %count, %zero : i64
    %result = arith.andi %nonempty, %all : i1
    func.return %result : i1
  }

  func.func @LyUnicode_IsDecimal(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> i1 attributes {ly.runtime.contract = "builtins.str", ly.runtime.method = "isdecimal"} {
    %which = arith.constant 0 : i64
    %result = func.call @__ly_unicode_all_valued(%header, %bytes, %which) : (memref<2xi64>, memref<?xi8>, i64) -> i1
    func.return %result : i1
  }

  func.func @LyUnicode_IsDigit(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> i1 attributes {ly.runtime.contract = "builtins.str", ly.runtime.method = "isdigit"} {
    %which = arith.constant 1 : i64
    %result = func.call @__ly_unicode_all_valued(%header, %bytes, %which) : (memref<2xi64>, memref<?xi8>, i64) -> i1
    func.return %result : i1
  }

  func.func @LyUnicode_IsNumeric(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> i1 attributes {ly.runtime.contract = "builtins.str", ly.runtime.method = "isnumeric"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %minus_one = arith.constant -1 : i64
    %true_bit = arith.constant true
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %count = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %count_index = arith.index_cast %count : i64 to index
    %all = scf.for %i = %c0 to %count_index step %c1 iter_args(%acc = %true_bit) -> (i1) {
      %cp = func.call @__ly_unicode_get(%bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
      %cat, %numeric = func.call @__ly_ucd_info(%cp) : (i64) -> (i64, i64)
      %has = arith.cmpi ne, %numeric, %minus_one : i64
      %next = arith.andi %acc, %has : i1
      scf.yield %next : i1
    }
    %nonempty = arith.cmpi sgt, %count, %zero : i64
    %result = arith.andi %nonempty, %all : i1
    func.return %result : i1
  }

  // CPython unicode_isupper/islower: at least one cased character, and no
  // character of the opposite case (title-case counts as both-ish: it fails
  // either predicate).
  func.func private @__ly_unicode_case_predicate(%header: memref<2xi64>, %bytes: memref<?xi8>, %fail_mask: i64, %want_mask: i64) -> i1 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %true_bit = arith.constant true
    %false_bit = arith.constant false
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %count = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %count_index = arith.index_cast %count : i64 to index
    %scan:2 = scf.for %i = %c0 to %count_index step %c1 iter_args(%ok = %true_bit, %cased = %false_bit) -> (i1, i1) {
      %cp = func.call @__ly_unicode_get(%bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
      %upper, %lower, %fold, %decimal, %digit, %flags = func.call @__ly_ucd_ctype(%cp) : (i64) -> (i64, i64, i64, i64, i64, i64)
      %fail_masked = arith.andi %flags, %fail_mask : i64
      %fails = arith.cmpi ne, %fail_masked, %zero : i64
      %not_fails = arith.xori %fails, %true_bit : i1
      %next_ok = arith.andi %ok, %not_fails : i1
      %want_masked = arith.andi %flags, %want_mask : i64
      %wants = arith.cmpi ne, %want_masked, %zero : i64
      %next_cased = arith.ori %cased, %wants : i1
      scf.yield %next_ok, %next_cased : i1, i1
    }
    %result = arith.andi %scan#0, %scan#1 : i1
    func.return %result : i1
  }

  func.func @LyUnicode_IsUpper(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> i1 attributes {ly.runtime.contract = "builtins.str", ly.runtime.method = "isupper"} {
    %fail_mask = arith.constant 10 : i64  // LOWER | TITLE
    %want_mask = arith.constant 4 : i64   // UPPER
    %result = func.call @__ly_unicode_case_predicate(%header, %bytes, %fail_mask, %want_mask) : (memref<2xi64>, memref<?xi8>, i64, i64) -> i1
    func.return %result : i1
  }

  func.func @LyUnicode_IsLower(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> i1 attributes {ly.runtime.contract = "builtins.str", ly.runtime.method = "islower"} {
    %fail_mask = arith.constant 12 : i64  // UPPER | TITLE
    %want_mask = arith.constant 2 : i64   // LOWER
    %result = func.call @__ly_unicode_case_predicate(%header, %bytes, %fail_mask, %want_mask) : (memref<2xi64>, memref<?xi8>, i64, i64) -> i1
    func.return %result : i1
  }

  // Printability per CPython str.isprintable / repr: everything except
  // categories Cn, Zs, Zl, Zp, Cc, Cf, Cs, Co -- with U+0020 SPACE as the
  // sole Zs exception. The mask packs those category-enum bits
  // (Cn=0, Zs=23, Zl=24, Zp=25, Cc=26, Cf=27, Cs=28, Co=29).
  func.func private @__ly_ucd_is_printable(%cp: i64) -> i1 {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %space = arith.constant 32 : i64
    %mask = arith.constant 1065353217 : i64
    %cat, %numeric = func.call @__ly_ucd_info(%cp) : (i64) -> (i64, i64)
    %shifted = arith.shrui %mask, %cat : i64
    %bit = arith.andi %shifted, %one : i64
    %unprintable = arith.cmpi ne, %bit, %zero : i64
    %true_bit = arith.constant true
    %printable_cat = arith.xori %unprintable, %true_bit : i1
    %is_space = arith.cmpi eq, %cp, %space : i64
    %printable = arith.ori %printable_cat, %is_space : i1
    func.return %printable : i1
  }

  func.func @LyUnicode_IsPrintable(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> i1 attributes {ly.runtime.contract = "builtins.str", ly.runtime.method = "isprintable"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %true_bit = arith.constant true
    %width = func.call @__ly_unicode_width(%header) : (memref<2xi64>) -> i64
    %count = func.call @__ly_unicode_count(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %count_index = arith.index_cast %count : i64 to index
    %all = scf.for %i = %c0 to %count_index step %c1 iter_args(%acc = %true_bit) -> (i1) {
      %cp = func.call @__ly_unicode_get(%bytes, %width, %i) : (memref<?xi8>, i64, index) -> i64
      %printable = func.call @__ly_ucd_is_printable(%cp) : (i64) -> i1
      %next = arith.andi %acc, %printable : i1
      scf.yield %next : i1
    }
    func.return %all : i1
  }
}
