// The format-spec mini-language -- CPython's Python/formatter_unicode.c:
// parsing the spec, and rendering an int, float or str by it.

module {
  // ===== declared here, defined in another runtime file or built by the lowering =====
  func.func private @__ly_check_alloc_count(%count: i64, %element_bytes: i64, %extra: i64)
  func.func private @__ly_long_bit_length(%meta: memref<2xi64>, %digits: memref<?xi32>) -> i64
  func.func private @__ly_raise_static_message(%class_id: i64, %message: memref<?xi8>, %length: i64)
  func.func private @__ly_unicode_alloc(%count: i64, %width: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.primitive = "alloc"}
  func.func private @__ly_unicode_count(%header: memref<2xi64>, %bytes: memref<?xi8>) -> i64
  func.func private @__ly_unicode_get(%bytes: memref<?xi8>, %width: i64, %i: index) -> i64
  func.func private @__ly_unicode_put(%bytes: memref<?xi8>, %width: i64, %i: index, %cp: i64)
  func.func private @__ly_unicode_utf8_fill(%header: memref<2xi64>, %bytes: memref<?xi8>, %out: memref<?xi8>)
  func.func private @__ly_unicode_utf8_length(%header: memref<2xi64>, %bytes: memref<?xi8>) -> i64
  func.func private @__ly_unicode_width(%header: memref<2xi64>) -> i64
  func.func private @__ly_unicode_width_for(%cp: i64) -> i64

  // ===== impls: format spec mini-language =====
  // Shared runtime for str/int/float/bool __format__. The parsed spec
  // travels as a 10-slot i64 record: [0] fill cp (-1 unset), [1] align cp
  // (0 unset), [2] sign cp (0 unset), [3] '#' flag, [4] '0' flag,
  // [5] width (-1 unset), [6] grouping cp (0 none), [7] precision
  // (-1 unset), [8] type cp (0 none), [9] 'z' flag. Formatting works on
  // code-point buffers (i32) because fills and 'c' bodies are arbitrary
  // code points; the ASCII digit machinery widens into them for free.

  memref.global "private" constant @__ly_fmt_msg_invalid_spec_prefix : memref<26xi8> = dense<[73, 110, 118, 97, 108, 105, 100, 32, 102, 111, 114, 109, 97, 116, 32, 115, 112, 101, 99, 105, 102, 105, 101, 114, 32, 39]>
  memref.global "private" constant @__ly_fmt_msg_for_object_of_type : memref<22xi8> = dense<[39, 32, 102, 111, 114, 32, 111, 98, 106, 101, 99, 116, 32, 111, 102, 32, 116, 121, 112, 101, 32, 39]>
  memref.global "private" constant @__ly_fmt_msg_quote : memref<1xi8> = dense<[39]>
  memref.global "private" constant @__ly_fmt_msg_unknown_code_prefix : memref<21xi8> = dense<[85, 110, 107, 110, 111, 119, 110, 32, 102, 111, 114, 109, 97, 116, 32, 99, 111, 100, 101, 32, 39]>
  memref.global "private" constant @__ly_fmt_msg_sign_str : memref<43xi8> = dense<[83, 105, 103, 110, 32, 110, 111, 116, 32, 97, 108, 108, 111, 119, 101, 100, 32, 105, 110, 32, 115, 116, 114, 105, 110, 103, 32, 102, 111, 114, 109, 97, 116, 32, 115, 112, 101, 99, 105, 102, 105, 101, 114]>
  memref.global "private" constant @__ly_fmt_msg_eq_align_str : memref<52xi8> = dense<[39, 61, 39, 32, 97, 108, 105, 103, 110, 109, 101, 110, 116, 32, 110, 111, 116, 32, 97, 108, 108, 111, 119, 101, 100, 32, 105, 110, 32, 115, 116, 114, 105, 110, 103, 32, 102, 111, 114, 109, 97, 116, 32, 115, 112, 101, 99, 105, 102, 105, 101, 114]>
  memref.global "private" constant @__ly_fmt_msg_alt_str : memref<57xi8> = dense<[65, 108, 116, 101, 114, 110, 97, 116, 101, 32, 102, 111, 114, 109, 32, 40, 35, 41, 32, 110, 111, 116, 32, 97, 108, 108, 111, 119, 101, 100, 32, 105, 110, 32, 115, 116, 114, 105, 110, 103, 32, 102, 111, 114, 109, 97, 116, 32, 115, 112, 101, 99, 105, 102, 105, 101, 114]>
  memref.global "private" constant @__ly_fmt_msg_cannot_specify_prefix : memref<16xi8> = dense<[67, 97, 110, 110, 111, 116, 32, 115, 112, 101, 99, 105, 102, 121, 32, 39]>
  memref.global "private" constant @__ly_fmt_msg_with_mid : memref<8xi8> = dense<[39, 32, 119, 105, 116, 104, 32, 39]>
  memref.global "private" constant @__ly_fmt_msg_dot_tail : memref<2xi8> = dense<[39, 46]>
  memref.global "private" constant @__ly_fmt_msg_both_groupings : memref<32xi8> = dense<[67, 97, 110, 110, 111, 116, 32, 115, 112, 101, 99, 105, 102, 121, 32, 98, 111, 116, 104, 32, 39, 44, 39, 32, 97, 110, 100, 32, 39, 95, 39, 46]>
  memref.global "private" constant @__ly_fmt_msg_missing_precision : memref<34xi8> = dense<[70, 111, 114, 109, 97, 116, 32, 115, 112, 101, 99, 105, 102, 105, 101, 114, 32, 109, 105, 115, 115, 105, 110, 103, 32, 112, 114, 101, 99, 105, 115, 105, 111, 110]>
  memref.global "private" constant @__ly_fmt_msg_int_precision : memref<49xi8> = dense<[80, 114, 101, 99, 105, 115, 105, 111, 110, 32, 110, 111, 116, 32, 97, 108, 108, 111, 119, 101, 100, 32, 105, 110, 32, 105, 110, 116, 101, 103, 101, 114, 32, 102, 111, 114, 109, 97, 116, 32, 115, 112, 101, 99, 105, 102, 105, 101, 114]>
  memref.global "private" constant @__ly_fmt_msg_sign_c : memref<50xi8> = dense<[83, 105, 103, 110, 32, 110, 111, 116, 32, 97, 108, 108, 111, 119, 101, 100, 32, 119, 105, 116, 104, 32, 105, 110, 116, 101, 103, 101, 114, 32, 102, 111, 114, 109, 97, 116, 32, 115, 112, 101, 99, 105, 102, 105, 101, 114, 32, 39, 99, 39]>
  memref.global "private" constant @__ly_fmt_msg_alt_c : memref<64xi8> = dense<[65, 108, 116, 101, 114, 110, 97, 116, 101, 32, 102, 111, 114, 109, 32, 40, 35, 41, 32, 110, 111, 116, 32, 97, 108, 108, 111, 119, 101, 100, 32, 119, 105, 116, 104, 32, 105, 110, 116, 101, 103, 101, 114, 32, 102, 111, 114, 109, 97, 116, 32, 115, 112, 101, 99, 105, 102, 105, 101, 114, 32, 39, 99, 39]>
  memref.global "private" constant @__ly_fmt_msg_z_str : memref<65xi8> = dense<[78, 101, 103, 97, 116, 105, 118, 101, 32, 122, 101, 114, 111, 32, 99, 111, 101, 114, 99, 105, 111, 110, 32, 40, 122, 41, 32, 110, 111, 116, 32, 97, 108, 108, 111, 119, 101, 100, 32, 105, 110, 32, 115, 116, 114, 105, 110, 103, 32, 102, 111, 114, 109, 97, 116, 32, 115, 112, 101, 99, 105, 102, 105, 101, 114]>
  memref.global "private" constant @__ly_fmt_msg_z_int : memref<66xi8> = dense<[78, 101, 103, 97, 116, 105, 118, 101, 32, 122, 101, 114, 111, 32, 99, 111, 101, 114, 99, 105, 111, 110, 32, 40, 122, 41, 32, 110, 111, 116, 32, 97, 108, 108, 111, 119, 101, 100, 32, 105, 110, 32, 105, 110, 116, 101, 103, 101, 114, 32, 102, 111, 114, 109, 97, 116, 32, 115, 112, 101, 99, 105, 102, 105, 101, 114]>
  memref.global "private" constant @__ly_fmt_msg_c_range : memref<29xi8> = dense<[37, 99, 32, 97, 114, 103, 32, 110, 111, 116, 32, 105, 110, 32, 114, 97, 110, 103, 101, 40, 48, 120, 49, 49, 48, 48, 48, 48, 41]>
  memref.global "private" constant @__ly_fmt_msg_name_int : memref<3xi8> = dense<[105, 110, 116]>
  memref.global "private" constant @__ly_fmt_msg_name_float : memref<5xi8> = dense<[102, 108, 111, 97, 116]>
  memref.global "private" constant @__ly_fmt_msg_name_str : memref<3xi8> = dense<[115, 116, 114]>
  memref.global "private" constant @__ly_fmt_msg_name_bool : memref<4xi8> = dense<[98, 111, 111, 108]>

  func.func private @__ly_fmt_copy_bytes(%dst: memref<?xi8>, %dpos: i64, %src: memref<?xi8>, %len: i64) -> i64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %n = arith.index_cast %len : i64 to index
    %base = arith.index_cast %dpos : i64 to index
    scf.for %i = %c0 to %n step %c1 {
      %b = memref.load %src[%i] : memref<?xi8>
      %o = arith.addi %base, %i : index
      memref.store %b, %dst[%o] : memref<?xi8>
    }
    %res = arith.addi %dpos, %len : i64
    func.return %res : i64
  }

  // UTF-8 encode one code point at byte position; returns the new position.
  func.func private @__ly_fmt_utf8_put(%buf: memref<?xi8>, %pos: i64, %cp: i64) -> i64 {
    %c0_1 = arith.constant 1 : index
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %four = arith.constant 4 : i64
    %c6 = arith.constant 6 : i64
    %c12 = arith.constant 12 : i64
    %c18 = arith.constant 18 : i64
    %mask6 = arith.constant 63 : i64
    %cont = arith.constant 128 : i64
    %lim1 = arith.constant 128 : i64
    %lim2 = arith.constant 2048 : i64
    %lim3 = arith.constant 65536 : i64
    %is1 = arith.cmpi ult, %cp, %lim1 : i64
    %is2 = arith.cmpi ult, %cp, %lim2 : i64
    %is3 = arith.cmpi ult, %cp, %lim3 : i64
    %p_idx = arith.index_cast %pos : i64 to index
    %res = scf.if %is1 -> (i64) {
      %b = arith.trunci %cp : i64 to i8
      memref.store %b, %buf[%p_idx] : memref<?xi8>
      %np = arith.addi %pos, %one : i64
      scf.yield %np : i64
    } else {
      %r2 = scf.if %is2 -> (i64) {
        %hi = arith.shrui %cp, %c6 : i64
        %h192 = arith.constant 192 : i64
        %hb0 = arith.andi %hi, %mask6 : i64
        %hb = arith.ori %hb0, %h192 : i64
        %lo0 = arith.andi %cp, %mask6 : i64
        %lo = arith.ori %lo0, %cont : i64
        %hb8 = arith.trunci %hb : i64 to i8
        %lo8 = arith.trunci %lo : i64 to i8
        memref.store %hb8, %buf[%p_idx] : memref<?xi8>
        %p1 = arith.addi %p_idx, %c0_1 : index
        memref.store %lo8, %buf[%p1] : memref<?xi8>
        %np = arith.addi %pos, %two : i64
        scf.yield %np : i64
      } else {
        %r3 = scf.if %is3 -> (i64) {
          %h224 = arith.constant 224 : i64
          %b0s = arith.shrui %cp, %c12 : i64
          %b0m = arith.andi %b0s, %mask6 : i64
          %b0 = arith.ori %b0m, %h224 : i64
          %b1s = arith.shrui %cp, %c6 : i64
          %b1m = arith.andi %b1s, %mask6 : i64
          %b1 = arith.ori %b1m, %cont : i64
          %b2m = arith.andi %cp, %mask6 : i64
          %b2 = arith.ori %b2m, %cont : i64
          %b0_8 = arith.trunci %b0 : i64 to i8
          %b1_8 = arith.trunci %b1 : i64 to i8
          %b2_8 = arith.trunci %b2 : i64 to i8
          memref.store %b0_8, %buf[%p_idx] : memref<?xi8>
          %p1a = arith.addi %p_idx, %c0_1 : index
          memref.store %b1_8, %buf[%p1a] : memref<?xi8>
          %p2a = arith.addi %p1a, %c0_1 : index
          memref.store %b2_8, %buf[%p2a] : memref<?xi8>
          %np = arith.addi %pos, %three : i64
          scf.yield %np : i64
        } else {
          %h240 = arith.constant 240 : i64
          %b0s = arith.shrui %cp, %c18 : i64
          %b0m = arith.andi %b0s, %mask6 : i64
          %b0 = arith.ori %b0m, %h240 : i64
          %b1s = arith.shrui %cp, %c12 : i64
          %b1m = arith.andi %b1s, %mask6 : i64
          %b1 = arith.ori %b1m, %cont : i64
          %b2s = arith.shrui %cp, %c6 : i64
          %b2m = arith.andi %b2s, %mask6 : i64
          %b2 = arith.ori %b2m, %cont : i64
          %b3m = arith.andi %cp, %mask6 : i64
          %b3 = arith.ori %b3m, %cont : i64
          %b0_8 = arith.trunci %b0 : i64 to i8
          %b1_8 = arith.trunci %b1 : i64 to i8
          %b2_8 = arith.trunci %b2 : i64 to i8
          %b3_8 = arith.trunci %b3 : i64 to i8
          memref.store %b0_8, %buf[%p_idx] : memref<?xi8>
          %p1b = arith.addi %p_idx, %c0_1 : index
          memref.store %b1_8, %buf[%p1b] : memref<?xi8>
          %p2b = arith.addi %p1b, %c0_1 : index
          memref.store %b2_8, %buf[%p2b] : memref<?xi8>
          %p3b = arith.addi %p2b, %c0_1 : index
          memref.store %b3_8, %buf[%p3b] : memref<?xi8>
          %np = arith.addi %pos, %four : i64
          scf.yield %np : i64
        }
        scf.yield %r3 : i64
      }
      scf.yield %r2 : i64
    }
    func.return %res : i64
  }

  func.func private @__ly_fmt_raise_bytes(%message: memref<?xi8>, %length: i64) {
    %value_error = arith.constant {ly.class_id_of = "builtins.ValueError"} 53 : i64
    func.call @__ly_raise_static_message(%value_error, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  // "Unknown format code 'C' for object of type 'T'"
  func.func private @__ly_fmt_raise_unknown_code(%code: i64, %name: memref<?xi8>, %name_len: i64) {
    %buf_s = memref.alloca() : memref<64xi8>
    %buf = memref.cast %buf_s : memref<64xi8> to memref<?xi8>
    %zero = arith.constant 0 : i64
    %p0s = memref.get_global @__ly_fmt_msg_unknown_code_prefix : memref<21xi8>
    %p0 = memref.cast %p0s : memref<21xi8> to memref<?xi8>
    %l21 = arith.constant 21 : i64
    %a = func.call @__ly_fmt_copy_bytes(%buf, %zero, %p0, %l21) : (memref<?xi8>, i64, memref<?xi8>, i64) -> i64
    %b = func.call @__ly_fmt_utf8_put(%buf, %a, %code) : (memref<?xi8>, i64, i64) -> i64
    %p1s = memref.get_global @__ly_fmt_msg_for_object_of_type : memref<22xi8>
    %p1 = memref.cast %p1s : memref<22xi8> to memref<?xi8>
    %l22 = arith.constant 22 : i64
    %c = func.call @__ly_fmt_copy_bytes(%buf, %b, %p1, %l22) : (memref<?xi8>, i64, memref<?xi8>, i64) -> i64
    %d = func.call @__ly_fmt_copy_bytes(%buf, %c, %name, %name_len) : (memref<?xi8>, i64, memref<?xi8>, i64) -> i64
    %qs = memref.get_global @__ly_fmt_msg_quote : memref<1xi8>
    %q = memref.cast %qs : memref<1xi8> to memref<?xi8>
    %l1 = arith.constant 1 : i64
    %e = func.call @__ly_fmt_copy_bytes(%buf, %d, %q, %l1) : (memref<?xi8>, i64, memref<?xi8>, i64) -> i64
    func.call @__ly_fmt_raise_bytes(%buf, %e) : (memref<?xi8>, i64) -> ()
    func.return
  }

  // "Invalid format specifier 'SPEC' for object of type 'T'"
  func.func private @__ly_fmt_raise_invalid_spec(%spec_h: memref<2xi64>, %spec_b: memref<?xi8>, %name: memref<?xi8>, %name_len: i64) {
    %zero = arith.constant 0 : i64
    %spec_len = func.call @__ly_unicode_utf8_length(%spec_h, %spec_b) : (memref<2xi64>, memref<?xi8>) -> i64
    %fixed = arith.constant 64 : i64
    %total = arith.addi %spec_len, %fixed : i64
    %total_idx = arith.index_cast %total : i64 to index
    %buf = memref.alloc(%total_idx) : memref<?xi8>
    %spec_len_idx = arith.index_cast %spec_len : i64 to index
    %tmp = memref.alloc(%spec_len_idx) : memref<?xi8>
    func.call @__ly_unicode_utf8_fill(%spec_h, %spec_b, %tmp) : (memref<2xi64>, memref<?xi8>, memref<?xi8>) -> ()
    %p0s = memref.get_global @__ly_fmt_msg_invalid_spec_prefix : memref<26xi8>
    %p0 = memref.cast %p0s : memref<26xi8> to memref<?xi8>
    %l26 = arith.constant 26 : i64
    %a = func.call @__ly_fmt_copy_bytes(%buf, %zero, %p0, %l26) : (memref<?xi8>, i64, memref<?xi8>, i64) -> i64
    %b = func.call @__ly_fmt_copy_bytes(%buf, %a, %tmp, %spec_len) : (memref<?xi8>, i64, memref<?xi8>, i64) -> i64
    %p1s = memref.get_global @__ly_fmt_msg_for_object_of_type : memref<22xi8>
    %p1 = memref.cast %p1s : memref<22xi8> to memref<?xi8>
    %l22 = arith.constant 22 : i64
    %c = func.call @__ly_fmt_copy_bytes(%buf, %b, %p1, %l22) : (memref<?xi8>, i64, memref<?xi8>, i64) -> i64
    %d = func.call @__ly_fmt_copy_bytes(%buf, %c, %name, %name_len) : (memref<?xi8>, i64, memref<?xi8>, i64) -> i64
    %qs = memref.get_global @__ly_fmt_msg_quote : memref<1xi8>
    %q = memref.cast %qs : memref<1xi8> to memref<?xi8>
    %l1 = arith.constant 1 : i64
    %e = func.call @__ly_fmt_copy_bytes(%buf, %d, %q, %l1) : (memref<?xi8>, i64, memref<?xi8>, i64) -> i64
    memref.dealloc %tmp : memref<?xi8>
    func.call @__ly_fmt_raise_bytes(%buf, %e) : (memref<?xi8>, i64) -> ()
    memref.dealloc %buf : memref<?xi8>
    func.return
  }

  // "Cannot specify 'G' with 'C'."
  func.func private @__ly_fmt_raise_cannot_group(%gcp: i64, %wcp: i64) {
    %buf_s = memref.alloca() : memref<48xi8>
    %buf = memref.cast %buf_s : memref<48xi8> to memref<?xi8>
    %zero = arith.constant 0 : i64
    %p0s = memref.get_global @__ly_fmt_msg_cannot_specify_prefix : memref<16xi8>
    %p0 = memref.cast %p0s : memref<16xi8> to memref<?xi8>
    %l16 = arith.constant 16 : i64
    %a = func.call @__ly_fmt_copy_bytes(%buf, %zero, %p0, %l16) : (memref<?xi8>, i64, memref<?xi8>, i64) -> i64
    %b = func.call @__ly_fmt_utf8_put(%buf, %a, %gcp) : (memref<?xi8>, i64, i64) -> i64
    %p1s = memref.get_global @__ly_fmt_msg_with_mid : memref<8xi8>
    %p1 = memref.cast %p1s : memref<8xi8> to memref<?xi8>
    %l8 = arith.constant 8 : i64
    %c = func.call @__ly_fmt_copy_bytes(%buf, %b, %p1, %l8) : (memref<?xi8>, i64, memref<?xi8>, i64) -> i64
    %d = func.call @__ly_fmt_utf8_put(%buf, %c, %wcp) : (memref<?xi8>, i64, i64) -> i64
    %p2s = memref.get_global @__ly_fmt_msg_dot_tail : memref<2xi8>
    %p2 = memref.cast %p2s : memref<2xi8> to memref<?xi8>
    %l2 = arith.constant 2 : i64
    %e = func.call @__ly_fmt_copy_bytes(%buf, %d, %p2, %l2) : (memref<?xi8>, i64, memref<?xi8>, i64) -> i64
    func.call @__ly_fmt_raise_bytes(%buf, %e) : (memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @__ly_fmt_raise_sign_str() {
    %ms = memref.get_global @__ly_fmt_msg_sign_str : memref<43xi8>
    %m = memref.cast %ms : memref<43xi8> to memref<?xi8>
    %l = arith.constant 43 : i64
    func.call @__ly_fmt_raise_bytes(%m, %l) : (memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @__ly_fmt_raise_eq_align_str() {
    %ms = memref.get_global @__ly_fmt_msg_eq_align_str : memref<52xi8>
    %m = memref.cast %ms : memref<52xi8> to memref<?xi8>
    %l = arith.constant 52 : i64
    func.call @__ly_fmt_raise_bytes(%m, %l) : (memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @__ly_fmt_raise_alt_str() {
    %ms = memref.get_global @__ly_fmt_msg_alt_str : memref<57xi8>
    %m = memref.cast %ms : memref<57xi8> to memref<?xi8>
    %l = arith.constant 57 : i64
    func.call @__ly_fmt_raise_bytes(%m, %l) : (memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @__ly_fmt_raise_both_groupings() {
    %ms = memref.get_global @__ly_fmt_msg_both_groupings : memref<32xi8>
    %m = memref.cast %ms : memref<32xi8> to memref<?xi8>
    %l = arith.constant 32 : i64
    func.call @__ly_fmt_raise_bytes(%m, %l) : (memref<?xi8>, i64) -> ()
    func.return
  }

  // CPython's get_integer past PY_SSIZE_T_MAX, for a width or a precision.
  memref.global "private" constant @__ly_fmt_msg_too_many_digits : memref<40xi8> = dense<[84, 111, 111, 32, 109, 97, 110, 121, 32, 100, 101, 99, 105, 109, 97, 108, 32, 100, 105, 103, 105, 116, 115, 32, 105, 110, 32, 102, 111, 114, 109, 97, 116, 32, 115, 116, 114, 105, 110, 103]>
  func.func private @__ly_fmt_raise_too_many_digits() {
    %ms = memref.get_global @__ly_fmt_msg_too_many_digits : memref<40xi8>
    %m = memref.cast %ms : memref<40xi8> to memref<?xi8>
    %l = arith.constant 40 : i64
    func.call @__ly_fmt_raise_bytes(%m, %l) : (memref<?xi8>, i64) -> ()
    func.return
  }

  // CPython's float formatting past INT_MAX digits of precision.
  memref.global "private" constant @__ly_fmt_msg_precision_too_big : memref<17xi8> = dense<[112, 114, 101, 99, 105, 115, 105, 111, 110, 32, 116, 111, 111, 32, 98, 105, 103]>
  func.func private @__ly_fmt_raise_precision_too_big() {
    %ms = memref.get_global @__ly_fmt_msg_precision_too_big : memref<17xi8>
    %m = memref.cast %ms : memref<17xi8> to memref<?xi8>
    %l = arith.constant 17 : i64
    func.call @__ly_fmt_raise_bytes(%m, %l) : (memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @__ly_fmt_raise_missing_precision() {
    %ms = memref.get_global @__ly_fmt_msg_missing_precision : memref<34xi8>
    %m = memref.cast %ms : memref<34xi8> to memref<?xi8>
    %l = arith.constant 34 : i64
    func.call @__ly_fmt_raise_bytes(%m, %l) : (memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @__ly_fmt_raise_int_precision() {
    %ms = memref.get_global @__ly_fmt_msg_int_precision : memref<49xi8>
    %m = memref.cast %ms : memref<49xi8> to memref<?xi8>
    %l = arith.constant 49 : i64
    func.call @__ly_fmt_raise_bytes(%m, %l) : (memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @__ly_fmt_raise_sign_c() {
    %ms = memref.get_global @__ly_fmt_msg_sign_c : memref<50xi8>
    %m = memref.cast %ms : memref<50xi8> to memref<?xi8>
    %l = arith.constant 50 : i64
    func.call @__ly_fmt_raise_bytes(%m, %l) : (memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @__ly_fmt_raise_alt_c() {
    %ms = memref.get_global @__ly_fmt_msg_alt_c : memref<64xi8>
    %m = memref.cast %ms : memref<64xi8> to memref<?xi8>
    %l = arith.constant 64 : i64
    func.call @__ly_fmt_raise_bytes(%m, %l) : (memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @__ly_fmt_raise_z_int() {
    %ms = memref.get_global @__ly_fmt_msg_z_int : memref<66xi8>
    %m = memref.cast %ms : memref<66xi8> to memref<?xi8>
    %l = arith.constant 66 : i64
    func.call @__ly_fmt_raise_bytes(%m, %l) : (memref<?xi8>, i64) -> ()
    func.return
  }

  // OverflowError in CPython; ValueError until the R2 taxonomy port lands.
  func.func private @__ly_fmt_raise_c_range() {
    %ms = memref.get_global @__ly_fmt_msg_c_range : memref<29xi8>
    %m = memref.cast %ms : memref<29xi8> to memref<?xi8>
    %l = arith.constant 29 : i64
    func.call @__ly_fmt_raise_bytes(%m, %l) : (memref<?xi8>, i64) -> ()
    func.return
  }

  // Parses the CPython format-spec mini-language into the 10-slot record.
  // Returns false when trailing characters remain (the caller raises with
  // its own type name); the shared shape errors raise here directly.
  func.func private @__ly_fmt_parse_spec(%spec_h: memref<2xi64>, %spec_b: memref<?xi8>, %out: memref<?xi64>) -> i1 {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %minus_one = arith.constant -1 : i64
    %ten = arith.constant 10 : i64
    %ssize_max = arith.constant 9223372036854775807 : i64
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
    memref.store %minus_one, %out[%s0] : memref<?xi64>
    memref.store %zero, %out[%s1] : memref<?xi64>
    memref.store %zero, %out[%s2] : memref<?xi64>
    memref.store %zero, %out[%s3] : memref<?xi64>
    memref.store %zero, %out[%s4] : memref<?xi64>
    memref.store %minus_one, %out[%s5] : memref<?xi64>
    memref.store %zero, %out[%s6] : memref<?xi64>
    memref.store %minus_one, %out[%s7] : memref<?xi64>
    memref.store %zero, %out[%s8] : memref<?xi64>
    memref.store %zero, %out[%s9] : memref<?xi64>
    %wid = func.call @__ly_unicode_width(%spec_h) : (memref<2xi64>) -> i64
    %n = func.call @__ly_unicode_count(%spec_h, %spec_b) : (memref<2xi64>, memref<?xi8>) -> i64

    // fill+align (two-char), else align (one-char)
    %has2 = arith.cmpi sge, %n, %two : i64
    %c1v = scf.if %has2 -> (i64) {
      %i1x = arith.constant 1 : index
      %v = func.call @__ly_unicode_get(%spec_b, %wid, %i1x) : (memref<?xi8>, i64, index) -> i64
      scf.yield %v : i64
    } else {
      scf.yield %zero : i64
    }
    %lt = arith.constant 60 : i64
    %gt = arith.constant 62 : i64
    %caret = arith.constant 94 : i64
    %eq = arith.constant 61 : i64
    %c1_lt = arith.cmpi eq, %c1v, %lt : i64
    %c1_gt = arith.cmpi eq, %c1v, %gt : i64
    %c1_ca = arith.cmpi eq, %c1v, %caret : i64
    %c1_eq = arith.cmpi eq, %c1v, %eq : i64
    %c1_a = arith.ori %c1_lt, %c1_gt : i1
    %c1_b = arith.ori %c1_ca, %c1_eq : i1
    %c1_align = arith.ori %c1_a, %c1_b : i1
    %two_char = arith.andi %has2, %c1_align : i1
    %pos_align = scf.if %two_char -> (i64) {
      %i0x = arith.constant 0 : index
      %fillv = func.call @__ly_unicode_get(%spec_b, %wid, %i0x) : (memref<?xi8>, i64, index) -> i64
      memref.store %fillv, %out[%s0] : memref<?xi64>
      memref.store %c1v, %out[%s1] : memref<?xi64>
      scf.yield %two : i64
    } else {
      %has1 = arith.cmpi sge, %n, %one : i64
      %c0v = scf.if %has1 -> (i64) {
        %i0y = arith.constant 0 : index
        %v = func.call @__ly_unicode_get(%spec_b, %wid, %i0y) : (memref<?xi8>, i64, index) -> i64
        scf.yield %v : i64
      } else {
        scf.yield %zero : i64
      }
      %c0_lt = arith.cmpi eq, %c0v, %lt : i64
      %c0_gt = arith.cmpi eq, %c0v, %gt : i64
      %c0_ca = arith.cmpi eq, %c0v, %caret : i64
      %c0_eq = arith.cmpi eq, %c0v, %eq : i64
      %c0_a = arith.ori %c0_lt, %c0_gt : i1
      %c0_b = arith.ori %c0_ca, %c0_eq : i1
      %c0_align0 = arith.ori %c0_a, %c0_b : i1
      %c0_align = arith.andi %has1, %c0_align0 : i1
      %p = scf.if %c0_align -> (i64) {
        memref.store %c0v, %out[%s1] : memref<?xi64>
        scf.yield %one : i64
      } else {
        scf.yield %zero : i64
      }
      scf.yield %p : i64
    }

    // helper closure equivalent: current char or 0
    %plus = arith.constant 43 : i64
    %minus_ch = arith.constant 45 : i64
    %space = arith.constant 32 : i64
    %in_sign = arith.cmpi slt, %pos_align, %n : i64
    %sc = scf.if %in_sign -> (i64) {
      %ix = arith.index_cast %pos_align : i64 to index
      %v = func.call @__ly_unicode_get(%spec_b, %wid, %ix) : (memref<?xi8>, i64, index) -> i64
      scf.yield %v : i64
    } else {
      scf.yield %zero : i64
    }
    %is_plus = arith.cmpi eq, %sc, %plus : i64
    %is_minus = arith.cmpi eq, %sc, %minus_ch : i64
    %is_space = arith.cmpi eq, %sc, %space : i64
    %is_sign0 = arith.ori %is_plus, %is_minus : i1
    %is_sign = arith.ori %is_sign0, %is_space : i1
    %pos_sign = scf.if %is_sign -> (i64) {
      memref.store %sc, %out[%s2] : memref<?xi64>
      %np = arith.addi %pos_align, %one : i64
      scf.yield %np : i64
    } else {
      scf.yield %pos_align : i64
    }

    // 'z'
    %zch = arith.constant 122 : i64
    %in_z = arith.cmpi slt, %pos_sign, %n : i64
    %zc = scf.if %in_z -> (i64) {
      %ix = arith.index_cast %pos_sign : i64 to index
      %v = func.call @__ly_unicode_get(%spec_b, %wid, %ix) : (memref<?xi8>, i64, index) -> i64
      scf.yield %v : i64
    } else {
      scf.yield %zero : i64
    }
    %is_z = arith.cmpi eq, %zc, %zch : i64
    %pos_z = scf.if %is_z -> (i64) {
      memref.store %one, %out[%s9] : memref<?xi64>
      %np = arith.addi %pos_sign, %one : i64
      scf.yield %np : i64
    } else {
      scf.yield %pos_sign : i64
    }

    // '#'
    %hash = arith.constant 35 : i64
    %in_h = arith.cmpi slt, %pos_z, %n : i64
    %hc = scf.if %in_h -> (i64) {
      %ix = arith.index_cast %pos_z : i64 to index
      %v = func.call @__ly_unicode_get(%spec_b, %wid, %ix) : (memref<?xi8>, i64, index) -> i64
      scf.yield %v : i64
    } else {
      scf.yield %zero : i64
    }
    %is_h = arith.cmpi eq, %hc, %hash : i64
    %pos_h = scf.if %is_h -> (i64) {
      memref.store %one, %out[%s3] : memref<?xi64>
      %np = arith.addi %pos_z, %one : i64
      scf.yield %np : i64
    } else {
      scf.yield %pos_z : i64
    }

    // '0' flag
    %zero_ch = arith.constant 48 : i64
    %in_0 = arith.cmpi slt, %pos_h, %n : i64
    %zc0 = scf.if %in_0 -> (i64) {
      %ix = arith.index_cast %pos_h : i64 to index
      %v = func.call @__ly_unicode_get(%spec_b, %wid, %ix) : (memref<?xi8>, i64, index) -> i64
      scf.yield %v : i64
    } else {
      scf.yield %zero : i64
    }
    %is_0 = arith.cmpi eq, %zc0, %zero_ch : i64
    %pos_0 = scf.if %is_0 -> (i64) {
      memref.store %one, %out[%s4] : memref<?xi64>
      %np = arith.addi %pos_h, %one : i64
      scf.yield %np : i64
    } else {
      scf.yield %pos_h : i64
    }

    // width digits
    %nine_ch = arith.constant 57 : i64
    %w:3 = scf.while (%p = %pos_0, %acc = %zero, %any = %zero) : (i64, i64, i64) -> (i64, i64, i64) {
      %in = arith.cmpi slt, %p, %n : i64
      %ch = scf.if %in -> (i64) {
        %ix = arith.index_cast %p : i64 to index
        %v = func.call @__ly_unicode_get(%spec_b, %wid, %ix) : (memref<?xi8>, i64, index) -> i64
        scf.yield %v : i64
      } else {
        scf.yield %zero : i64
      }
      %ge0 = arith.cmpi sge, %ch, %zero_ch : i64
      %le9 = arith.cmpi sle, %ch, %nine_ch : i64
      %digit0 = arith.andi %ge0, %le9 : i1
      %isdigit = arith.andi %in, %digit0 : i1
      scf.condition(%isdigit) %p, %acc, %any : i64, i64, i64
    } do {
    ^bb0(%p: i64, %acc: i64, %any: i64):
      %ix = arith.index_cast %p : i64 to index
      %ch = func.call @__ly_unicode_get(%spec_b, %wid, %ix) : (memref<?xi8>, i64, index) -> i64
      %d = arith.subi %ch, %zero_ch : i64
      // CPython's get_integer: `acc * 10 + d` passes PY_SSIZE_T_MAX iff
      // `acc > (PY_SSIZE_T_MAX - d) / 10`. ⛔ Not clamped, as it was (at
      // 10**15): a width past the word is CPython's ValueError, and one under
      // it is a size like any other (MemoryError where it cannot be met).
      %room = arith.subi %ssize_max, %d : i64
      %most = arith.divsi %room, %ten : i64
      %over = arith.cmpi sgt, %acc, %most : i64
      scf.if %over {
        func.call @__ly_fmt_raise_too_many_digits() : () -> ()
      }
      %acc10 = arith.muli %acc, %ten : i64
      %nacc = arith.addi %acc10, %d : i64
      %np = arith.addi %p, %one : i64
      scf.yield %np, %nacc, %one : i64, i64, i64
    }
    %had_width = arith.cmpi ne, %w#2, %zero : i64
    scf.if %had_width {
      memref.store %w#1, %out[%s5] : memref<?xi64>
    }

    // grouping
    %comma = arith.constant 44 : i64
    %under = arith.constant 95 : i64
    %in_g = arith.cmpi slt, %w#0, %n : i64
    %gc = scf.if %in_g -> (i64) {
      %ix = arith.index_cast %w#0 : i64 to index
      %v = func.call @__ly_unicode_get(%spec_b, %wid, %ix) : (memref<?xi8>, i64, index) -> i64
      scf.yield %v : i64
    } else {
      scf.yield %zero : i64
    }
    %g_comma = arith.cmpi eq, %gc, %comma : i64
    %g_under = arith.cmpi eq, %gc, %under : i64
    %is_g = arith.ori %g_comma, %g_under : i1
    %pos_g = scf.if %is_g -> (i64) {
      memref.store %gc, %out[%s6] : memref<?xi64>
      %np = arith.addi %w#0, %one : i64
      // a second grouping char right after is the dedicated error
      %in2 = arith.cmpi slt, %np, %n : i64
      %g2 = scf.if %in2 -> (i64) {
        %ix2 = arith.index_cast %np : i64 to index
        %v2 = func.call @__ly_unicode_get(%spec_b, %wid, %ix2) : (memref<?xi8>, i64, index) -> i64
        scf.yield %v2 : i64
      } else {
        scf.yield %zero : i64
      }
      %g2_comma = arith.cmpi eq, %g2, %comma : i64
      %g2_under = arith.cmpi eq, %g2, %under : i64
      %g2_is = arith.ori %g2_comma, %g2_under : i1
      scf.if %g2_is {
        func.call @__ly_fmt_raise_both_groupings() : () -> ()
      }
      scf.yield %np : i64
    } else {
      scf.yield %w#0 : i64
    }

    // '.' precision
    %dot = arith.constant 46 : i64
    %in_p = arith.cmpi slt, %pos_g, %n : i64
    %pc = scf.if %in_p -> (i64) {
      %ix = arith.index_cast %pos_g : i64 to index
      %v = func.call @__ly_unicode_get(%spec_b, %wid, %ix) : (memref<?xi8>, i64, index) -> i64
      scf.yield %v : i64
    } else {
      scf.yield %zero : i64
    }
    %is_dot = arith.cmpi eq, %pc, %dot : i64
    %pos_p = scf.if %is_dot -> (i64) {
      %pstart = arith.addi %pos_g, %one : i64
      %pw:3 = scf.while (%p = %pstart, %acc = %zero, %any = %zero) : (i64, i64, i64) -> (i64, i64, i64) {
        %in = arith.cmpi slt, %p, %n : i64
        %ch = scf.if %in -> (i64) {
          %ix = arith.index_cast %p : i64 to index
          %v = func.call @__ly_unicode_get(%spec_b, %wid, %ix) : (memref<?xi8>, i64, index) -> i64
          scf.yield %v : i64
        } else {
          scf.yield %zero : i64
        }
        %ge0 = arith.cmpi sge, %ch, %zero_ch : i64
        %le9 = arith.cmpi sle, %ch, %nine_ch : i64
        %digit0 = arith.andi %ge0, %le9 : i1
        %isdigit = arith.andi %in, %digit0 : i1
        scf.condition(%isdigit) %p, %acc, %any : i64, i64, i64
      } do {
      ^bb0(%p: i64, %acc: i64, %any: i64):
        %ix = arith.index_cast %p : i64 to index
        %ch = func.call @__ly_unicode_get(%spec_b, %wid, %ix) : (memref<?xi8>, i64, index) -> i64
        %d = arith.subi %ch, %zero_ch : i64
        %room = arith.subi %ssize_max, %d : i64
        %most = arith.divsi %room, %ten : i64
        %over = arith.cmpi sgt, %acc, %most : i64
        scf.if %over {
          func.call @__ly_fmt_raise_too_many_digits() : () -> ()
        }
        %acc10 = arith.muli %acc, %ten : i64
        %nacc = arith.addi %acc10, %d : i64
        %np = arith.addi %p, %one : i64
        scf.yield %np, %nacc, %one : i64, i64, i64
      }
      %had_prec = arith.cmpi ne, %pw#2, %zero : i64
      %none = arith.cmpi eq, %pw#2, %zero : i64
      scf.if %none {
        func.call @__ly_fmt_raise_missing_precision() : () -> ()
      }
      scf.if %had_prec {
        memref.store %pw#1, %out[%s7] : memref<?xi64>
      }
      scf.yield %pw#0 : i64
    } else {
      scf.yield %pos_g : i64
    }

    // type char
    %in_t = arith.cmpi slt, %pos_p, %n : i64
    %pos_t = scf.if %in_t -> (i64) {
      %ix = arith.index_cast %pos_p : i64 to index
      %v = func.call @__ly_unicode_get(%spec_b, %wid, %ix) : (memref<?xi8>, i64, index) -> i64
      memref.store %v, %out[%s8] : memref<?xi64>
      %np = arith.addi %pos_p, %one : i64
      scf.yield %np : i64
    } else {
      scf.yield %pos_p : i64
    }
    %ok = arith.cmpi eq, %pos_t, %n : i64
    func.return %ok : i1
  }

  // Materialize a code-point buffer as a canonical str.
  func.func private @__ly_fmt_str_from_cps(%cps: memref<?xi32>, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %n = arith.index_cast %len : i64 to index
    %maxcp = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %zero) -> (i64) {
      %v32 = memref.load %cps[%i] : memref<?xi32>
      %v = arith.extui %v32 : i32 to i64
      %bigger = arith.cmpi ugt, %v, %acc : i64
      %next = arith.select %bigger, %v, %acc : i64
      scf.yield %next : i64
    }
    %width = func.call @__ly_unicode_width_for(%maxcp) : (i64) -> i64
    %header, %bytes = func.call @__ly_unicode_alloc(%len, %width) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)
    scf.for %i = %c0 to %n step %c1 {
      %v32 = memref.load %cps[%i] : memref<?xi32>
      %v = arith.extui %v32 : i32 to i64
      func.call @__ly_unicode_put(%bytes, %width, %i, %v) : (memref<?xi8>, i64, index, i64) -> ()
    }
    func.return %header, %bytes : memref<2xi64>, memref<?xi8>
  }

  // Assemble a formatted number: sign, prefix, grouped integer digits, the
  // untouched rest (fraction/exponent/'%'), then width padding. '0'-fill
  // with '=' alignment refills *through* the grouping (CPython's regroup
  // rule), any other fill pads flat.
  func.func private @__ly_fmt_render_number(%sign_cp: i64, %pre0: i64, %pre1: i64, %body: memref<?xi32>, %body_len: i64, %int_len: i64, %group_cp: i64, %group_size: i64, %fill_cp: i64, %align_cp: i64, %width_in: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %width = arith.maxsi %width_in, %zero : i64
    // The result is at least `width` code units: one past the allocator's
    // reach is MemoryError, here as where the callers check it before they
    // allocate. ⛔ Not after the padding is worked out: the `=` fill with
    // grouping counts up to the width a digit at a time.
    %code_unit = arith.constant 4 : i64
    func.call @__ly_check_alloc_count(%width, %code_unit, %zero) : (i64, i64, i64) -> ()
    %has_sign = arith.cmpi ne, %sign_cp, %zero : i64
    %sign_len = arith.select %has_sign, %one, %zero : i64
    %has_p0 = arith.cmpi ne, %pre0, %zero : i64
    %p0_len = arith.select %has_p0, %one, %zero : i64
    %has_p1 = arith.cmpi ne, %pre1, %zero : i64
    %p1_len = arith.select %has_p1, %one, %zero : i64
    %pre_len = arith.addi %p0_len, %p1_len : i64
    %rest_len = arith.subi %body_len, %int_len : i64
    %fixed0 = arith.addi %sign_len, %pre_len : i64
    %fixed = arith.addi %fixed0, %rest_len : i64
    %grouped = arith.cmpi ne, %group_cp, %zero : i64

    %eq_align = arith.constant 61 : i64
    %zero_fill = arith.constant 48 : i64
    %is_eq = arith.cmpi eq, %align_cp, %eq_align : i64
    %is_zero_fill = arith.cmpi eq, %fill_cp, %zero_fill : i64
    %regroup0 = arith.andi %is_eq, %is_zero_fill : i1
    %regroup1 = arith.andi %regroup0, %grouped : i1
    %has_width = arith.cmpi sgt, %width, %zero : i64
    %regroup = arith.andi %regroup1, %has_width : i1

    // m = padded integer digit count
    %m = scf.if %regroup -> (i64) {
      %mres = scf.while (%mi = %int_len) : (i64) -> i64 {
        %mm1 = arith.subi %mi, %one : i64
        %mm1c = arith.maxsi %mm1, %zero : i64
        %seps = arith.divui %mm1c, %group_size : i64
        %glen = arith.addi %mi, %seps : i64
        %tot = arith.addi %glen, %fixed : i64
        %need_more = arith.cmpi slt, %tot, %width : i64
        scf.condition(%need_more) %mi : i64
      } do {
      ^bb0(%mi: i64):
        %nmi = arith.addi %mi, %one : i64
        scf.yield %nmi : i64
      }
      scf.yield %mres : i64
    } else {
      scf.yield %int_len : i64
    }

    %mm1 = arith.subi %m, %one : i64
    %mm1c = arith.maxsi %mm1, %zero : i64
    %seps0 = arith.divui %mm1c, %group_size : i64
    %seps = arith.select %grouped, %seps0, %zero : i64
    %m_pos = arith.cmpi sgt, %m, %zero : i64
    %seps_eff = arith.select %m_pos, %seps, %zero : i64
    %gint = arith.addi %m, %seps_eff : i64
    %core = arith.addi %fixed, %gint : i64
    %pad_raw = arith.subi %width, %core : i64
    %pad = arith.maxsi %pad_raw, %zero : i64
    %total = arith.addi %core, %pad : i64

    %lt_align = arith.constant 60 : i64
    %caret_align = arith.constant 94 : i64
    %is_lt = arith.cmpi eq, %align_cp, %lt_align : i64
    %is_caret = arith.cmpi eq, %align_cp, %caret_align : i64
    %half = arith.divui %pad, %two : i64
    %left_pad = scf.if %is_lt -> (i64) {
      scf.yield %zero : i64
    } else {
      %lp = scf.if %is_caret -> (i64) {
        scf.yield %half : i64
      } else {
        %lp2 = scf.if %is_eq -> (i64) {
          scf.yield %zero : i64
        } else {
          scf.yield %pad : i64
        }
        scf.yield %lp2 : i64
      }
      scf.yield %lp : i64
    }
    %mid_pad = scf.if %is_eq -> (i64) {
      scf.yield %pad : i64
    } else {
      scf.yield %zero : i64
    }
    %used = arith.addi %left_pad, %mid_pad : i64
    %right_pad = arith.subi %pad, %used : i64

    %total_idx = arith.index_cast %total : i64 to index
    %buf = memref.alloc(%total_idx) : memref<?xi32>
    %fill32 = arith.trunci %fill_cp : i64 to i32
    %lp_idx = arith.index_cast %left_pad : i64 to index
    scf.for %i = %c0 to %lp_idx step %c1 {
      memref.store %fill32, %buf[%i] : memref<?xi32>
    }
    // sign
    %pos1 = scf.if %has_sign -> (index) {
      %s32 = arith.trunci %sign_cp : i64 to i32
      memref.store %s32, %buf[%lp_idx] : memref<?xi32>
      %np = arith.addi %lp_idx, %c1 : index
      scf.yield %np : index
    } else {
      scf.yield %lp_idx : index
    }
    %pos2 = scf.if %has_p0 -> (index) {
      %v32 = arith.trunci %pre0 : i64 to i32
      memref.store %v32, %buf[%pos1] : memref<?xi32>
      %np = arith.addi %pos1, %c1 : index
      scf.yield %np : index
    } else {
      scf.yield %pos1 : index
    }
    %pos3 = scf.if %has_p1 -> (index) {
      %v32 = arith.trunci %pre1 : i64 to i32
      memref.store %v32, %buf[%pos2] : memref<?xi32>
      %np = arith.addi %pos2, %c1 : index
      scf.yield %np : index
    } else {
      scf.yield %pos2 : index
    }
    %mp_idx = arith.index_cast %mid_pad : i64 to index
    %pos4 = scf.for %i = %c0 to %mp_idx step %c1 iter_args(%p = %pos3) -> (index) {
      memref.store %fill32, %buf[%p] : memref<?xi32>
      %np = arith.addi %p, %c1 : index
      scf.yield %np : index
    }
    // grouped integer digits: m digits, zeros pad the front
    %pad_zeros = arith.subi %m, %int_len : i64
    %m_idx = arith.index_cast %m : i64 to index
    %g32 = arith.trunci %group_cp : i64 to i32
    %zero32 = arith.constant 48 : i32
    %pos5 = scf.for %i = %c0 to %m_idx step %c1 iter_args(%p = %pos4) -> (index) {
      %i_i64 = arith.index_cast %i : index to i64
      %not_first = arith.cmpi sgt, %i_i64, %zero : i64
      %remaining = arith.subi %m, %i_i64 : i64
      %rem_mod = arith.remui %remaining, %group_size : i64
      %at_sep0 = arith.cmpi eq, %rem_mod, %zero : i64
      %at_sep1 = arith.andi %not_first, %at_sep0 : i1
      %at_sep = arith.andi %at_sep1, %grouped : i1
      %p_sep = scf.if %at_sep -> (index) {
        memref.store %g32, %buf[%p] : memref<?xi32>
        %np = arith.addi %p, %c1 : index
        scf.yield %np : index
      } else {
        scf.yield %p : index
      }
      %is_pad = arith.cmpi slt, %i_i64, %pad_zeros : i64
      %digit = scf.if %is_pad -> (i32) {
        scf.yield %zero32 : i32
      } else {
        %src_i64 = arith.subi %i_i64, %pad_zeros : i64
        %src = arith.index_cast %src_i64 : i64 to index
        %v = memref.load %body[%src] : memref<?xi32>
        scf.yield %v : i32
      }
      memref.store %digit, %buf[%p_sep] : memref<?xi32>
      %np2 = arith.addi %p_sep, %c1 : index
      scf.yield %np2 : index
    }
    // rest
    %int_idx = arith.index_cast %int_len : i64 to index
    %body_idx = arith.index_cast %body_len : i64 to index
    %pos6 = scf.for %i = %int_idx to %body_idx step %c1 iter_args(%p = %pos5) -> (index) {
      %v = memref.load %body[%i] : memref<?xi32>
      memref.store %v, %buf[%p] : memref<?xi32>
      %np = arith.addi %p, %c1 : index
      scf.yield %np : index
    }
    %rp_idx = arith.index_cast %right_pad : i64 to index
    %pos7 = scf.for %i = %c0 to %rp_idx step %c1 iter_args(%p = %pos6) -> (index) {
      memref.store %fill32, %buf[%p] : memref<?xi32>
      %np = arith.addi %p, %c1 : index
      scf.yield %np : index
    }
    %header, %bytes = func.call @__ly_fmt_str_from_cps(%buf, %total) : (memref<?xi32>, i64) -> (memref<2xi64>, memref<?xi8>)
    memref.dealloc %buf : memref<?xi32>
    func.return %header, %bytes : memref<2xi64>, memref<?xi8>
  }

  // Fixed-notation body: integer digits, optional '.', fraction digits.
  // frac_len -1 means the natural fraction (digits past the point, at least
  // min_frac); a non-negative frac_len pads/truncates to exactly that many.
  // Returns (body length, integer digit count).
  func.func private @__ly_fmt_body_fixed(%out: memref<?xi32>, %digits: memref<?xi8>, %count: i64, %decpt: i64, %frac_len: i64, %min_frac: i64, %force_dot: i1) -> (i64, i64) {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero_ch = arith.constant 48 : i32
    %dot_ch = arith.constant 46 : i32
    %int_len = arith.maxsi %decpt, %one : i64
    %natural0 = arith.subi %count, %decpt : i64
    %natural1 = arith.maxsi %natural0, %zero : i64
    %natural = arith.maxsi %natural1, %min_frac : i64
    %frac_given = arith.cmpi sge, %frac_len, %zero : i64
    %frac = arith.select %frac_given, %frac_len, %natural : i64
    %int_idx = arith.index_cast %int_len : i64 to index
    %count_idx = arith.index_cast %count : i64 to index
    %dec_pos = arith.cmpi sgt, %decpt, %zero : i64
    // integer digits
    %p1 = scf.if %dec_pos -> (index) {
      %p = scf.for %i = %c0 to %int_idx step %c1 iter_args(%q = %c0) -> (index) {
        %in = arith.cmpi ult, %i, %count_idx : index
        %ch = scf.if %in -> (i32) {
          %b = memref.load %digits[%i] : memref<?xi8>
          %w = arith.extui %b : i8 to i32
          scf.yield %w : i32
        } else {
          scf.yield %zero_ch : i32
        }
        memref.store %ch, %out[%q] : memref<?xi32>
        %nq = arith.addi %q, %c1 : index
        scf.yield %nq : index
      }
      scf.yield %p : index
    } else {
      memref.store %zero_ch, %out[%c0] : memref<?xi32>
      scf.yield %c1 : index
    }
    %int_written_i64 = arith.index_cast %p1 : index to i64
    %frac_pos = arith.cmpi sgt, %frac, %zero : i64
    %need_dot = arith.ori %frac_pos, %force_dot : i1
    %p2 = scf.if %need_dot -> (index) {
      memref.store %dot_ch, %out[%p1] : memref<?xi32>
      %np = arith.addi %p1, %c1 : index
      scf.yield %np : index
    } else {
      scf.yield %p1 : index
    }
    %frac_idx = arith.index_cast %frac : i64 to index
    %p3 = scf.for %i = %c0 to %frac_idx step %c1 iter_args(%q = %p2) -> (index) {
      %j = arith.index_cast %i : index to i64
      %src = arith.addi %decpt, %j : i64
      %ge0 = arith.cmpi sge, %src, %zero : i64
      %ltc = arith.cmpi slt, %src, %count : i64
      %in = arith.andi %ge0, %ltc : i1
      %ch = scf.if %in -> (i32) {
        %si = arith.index_cast %src : i64 to index
        %b = memref.load %digits[%si] : memref<?xi8>
        %w = arith.extui %b : i8 to i32
        scf.yield %w : i32
      } else {
        scf.yield %zero_ch : i32
      }
      memref.store %ch, %out[%q] : memref<?xi32>
      %nq = arith.addi %q, %c1 : index
      scf.yield %nq : index
    }
    %body_len = arith.index_cast %p3 : index to i64
    func.return %body_len, %int_written_i64 : i64, i64
  }

  // Exponent-notation body: d0[.frac]e(sign)XX. mant_frac -1 keeps the natural
  // count-1 mantissa fraction. Returns (body length, integer digit count=1).
  func.func private @__ly_fmt_body_exp(%out: memref<?xi32>, %digits: memref<?xi8>, %count: i64, %decpt: i64, %mant_frac: i64, %force_dot: i1, %e_cp: i64) -> (i64, i64) {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %ten = arith.constant 10 : i64
    %hundred = arith.constant 100 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero_ch = arith.constant 48 : i32
    %dot_ch = arith.constant 46 : i32
    %plus_ch = arith.constant 43 : i32
    %minus_ch = arith.constant 45 : i32
    %count_idx = arith.index_cast %count : i64 to index
    %has_digit = arith.cmpi sgt, %count, %zero : i64
    %d0 = scf.if %has_digit -> (i32) {
      %b = memref.load %digits[%c0] : memref<?xi8>
      %w = arith.extui %b : i8 to i32
      scf.yield %w : i32
    } else {
      scf.yield %zero_ch : i32
    }
    memref.store %d0, %out[%c0] : memref<?xi32>
    %natural = arith.subi %count, %one : i64
    %natural_c = arith.maxsi %natural, %zero : i64
    %frac_given = arith.cmpi sge, %mant_frac, %zero : i64
    %frac = arith.select %frac_given, %mant_frac, %natural_c : i64
    %frac_pos = arith.cmpi sgt, %frac, %zero : i64
    %need_dot = arith.ori %frac_pos, %force_dot : i1
    %p1 = scf.if %need_dot -> (index) {
      memref.store %dot_ch, %out[%c1] : memref<?xi32>
      %np = arith.addi %c1, %c1 : index
      scf.yield %np : index
    } else {
      scf.yield %c1 : index
    }
    %frac_idx = arith.index_cast %frac : i64 to index
    %p2 = scf.for %i = %c0 to %frac_idx step %c1 iter_args(%q = %p1) -> (index) {
      %ip1 = arith.addi %i, %c1 : index
      %in = arith.cmpi ult, %ip1, %count_idx : index
      %ch = scf.if %in -> (i32) {
        %b = memref.load %digits[%ip1] : memref<?xi8>
        %w = arith.extui %b : i8 to i32
        scf.yield %w : i32
      } else {
        scf.yield %zero_ch : i32
      }
      memref.store %ch, %out[%q] : memref<?xi32>
      %nq = arith.addi %q, %c1 : index
      scf.yield %nq : index
    }
    %e32 = arith.trunci %e_cp : i64 to i32
    memref.store %e32, %out[%p2] : memref<?xi32>
    %p3 = arith.addi %p2, %c1 : index
    %exp = arith.subi %decpt, %one : i64
    %neg = arith.cmpi slt, %exp, %zero : i64
    %sign32 = arith.select %neg, %minus_ch, %plus_ch : i32
    memref.store %sign32, %out[%p3] : memref<?xi32>
    %p4 = arith.addi %p3, %c1 : index
    %negated = arith.subi %zero, %exp : i64
    %eabs = arith.select %neg, %negated, %exp : i64
    %ge100 = arith.cmpi sge, %eabs, %hundred : i64
    %p5 = scf.if %ge100 -> (index) {
      %h = arith.divui %eabs, %hundred : i64
      %hch = arith.addi %h, %zero : i64
      %h48 = arith.constant 48 : i64
      %hc = arith.addi %hch, %h48 : i64
      %hc32 = arith.trunci %hc : i64 to i32
      memref.store %hc32, %out[%p4] : memref<?xi32>
      %np = arith.addi %p4, %c1 : index
      scf.yield %np : index
    } else {
      scf.yield %p4 : index
    }
    %rem = arith.remui %eabs, %hundred : i64
    %tens = arith.divui %rem, %ten : i64
    %c48 = arith.constant 48 : i64
    %tch = arith.addi %tens, %c48 : i64
    %tch32 = arith.trunci %tch : i64 to i32
    memref.store %tch32, %out[%p5] : memref<?xi32>
    %p6 = arith.addi %p5, %c1 : index
    %units = arith.remui %rem, %ten : i64
    %uch = arith.addi %units, %c48 : i64
    %uch32 = arith.trunci %uch : i64 to i32
    memref.store %uch32, %out[%p6] : memref<?xi32>
    %p7 = arith.addi %p6, %c1 : index
    %body_len = arith.index_cast %p7 : index to i64
    func.return %body_len, %one : i64, i64
  }

  // Base-2^shift digits (bin/oct/hex) straight off the 2^30 limbs, most
  // significant first. Returns the digit count.
  func.func private @__ly_fmt_int_base2k(%meta: memref<2xi64>, %digits: memref<?xi32>, %shift: i64, %upper: i1, %out: memref<?xi32>) -> i64 {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %ten = arith.constant 10 : i64
    %thirty = arith.constant 30 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %count_slot = arith.constant 1 : index
    %count = memref.load %meta[%count_slot] : memref<2xi64>
    %bits = func.call @__ly_long_bit_length(%meta, %digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %bits_m1 = arith.addi %bits, %shift : i64
    %bits_adj = arith.subi %bits_m1, %one : i64
    %nd0 = arith.divui %bits_adj, %shift : i64
    %nd = arith.maxsi %nd0, %one : i64
    %nd_idx = arith.index_cast %nd : i64 to index
    %shift_idx = arith.index_cast %shift : i64 to index
    scf.for %di = %c0 to %nd_idx step %c1 {
      %di_i64 = arith.index_cast %di : index to i64
      %rev = arith.subi %nd, %di_i64 : i64
      %rev_m1 = arith.subi %rev, %one : i64
      %base_bit = arith.muli %rev_m1, %shift : i64
      %v = scf.for %b = %c0 to %shift_idx step %c1 iter_args(%acc = %zero) -> (i64) {
        %b_i64 = arith.index_cast %b : index to i64
        %j = arith.addi %base_bit, %b_i64 : i64
        %limb_i = arith.divui %j, %thirty : i64
        %bit_i = arith.remui %j, %thirty : i64
        %in = arith.cmpi slt, %limb_i, %count : i64
        %bit = scf.if %in -> (i64) {
          %li = arith.index_cast %limb_i : i64 to index
          %limb32 = memref.load %digits[%li] : memref<?xi32>
          %limb = arith.extui %limb32 : i32 to i64
          %shifted = arith.shrui %limb, %bit_i : i64
          %m = arith.andi %shifted, %one : i64
          scf.yield %m : i64
        } else {
          scf.yield %zero : i64
        }
        %sh = arith.shli %bit, %b_i64 : i64
        %nacc = arith.ori %acc, %sh : i64
        scf.yield %nacc : i64
      }
      %lt10 = arith.cmpi slt, %v, %ten : i64
      %d48 = arith.constant 48 : i64
      %num_ch = arith.addi %v, %d48 : i64
      %a_lower = arith.constant 87 : i64
      %a_upper = arith.constant 55 : i64
      %a_base = arith.select %upper, %a_upper, %a_lower : i64
      %alpha_ch = arith.addi %v, %a_base : i64
      %ch = arith.select %lt10, %num_ch, %alpha_ch : i64
      %ch32 = arith.trunci %ch : i64 to i32
      memref.store %ch32, %out[%di] : memref<?xi32>
    }
    func.return %nd : i64
  }
}
