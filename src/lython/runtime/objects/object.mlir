module attributes {ly.runtime.contracts = ["types.NoneType", "builtins.object", "builtins.bool", "builtins.BaseException", "builtins.int", "builtins.str"]} {
  memref.global "private" constant @__ly_bool_repr_true : memref<4xi8> = dense<[84, 114, 117, 101]>
  memref.global "private" constant @__ly_bool_repr_false : memref<5xi8> = dense<[70, 97, 108, 115, 101]>

  // ABI shape declarations are manifest entries: they describe the physical
  // runtime bundle for contracts whose structural signatures live elsewhere.
  func.func private @LyNone_Shape() attributes {ly.runtime.contract = "types.NoneType", ly.runtime.shape}

  func.func private @LyObject_Shape() -> memref<16xi64> attributes {ly.runtime.contract = "builtins.object", ly.runtime.shape}

  func.func private @LyBool_Shape() -> i1 attributes {ly.runtime.contract = "builtins.bool", ly.runtime.shape}

  func.func private @LyUnicode_FromBytes(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0]}

  func.func private @LyBuiltin_Len() -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.runtime.builtin = "len", ly.runtime.builtin_lowering = "method", ly.runtime.builtin_method = "__len__", ly.runtime.contract = "builtins.object", ly.runtime.primitive = "builtin_len", ly.runtime.result_contract = "builtins.int"}

  func.func @LyObject_Init(%header: memref<16xi64> {ly.ownership.object_header}) attributes {ly.runtime.class_id = 0 : i64, ly.runtime.contract = "builtins.object", ly.runtime.method = "__init__"} {
    func.return
  }

  func.func private @LyObject_ReleaseBoxedPayloadRaw(%box: memref<16xi64>)

  func.func private @LyObject_PrintLine(%box: memref<16xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "builtins.object", ly.runtime.primitive = "print_line"}

  func.func @LyObject_DecRef(%box: memref<16xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.object", ly.runtime.deallocator} {
    %storage = memref.cast %box : memref<16xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    func.call @LyObject_ReleaseBoxedPayloadRaw(%box) : (memref<16xi64>) -> ()
    memref.dealloc %box {ly.ownership.object_dealloc_part = "boxed_object"} : memref<16xi64>
    cf.br ^done

  ^done:
    func.return
  }

  func.func @LyObject_DefaultRepr(%header: memref<2xi64, strided<[1], offset: ?>> {ly.ownership.object_header}, %prefix: memref<?xi8>, %prefix_len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.object", ly.runtime.primitive = "default_repr", ly.runtime.result_contract = "builtins.str"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %sixteen = arith.constant 16 : i64
    %max_digits_index = arith.constant 16 : index
    %lower = arith.constant 0 : index
    %step = arith.constant 1 : index
    %ascii_zero = arith.constant 48 : i64
    %ascii_a_minus_ten = arith.constant 87 : i64
    %ascii_gt = arith.constant 62 : i8

    %ptr_index = memref.extract_aligned_pointer_as_index %header : memref<2xi64, strided<[1], offset: ?>> -> index
    %ptr = arith.index_cast %ptr_index : index to i64
    %counted:2 = scf.for %i = %lower to %max_digits_index step %step iter_args(%n = %ptr, %digits = %zero) -> (i64, i64) {
      %active = arith.cmpi ne, %n, %zero : i64
      %next_n_active = arith.divui %n, %sixteen : i64
      %next_n = arith.select %active, %next_n_active, %n : i1, i64
      %incremented = arith.addi %digits, %one : i64
      %next_digits = arith.select %active, %incremented, %digits : i1, i64
      scf.yield %next_n, %next_digits : i64, i64
    }
    %ptr_is_zero = arith.cmpi eq, %ptr, %zero : i64
    %hex_digits = arith.select %ptr_is_zero, %one, %counted#1 : i1, i64
    %body_len = arith.addi %hex_digits, %one : i64
    %total_len = arith.addi %prefix_len, %body_len : i64
    %total_len_index = arith.index_cast %total_len : i64 to index
    %prefix_len_index = arith.index_cast %prefix_len : i64 to index
    %hex_digits_index = arith.index_cast %hex_digits : i64 to index
    %buffer = memref.alloca(%total_len_index) : memref<?xi8>

    scf.for %i = %lower to %prefix_len_index step %step {
      %byte = memref.load %prefix[%i] : memref<?xi8>
      memref.store %byte, %buffer[%i] : memref<?xi8>
    }

    scf.for %i = %lower to %hex_digits_index step %step iter_args(%n = %ptr) -> (i64) {
      %digit = arith.remui %n, %sixteen : i64
      %ten = arith.constant 10 : i64
      %is_decimal = arith.cmpi ult, %digit, %ten : i64
      %decimal_ch = arith.addi %digit, %ascii_zero : i64
      %alpha_ch = arith.addi %digit, %ascii_a_minus_ten : i64
      %ch_i64 = arith.select %is_decimal, %decimal_ch, %alpha_ch : i1, i64
      %ch = arith.trunci %ch_i64 : i64 to i8
      %one_index = arith.constant 1 : index
      %last_digit = arith.subi %hex_digits_index, %one_index : index
      %offset = arith.subi %last_digit, %i : index
      %dest = arith.addi %prefix_len_index, %offset : index
      memref.store %ch, %buffer[%dest] : memref<?xi8>
      %next = arith.divui %n, %sixteen : i64
      scf.yield %next : i64
    }

    %suffix_pos = arith.addi %prefix_len_index, %hex_digits_index : index
    memref.store %ascii_gt, %buffer[%suffix_pos] : memref<?xi8>
    %start = arith.constant 0 : index
    %result_header, %result_bytes = func.call @LyUnicode_FromBytes(%buffer, %start, %total_len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result_header, %result_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyBool_Repr(%value: i1) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bool", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %result:2 = scf.if %value -> (memref<2xi64>, memref<?xi8>) {
      %true_static = memref.get_global @__ly_bool_repr_true : memref<4xi8>
      %true_bytes = memref.cast %true_static : memref<4xi8> to memref<?xi8>
      %true_len = arith.constant 4 : i64
      %header, %bytes = func.call @LyUnicode_FromBytes(%true_bytes, %c0, %true_len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %header, %bytes : memref<2xi64>, memref<?xi8>
    } else {
      %false_static = memref.get_global @__ly_bool_repr_false : memref<5xi8>
      %false_bytes = memref.cast %false_static : memref<5xi8> to memref<?xi8>
      %false_len = arith.constant 5 : i64
      %header, %bytes = func.call @LyUnicode_FromBytes(%false_bytes, %c0, %false_len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %header, %bytes : memref<2xi64>, memref<?xi8>
    }
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyBool_Str(%value: i1) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bool", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %header, %bytes = func.call @LyBool_Repr(%value) : (i1) -> (memref<2xi64>, memref<?xi8>)
    func.return %header, %bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @Ly_IncRef(%header: memref<2xi64, strided<[1], offset: ?>> {ly.ownership.object_header}) attributes {ly.ownership.retain_args = [0], ly.runtime.primitive = "retain"} {
    %slot = arith.constant 0 : index
    %zero = arith.constant 0 : i64
    %immortal = arith.constant 9223372036854775807 : i64
    // Tagged fast path: a header whose aligned pointer has bit 0 set is an
    // inline tagged long (rfc/tagged-long.md). It owns no memory and must
    // not be dereferenced at all.
    %ptr_index = memref.extract_aligned_pointer_as_index %header : memref<2xi64, strided<[1], offset: ?>> -> index
    %ptr_bits = arith.index_cast %ptr_index : index to i64
    %tag_one = arith.constant 1 : i64
    %tag_bit = arith.andi %ptr_bits, %tag_one : i64
    %is_tagged = arith.cmpi eq, %tag_bit, %tag_one : i64
    cf.cond_br %is_tagged, ^done, ^probe

  ^probe:
    // Immortal fast path: immortality is fixed at object creation and never
    // changes, so a pre-RMW acquire read is a stable witness. Skipping the
    // RMW keeps immortal headers write-free, which lets constant literals
    // live in read-only sections.
    %observed = memref.load %header[%slot] {ly.atomic.ordering = "acquire", ly.atomic.role = "object.refcount.load"} : memref<2xi64, strided<[1], offset: ?>>
    %pre_immortal = arith.cmpi eq, %observed, %immortal : i64
    cf.cond_br %pre_immortal, ^done, ^mutate

  ^mutate:
    %previous = memref.generic_atomic_rmw %header[%slot] : memref<2xi64, strided<[1], offset: ?>> {
    ^bb0(%current : i64):
      %body_zero = arith.constant 0 : i64
      %body_one = arith.constant 1 : i64
      %body_immortal = arith.constant 9223372036854775807 : i64
      %body_positive = arith.cmpi sgt, %current, %body_zero : i64
      %body_immortal_check = arith.cmpi eq, %current, %body_immortal : i64
      %body_incremented = arith.addi %current, %body_one : i64
      %body_positive_next = arith.select %body_positive, %body_incremented, %current : i1, i64
      %body_next = arith.select %body_immortal_check, %current, %body_positive_next : i1, i64
      memref.atomic_yield %body_next : i64
    } {ly.atomic.ordering = "acq_rel", ly.atomic.retain_premise = "entry-borrowed", ly.atomic.role = "object.refcount.retain"}
    %is_immortal = arith.cmpi eq, %previous, %immortal : i64
    cf.cond_br %is_immortal, ^done, ^check_positive

  ^check_positive:
    %positive = arith.cmpi sgt, %previous, %zero : i64
    cf.assert %positive, "Ly_IncRef observed non-positive refcount"
    cf.br ^done

  ^done:
    func.return
  }

  func.func @LyObject_ReleaseStorageToZero(%storage: memref<?xi64>) -> i1 attributes {ly.runtime.contract = "builtins.object", ly.runtime.primitive = "release_to_zero"} {
    %slot = arith.constant 0 : index
    %zero = arith.constant 0 : i64
    %immortal = arith.constant 9223372036854775807 : i64
    // Immortal fast path: see Ly_IncRef. Immortal storage is never written,
    // so constant literal objects stay in read-only storage.
    %observed = memref.load %storage[%slot] {ly.atomic.ordering = "acquire", ly.atomic.role = "object.refcount.load"} : memref<?xi64>
    %pre_immortal = arith.cmpi eq, %observed, %immortal : i64
    cf.cond_br %pre_immortal, ^done, ^mutate

  ^mutate:
    %previous = memref.generic_atomic_rmw %storage[%slot] : memref<?xi64> {
    ^bb0(%current : i64):
      %body_zero = arith.constant 0 : i64
      %body_one = arith.constant 1 : i64
      %body_immortal = arith.constant 9223372036854775807 : i64
      %body_positive = arith.cmpi sgt, %current, %body_zero : i64
      %body_immortal_check = arith.cmpi eq, %current, %body_immortal : i64
      %body_decremented = arith.subi %current, %body_one : i64
      %body_positive_next = arith.select %body_positive, %body_decremented, %current : i1, i64
      %body_next = arith.select %body_immortal_check, %current, %body_positive_next : i1, i64
      memref.atomic_yield %body_next : i64
    } {ly.atomic.ordering = "acq_rel", ly.atomic.role = "object.refcount.release"}
    %is_immortal = arith.cmpi eq, %previous, %immortal : i64
    cf.cond_br %is_immortal, ^done, ^check_positive

  ^check_positive:
    %positive = arith.cmpi sgt, %previous, %zero : i64
    cf.assert %positive, "Ly_DecRef observed non-positive refcount"
    %one = arith.constant 1 : i64
    %became_zero = arith.cmpi eq, %previous, %one : i64
    func.return %became_zero : i1

  ^done:
    %false = arith.constant false
    func.return %false : i1
  }

  func.func @LyBaseException_DecRef(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.BaseException", ly.runtime.deallocator} {
    %storage = memref.cast %header : memref<3xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    func.call @LyUnicode_DecRef(%message_header, %message_bytes) : (memref<2xi64>, memref<?xi8>) -> ()
    memref.dealloc %header {ly.ownership.object_dealloc_part = "header"} : memref<3xi64>
    cf.br ^done

  ^done:
    func.return
  }

  func.func @LyLong_DecRef(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %digits: memref<?xi32>) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.int", ly.runtime.deallocator} {
    %storage = memref.cast %header : memref<2xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    memref.dealloc %digits {ly.ownership.object_dealloc_part = "digits"} : memref<?xi32>
    memref.dealloc %meta {ly.ownership.object_dealloc_part = "meta"} : memref<2xi64>
    memref.dealloc %header {ly.ownership.object_dealloc_part = "header"} : memref<2xi64>
    cf.br ^done

  ^done:
    func.return
  }

  func.func @LyUnicode_DecRef(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.str", ly.runtime.deallocator} {
    %storage = memref.cast %header : memref<2xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    memref.dealloc %bytes {ly.ownership.object_dealloc_part = "bytes"} : memref<?xi8>
    memref.dealloc %header {ly.ownership.object_dealloc_part = "header"} : memref<2xi64>
    cf.br ^done

  ^done:
    func.return
  }

}
