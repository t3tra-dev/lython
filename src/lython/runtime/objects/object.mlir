module {
  func.func @Ly_IncRef(%header: memref<2xi64, strided<[1], offset: ?>> {ly.ownership.object_header}) attributes {ly.ownership.retain_args = [0]} {
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

  func.func private @__ly_decref_release(%header: memref<2xi64, strided<[1], offset: ?>> {ly.ownership.object_header}) -> i1 attributes {ly.ownership.object_release_to_zero} {
    %slot = arith.constant 0 : index
    %zero = arith.constant 0 : i64
    %immortal = arith.constant 9223372036854775807 : i64
    // Tagged fast path: see Ly_IncRef and rfc/tagged-long.md.
    %ptr_index = memref.extract_aligned_pointer_as_index %header : memref<2xi64, strided<[1], offset: ?>> -> index
    %ptr_bits = arith.index_cast %ptr_index : index to i64
    %tag_one = arith.constant 1 : i64
    %tag_bit = arith.andi %ptr_bits, %tag_one : i64
    %is_tagged = arith.cmpi eq, %tag_bit, %tag_one : i64
    cf.cond_br %is_tagged, ^done, ^probe

  ^probe:
    // Immortal fast path: see Ly_IncRef. Immortal headers are never written,
    // so constant literal objects stay in read-only storage.
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

  func.func @LyException_DecRef(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.release_args = [0]} {
    %c0 = arith.constant 0 : index
    %sub_header = memref.subview %header[%c0] [2] [1] : memref<3xi64> to memref<2xi64, strided<[1], offset: ?>>
    %became_zero = func.call @__ly_decref_release(%sub_header) : (memref<2xi64, strided<[1], offset: ?>>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    func.call @LyUnicode_DecRef(%message_header, %message_bytes) : (memref<2xi64>, memref<?xi8>) -> ()
    memref.dealloc %header {ly.ownership.object_dealloc_part = "header"} : memref<3xi64>
    cf.br ^done

  ^done:
    func.return
  }

  func.func @LyLong_DecRef(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %digits: memref<?xi32>) attributes {ly.ownership.release_args = [0]} {
    %header_view = memref.cast %header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    %became_zero = func.call @__ly_decref_release(%header_view) : (memref<2xi64, strided<[1], offset: ?>>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    memref.dealloc %digits {ly.ownership.object_dealloc_part = "digits"} : memref<?xi32>
    memref.dealloc %meta {ly.ownership.object_dealloc_part = "meta"} : memref<2xi64>
    memref.dealloc %header {ly.ownership.object_dealloc_part = "header"} : memref<2xi64>
    cf.br ^done

  ^done:
    func.return
  }

  func.func @LyUnicode_DecRef(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) attributes {ly.ownership.release_args = [0]} {
    %header_view = memref.cast %header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    %became_zero = func.call @__ly_decref_release(%header_view) : (memref<2xi64, strided<[1], offset: ?>>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    memref.dealloc %bytes {ly.ownership.object_dealloc_part = "bytes"} : memref<?xi8>
    memref.dealloc %header {ly.ownership.object_dealloc_part = "header"} : memref<2xi64>
    cf.br ^done

  ^done:
    func.return
  }

}
