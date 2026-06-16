module {
  func.func private @LyHost_Print(memref<?xi8>, i64)
  func.func private @LyHost_PrintLine(memref<?xi8>, i64)

  func.func private @__ly_unicode_alloc(%len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0]} {
    %header = memref.alloc() {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<2xi64>
    %byte_count = arith.index_cast %len : i64 to index
    %bytes = memref.alloc(%byte_count) : memref<?xi8>
    %one = arith.constant 1 : i64
    %layout_str = arith.constant 4 : i64
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    memref.store %one, %header[%refcount_slot] : memref<2xi64>
    memref.store %layout_str, %header[%layout_slot] : memref<2xi64>
    func.return %header, %bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_FromBytes(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0]} {
    %header, %result_bytes = func.call @__ly_unicode_alloc(%len) : (i64) -> (memref<2xi64>, memref<?xi8>)
    %byte_count = arith.index_cast %len : i64 to index
    %lower = arith.constant 0 : index
    %step = arith.constant 1 : index
    scf.for %index = %lower to %byte_count step %step {
      %source_index = arith.addi %start, %index : index
      %byte = memref.load %bytes[%source_index] : memref<?xi8>
      memref.store %byte, %result_bytes[%index] : memref<?xi8>
    }
    func.return %header, %result_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_Length(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> i64 {
    %c0 = arith.constant 0 : index
    %length_index = memref.dim %bytes, %c0 : memref<?xi8>
    %length = arith.index_cast %length_index : index to i64
    func.return %length : i64
  }

  func.func @LyUnicode_CodepointLength(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> i64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %byte_count = memref.dim %bytes, %c0 : memref<?xi8>
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %continuation_mask = arith.constant 192 : i64
    %continuation_tag = arith.constant 128 : i64
    %count = scf.for %index = %c0 to %byte_count step %c1 iter_args(%current = %zero) -> (i64) {
      %byte = memref.load %bytes[%index] : memref<?xi8>
      %byte_i64 = arith.extui %byte : i8 to i64
      %tag = arith.andi %byte_i64, %continuation_mask : i64
      %is_continuation = arith.cmpi eq, %tag, %continuation_tag : i64
      %next = scf.if %is_continuation -> (i64) {
        scf.yield %current : i64
      } else {
        %incremented = arith.addi %current, %one : i64
        scf.yield %incremented : i64
      }
      scf.yield %next : i64
    }
    func.return %count : i64
  }

  func.func @LyUnicode_EqBool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_bytes: memref<?xi8>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_bytes: memref<?xi8>) -> i1 {
    %lhs_len = func.call @LyUnicode_Length(%lhs_header, %lhs_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %rhs_len = func.call @LyUnicode_Length(%rhs_header, %rhs_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %same_len = arith.cmpi eq, %lhs_len, %rhs_len : i64
    %result = scf.if %same_len -> (i1) {
      %lower = arith.constant 0 : index
      %upper = arith.index_cast %lhs_len : i64 to index
      %step = arith.constant 1 : index
      %true = arith.constant true
      %all_equal = scf.for %index = %lower to %upper step %step iter_args(%current = %true) -> (i1) {
        %lhs_byte = memref.load %lhs_bytes[%index] : memref<?xi8>
        %rhs_byte = memref.load %rhs_bytes[%index] : memref<?xi8>
        %byte_equal = arith.cmpi eq, %lhs_byte, %rhs_byte : i8
        %next = arith.andi %current, %byte_equal : i1
        scf.yield %next : i1
      }
      scf.yield %all_equal : i1
    } else {
      %false = arith.constant false
      scf.yield %false : i1
    }
    func.return %result : i1
  }

  func.func @LyUnicode_Copy(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0]} {
    %start = arith.constant 0 : index
    %length = func.call @LyUnicode_Length(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %new_header, %new_bytes = func.call @LyUnicode_FromBytes(%bytes, %start, %length) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %new_header, %new_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_Concat(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_bytes: memref<?xi8>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0]} {
    %lhs_len = func.call @LyUnicode_Length(%lhs_header, %lhs_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %rhs_len = func.call @LyUnicode_Length(%rhs_header, %rhs_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %total_len = arith.addi %lhs_len, %rhs_len : i64
    %header, %bytes = func.call @__ly_unicode_alloc(%total_len) : (i64) -> (memref<2xi64>, memref<?xi8>)
    %zero = arith.constant 0 : index
    %lhs_upper = arith.index_cast %lhs_len : i64 to index
    %rhs_upper = arith.index_cast %rhs_len : i64 to index
    %lhs_offset = arith.index_cast %lhs_len : i64 to index
    %step = arith.constant 1 : index
    scf.for %index = %zero to %lhs_upper step %step {
      %byte = memref.load %lhs_bytes[%index] : memref<?xi8>
      memref.store %byte, %bytes[%index] : memref<?xi8>
    }
    scf.for %index = %zero to %rhs_upper step %step {
      %byte = memref.load %rhs_bytes[%index] : memref<?xi8>
      %dest = arith.addi %lhs_offset, %index : index
      memref.store %byte, %bytes[%dest] : memref<?xi8>
    }
    func.return %header, %bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_Concat3(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_bytes: memref<?xi8>, %mid_header: memref<2xi64> {ly.ownership.object_header}, %mid_bytes: memref<?xi8>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0]} {
    %lhs_len = func.call @LyUnicode_Length(%lhs_header, %lhs_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %mid_len = func.call @LyUnicode_Length(%mid_header, %mid_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %rhs_len = func.call @LyUnicode_Length(%rhs_header, %rhs_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %lhs_mid_len = arith.addi %lhs_len, %mid_len : i64
    %total_len = arith.addi %lhs_mid_len, %rhs_len : i64
    %header, %bytes = func.call @__ly_unicode_alloc(%total_len) : (i64) -> (memref<2xi64>, memref<?xi8>)
    %zero = arith.constant 0 : index
    %lhs_upper = arith.index_cast %lhs_len : i64 to index
    %mid_upper = arith.index_cast %mid_len : i64 to index
    %rhs_upper = arith.index_cast %rhs_len : i64 to index
    %mid_offset = arith.index_cast %lhs_len : i64 to index
    %rhs_offset = arith.index_cast %lhs_mid_len : i64 to index
    %step = arith.constant 1 : index
    scf.for %index = %zero to %lhs_upper step %step {
      %byte = memref.load %lhs_bytes[%index] : memref<?xi8>
      memref.store %byte, %bytes[%index] : memref<?xi8>
    }
    scf.for %index = %zero to %mid_upper step %step {
      %byte = memref.load %mid_bytes[%index] : memref<?xi8>
      %dest = arith.addi %mid_offset, %index : index
      memref.store %byte, %bytes[%dest] : memref<?xi8>
    }
    scf.for %index = %zero to %rhs_upper step %step {
      %byte = memref.load %rhs_bytes[%index] : memref<?xi8>
      %dest = arith.addi %rhs_offset, %index : index
      memref.store %byte, %bytes[%dest] : memref<?xi8>
    }
    func.return %header, %bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_PrintLine(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) {
    %length = func.call @LyUnicode_Length(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    func.call @LyHost_PrintLine(%bytes, %length) : (memref<?xi8>, i64) -> ()
    func.return
  }

  func.func @LyUnicode_Print(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) {
    %length = func.call @LyUnicode_Length(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    func.call @LyHost_Print(%bytes, %length) : (memref<?xi8>, i64) -> ()
    func.return
  }

  func.func @LyUnicode_FromI64(%value: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0]} {
    %buffer = memref.alloca() : memref<21xi8>
    %zero = arith.constant 0 : i64
    %ten = arith.constant 10 : i64
    %ascii_zero = arith.constant 48 : i64
    %ascii_minus = arith.constant 45 : i8
    %end = arith.constant 20 : index
    %last = arith.constant 19 : index

    %is_negative = arith.cmpi slt, %value, %zero : i64
    %negated = arith.subi %zero, %value : i64
    %abs_value = arith.select %is_negative, %negated, %value : i64
    %is_zero = arith.cmpi eq, %abs_value, %zero : i64
    cf.cond_br %is_zero, ^format_zero, ^format_digits

  ^format_zero:
    %zero_ch_i64 = arith.constant 48 : i64
    %zero_ch = arith.trunci %zero_ch_i64 : i64 to i8
    memref.store %zero_ch, %buffer[%last] : memref<21xi8>
    cf.br ^finish(%last : index)

  ^format_digits:
    %lower = arith.constant 0 : index
    %upper = arith.constant 20 : index
    %step = arith.constant 1 : index
    %result:2 = scf.for %i = %lower to %upper step %step iter_args(%n = %abs_value, %pos = %last) -> (i64, index) {
      %active = arith.cmpi ne, %n, %zero : i64
      %next:2 = scf.if %active -> (i64, index) {
        %digit = arith.remui %n, %ten : i64
        %digit_ch_i64 = arith.addi %digit, %ascii_zero : i64
        %digit_ch = arith.trunci %digit_ch_i64 : i64 to i8
        memref.store %digit_ch, %buffer[%pos] : memref<21xi8>
        %quotient = arith.divui %n, %ten : i64
        %one_index = arith.constant 1 : index
        %next_pos = arith.subi %pos, %one_index : index
        scf.yield %quotient, %next_pos : i64, index
      } else {
        scf.yield %n, %pos : i64, index
      }
      scf.yield %next#0, %next#1 : i64, index
    }
    %one_finish = arith.constant 1 : index
    %first_digit = arith.addi %result#1, %one_finish : index
    %start = scf.if %is_negative -> (index) {
      %minus_pos = arith.subi %first_digit, %one_finish : index
      memref.store %ascii_minus, %buffer[%minus_pos] : memref<21xi8>
      scf.yield %minus_pos : index
    } else {
      scf.yield %first_digit : index
    }
    cf.br ^finish(%start : index)

  ^finish(%start_index: index):
    %length_index = arith.subi %end, %start_index : index
    %length = arith.index_cast %length_index : index to i64
    %buffer_view = memref.cast %buffer : memref<21xi8> to memref<?xi8>
    %header, %bytes = func.call @LyUnicode_FromBytes(%buffer_view, %start_index, %length) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %header, %bytes : memref<2xi64>, memref<?xi8>
  }
}
