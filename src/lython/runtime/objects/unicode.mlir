module attributes {ly.runtime.contracts = ["builtins.object", "builtins.str"]} {
  func.func private @LyHost_Print(memref<?xi8>, i64)
  func.func private @LyHost_PrintLine(memref<?xi8>, i64)

  func.func private @LyBuiltin_Repr() -> (memref<2xi64>, memref<?xi8>) attributes {ly.runtime.builtin = "repr", ly.runtime.builtin_lowering = "method", ly.runtime.builtin_method = "__repr__", ly.runtime.contract = "builtins.object", ly.runtime.primitive = "builtin_repr", ly.runtime.result_contract = "builtins.str"}

  func.func private @__ly_unicode_alloc(%len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.primitive = "alloc"} {
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

  func.func @LyUnicode_FromBytes(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 4 : i64, ly.runtime.contract = "builtins.str", ly.runtime.initializer = "__new__"} {
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

  func.func @LyUnicode_Length(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> i64 attributes {ly.runtime.contract = "builtins.str", ly.runtime.primitive = "byte_length"} {
    %c0 = arith.constant 0 : index
    %length_index = memref.dim %bytes, %c0 : memref<?xi8>
    %length = arith.index_cast %length_index : index to i64
    func.return %length : i64
  }

  func.func @LyUnicode_CodepointLength(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> i64 attributes {ly.runtime.contract = "builtins.str", ly.runtime.method = "__len__"} {
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

  func.func @LyUnicode_Bool(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> i1 attributes {ly.runtime.contract = "builtins.str", ly.runtime.method = "__bool__"} {
    %length = func.call @LyUnicode_Length(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %zero = arith.constant 0 : i64
    %result = arith.cmpi ne, %length, %zero : i64
    func.return %result : i1
  }

  func.func @LyUnicode_EqBool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_bytes: memref<?xi8>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_bytes: memref<?xi8>) -> i1 attributes {ly.runtime.contract = "builtins.str", ly.runtime.method = "__eq__"} {
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

  func.func @LyUnicode_NeBool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_bytes: memref<?xi8>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_bytes: memref<?xi8>) -> i1 attributes {ly.runtime.contract = "builtins.str", ly.runtime.method = "__ne__"} {
    %eq = func.call @LyUnicode_EqBool(%lhs_header, %lhs_bytes, %rhs_header, %rhs_bytes) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> i1
    %true = arith.constant true
    %false = arith.constant false
    %result = arith.select %eq, %false, %true : i1
    func.return %result : i1
  }

  func.func @LyUnicode_GetItem(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %raw_index: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__getitem__"} {
    %codepoints = func.call @LyUnicode_CodepointLength(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %is_negative = arith.cmpi slt, %raw_index, %zero : i64
    %from_end = arith.addi %raw_index, %codepoints : i64
    %index = arith.select %is_negative, %from_end, %raw_index : i1, i64
    %lower_ok = arith.cmpi sge, %index, %zero : i64
    %upper_ok = arith.cmpi slt, %index, %codepoints : i64
    %valid = arith.andi %lower_ok, %upper_ok : i1
    cf.assert %valid, "str index out of range"

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %byte_count = memref.dim %bytes, %c0 : memref<?xi8>
    %false = arith.constant false
    %continuation_mask = arith.constant 192 : i64
    %continuation_tag = arith.constant 128 : i64
    %scan:5 = scf.for %i = %c0 to %byte_count step %c1 iter_args(%ordinal = %zero, %start = %c0, %end = %byte_count, %found = %false, %done = %false) -> (i64, index, index, i1, i1) {
      %byte = memref.load %bytes[%i] : memref<?xi8>
      %byte_i64 = arith.extui %byte : i8 to i64
      %tag = arith.andi %byte_i64, %continuation_mask : i64
      %is_start = arith.cmpi ne, %tag, %continuation_tag : i64
      %ordinal_matches = arith.cmpi eq, %ordinal, %index : i64
      %not_found = arith.cmpi eq, %found, %false : i1
      %not_done = arith.cmpi eq, %done, %false : i1
      %start_candidate = arith.andi %is_start, %ordinal_matches : i1
      %take_start_base = arith.andi %start_candidate, %not_found : i1
      %take_start = arith.andi %take_start_base, %not_done : i1
      %next_start = arith.select %take_start, %i, %start : i1, index
      %next_found = arith.ori %found, %take_start : i1
      %end_candidate = arith.andi %found, %is_start : i1
      %take_end = arith.andi %end_candidate, %not_done : i1
      %next_end = arith.select %take_end, %i, %end : i1, index
      %next_done = arith.ori %done, %take_end : i1
      %incremented = arith.addi %ordinal, %one : i64
      %next_ordinal = arith.select %is_start, %incremented, %ordinal : i1, i64
      scf.yield %next_ordinal, %next_start, %next_end, %next_found, %next_done : i64, index, index, i1, i1
    }
    %length_index = arith.subi %scan#2, %scan#1 : index
    %length = arith.index_cast %length_index : index to i64
    %result_header, %result_bytes = func.call @LyUnicode_FromBytes(%bytes, %scan#1, %length) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result_header, %result_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_Copy(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.primitive = "copy"} {
    %start = arith.constant 0 : index
    %length = func.call @LyUnicode_Length(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %new_header, %new_bytes = func.call @LyUnicode_FromBytes(%bytes, %start, %length) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %new_header, %new_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicode_Concat(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_bytes: memref<?xi8>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__add__"} {
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

  func.func @LyUnicode_PrintLine(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) attributes {ly.runtime.builtin = "print", ly.runtime.builtin_lowering = "method_sink", ly.runtime.builtin_method = "__repr__", ly.runtime.builtin_sink_contract = "builtins.str", ly.runtime.contract = "builtins.str", ly.runtime.primitive = "print_line", ly.runtime.result_contract = "types.NoneType"} {
    %length = func.call @LyUnicode_Length(%header, %bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    func.call @LyHost_PrintLine(%bytes, %length) : (memref<?xi8>, i64) -> ()
    func.return
  }

  func.func @LyUnicode_Print(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) attributes {ly.runtime.contract = "builtins.str", ly.runtime.primitive = "print"} {
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

  func.func @LyUnicode_FromF64(%value: f64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0]} {
    %buffer = memref.alloca() : memref<48xi8>
    %zero_f = arith.constant 0.0 : f64
    %half_f = arith.constant 0.5 : f64
    %scale_f = arith.constant 1000000.0 : f64
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %six = arith.constant 6 : i64
    %ten = arith.constant 10 : i64
    %million = arith.constant 1000000 : i64
    %ascii_zero = arith.constant 48 : i64
    %ascii_minus = arith.constant 45 : i8
    %ascii_dot = arith.constant 46 : i8
    %start_loop = arith.constant 0 : index
    %frac_loop_end = arith.constant 6 : index
    %int_loop_end = arith.constant 20 : index
    %step = arith.constant 1 : index
    %end = arith.constant 48 : index
    %last = arith.constant 47 : index
    %one_index = arith.constant 1 : index

    %is_negative = arith.cmpf olt, %value, %zero_f : f64
    %negated = arith.subf %zero_f, %value : f64
    %abs_value = arith.select %is_negative, %negated, %value : i1, f64
    %raw_int = arith.fptosi %abs_value : f64 to i64
    %raw_int_float = arith.sitofp %raw_int : i64 to f64
    %fraction = arith.subf %abs_value, %raw_int_float : f64
    %scaled_fraction = arith.mulf %fraction, %scale_f : f64
    %rounded_fraction = arith.addf %scaled_fraction, %half_f : f64
    %raw_scaled = arith.fptosi %rounded_fraction : f64 to i64
    %carry_fraction = arith.cmpi eq, %raw_scaled, %million : i64
    %int_carry = arith.select %carry_fraction, %one, %zero : i1, i64
    %int_part = arith.addi %raw_int, %int_carry : i64
    %frac_part = arith.select %carry_fraction, %zero, %raw_scaled : i1, i64

    %trimmed:2 = scf.for %i = %start_loop to %frac_loop_end step %step iter_args(%n = %frac_part, %digits = %six) -> (i64, i64) {
      %more_than_one = arith.cmpi sgt, %digits, %one : i64
      %remainder = arith.remui %n, %ten : i64
      %trailing_zero = arith.cmpi eq, %remainder, %zero : i64
      %can_trim = arith.andi %more_than_one, %trailing_zero : i1
      %quotient = arith.divui %n, %ten : i64
      %next_n = arith.select %can_trim, %quotient, %n : i1, i64
      %decremented_digits = arith.subi %digits, %one : i64
      %next_digits = arith.select %can_trim, %decremented_digits, %digits : i1, i64
      scf.yield %next_n, %next_digits : i64, i64
    }

    %frac_digits_index = arith.index_cast %trimmed#1 : i64 to index
    %after_frac:2 = scf.for %i = %start_loop to %frac_loop_end step %step iter_args(%n = %trimmed#0, %pos = %last) -> (i64, index) {
      %active = arith.cmpi ult, %i, %frac_digits_index : index
      %next:2 = scf.if %active -> (i64, index) {
        %digit = arith.remui %n, %ten : i64
        %digit_ch_i64 = arith.addi %digit, %ascii_zero : i64
        %digit_ch = arith.trunci %digit_ch_i64 : i64 to i8
        memref.store %digit_ch, %buffer[%pos] : memref<48xi8>
        %quotient = arith.divui %n, %ten : i64
        %next_pos = arith.subi %pos, %one_index : index
        scf.yield %quotient, %next_pos : i64, index
      } else {
        scf.yield %n, %pos : i64, index
      }
      scf.yield %next#0, %next#1 : i64, index
    }

    memref.store %ascii_dot, %buffer[%after_frac#1] : memref<48xi8>
    %int_pos = arith.subi %after_frac#1, %one_index : index
    %int_is_zero = arith.cmpi eq, %int_part, %zero : i64
    %after_int:2 = scf.if %int_is_zero -> (i64, index) {
      %zero_ch_i64 = arith.constant 48 : i64
      %zero_ch = arith.trunci %zero_ch_i64 : i64 to i8
      memref.store %zero_ch, %buffer[%int_pos] : memref<48xi8>
      %next_pos = arith.subi %int_pos, %one_index : index
      scf.yield %zero, %next_pos : i64, index
    } else {
      %result:2 = scf.for %i = %start_loop to %int_loop_end step %step iter_args(%n = %int_part, %pos = %int_pos) -> (i64, index) {
        %active = arith.cmpi ne, %n, %zero : i64
        %next:2 = scf.if %active -> (i64, index) {
          %digit = arith.remui %n, %ten : i64
          %digit_ch_i64 = arith.addi %digit, %ascii_zero : i64
          %digit_ch = arith.trunci %digit_ch_i64 : i64 to i8
          memref.store %digit_ch, %buffer[%pos] : memref<48xi8>
          %quotient = arith.divui %n, %ten : i64
          %next_pos = arith.subi %pos, %one_index : index
          scf.yield %quotient, %next_pos : i64, index
        } else {
          scf.yield %n, %pos : i64, index
        }
        scf.yield %next#0, %next#1 : i64, index
      }
      scf.yield %result#0, %result#1 : i64, index
    }

    %first_digit = arith.addi %after_int#1, %one_index : index
    %start_index = scf.if %is_negative -> (index) {
      memref.store %ascii_minus, %buffer[%after_int#1] : memref<48xi8>
      scf.yield %after_int#1 : index
    } else {
      scf.yield %first_digit : index
    }
    %length_index = arith.subi %end, %start_index : index
    %length = arith.index_cast %length_index : index to i64
    %buffer_view = memref.cast %buffer : memref<48xi8> to memref<?xi8>
    %header, %bytes = func.call @LyUnicode_FromBytes(%buffer_view, %start_index, %length) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %header, %bytes : memref<2xi64>, memref<?xi8>
  }
}
