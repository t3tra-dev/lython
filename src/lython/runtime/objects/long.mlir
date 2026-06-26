module attributes {ly.runtime.contracts = ["builtins.int"]} {
  func.func private @LyLong_DecRef(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %digits: memref<?xi32>) attributes {ly.ownership.release_args = [0]}
  func.func private @LyFloat_FromF64(%value: f64) -> (memref<2xi64>, memref<1xf64>)
  func.func private @LyBaseException_New(%class_id: i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
  func.func private @LyBaseException_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
  func.func private @LyEH_ThrowException(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>)
  func.func private @LyUnicode_FromBytes(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0]}
  func.func private @LyUnicode_FromI64(%value: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0]}

  memref.global "private" constant @__ly_long_msg_division_by_zero : memref<16xi8> = dense<[100, 105, 118, 105, 115, 105, 111, 110, 32, 98, 121, 32, 122, 101, 114, 111]>
  memref.global "private" constant @__ly_long_msg_integer_division_or_modulo_by_zero : memref<34xi8> = dense<[105, 110, 116, 101, 103, 101, 114, 32, 100, 105, 118, 105, 115, 105, 111, 110, 32, 111, 114, 32, 109, 111, 100, 117, 108, 111, 32, 98, 121, 32, 122, 101, 114, 111]>
  memref.global "private" constant @__ly_long_msg_integer_modulo_by_zero : memref<22xi8> = dense<[105, 110, 116, 101, 103, 101, 114, 32, 109, 111, 100, 117, 108, 111, 32, 98, 121, 32, 122, 101, 114, 111]>
  memref.global "private" constant @__ly_long_msg_negative_shift_count : memref<20xi8> = dense<[110, 101, 103, 97, 116, 105, 118, 101, 32, 115, 104, 105, 102, 116, 32, 99, 111, 117, 110, 116]>
  memref.global "private" constant @__ly_long_zero_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" constant @__ly_long_zero_meta : memref<2xi64> = dense<[0, 0]>
  memref.global "private" constant @__ly_long_zero_digits : memref<1xi32> = dense<[0]>
  memref.global "private" constant @__ly_long_one_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" constant @__ly_long_one_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" constant @__ly_long_one_digits : memref<1xi32> = dense<[1]>
  memref.global "private" constant @__ly_long_two_header : memref<2xi64> = dense<[9223372036854775807, 1]>
  memref.global "private" constant @__ly_long_two_meta : memref<2xi64> = dense<[1, 1]>
  memref.global "private" constant @__ly_long_two_digits : memref<1xi32> = dense<[2]>

  // Long values are represented as ordinary heap-backed runtime objects here.
  // The runtime manifest must not fabricate memref descriptors from unrelated
  // LLVM values. If immediates are reintroduced later, they should be modeled as
  // a real lowering ABI primitive before entering this object implementation.
  func.func private @__ly_long_operand_view(%meta: memref<2xi64>, %digits: memref<?xi32>) -> (memref<2xi64>, memref<?xi32>) {
    func.return %meta, %digits : memref<2xi64>, memref<?xi32>
  }

  func.func private @__ly_long_raise_message(%class_id: i64, %message: memref<?xi8>, %length: i64) {
    %start = arith.constant 0 : index
    %exception:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    %message_header, %message_bytes = func.call @LyUnicode_FromBytes(%message, %start, %length) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %initialized:3 = func.call @LyBaseException_Init(%exception#0, %exception#1, %exception#2, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.call @LyEH_ThrowException(%initialized#0, %initialized#1, %initialized#2) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @__ly_long_raise_true_div_zero() {
    %class_id = arith.constant 61 : i64
    %length = arith.constant 16 : i64
    %message_static = memref.get_global @__ly_long_msg_division_by_zero : memref<16xi8>
    %message = memref.cast %message_static : memref<16xi8> to memref<?xi8>
    func.call @__ly_long_raise_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @__ly_long_raise_floor_div_zero() {
    %class_id = arith.constant 61 : i64
    %length = arith.constant 34 : i64
    %message_static = memref.get_global @__ly_long_msg_integer_division_or_modulo_by_zero : memref<34xi8>
    %message = memref.cast %message_static : memref<34xi8> to memref<?xi8>
    func.call @__ly_long_raise_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @__ly_long_raise_mod_zero() {
    %class_id = arith.constant 61 : i64
    %length = arith.constant 22 : i64
    %message_static = memref.get_global @__ly_long_msg_integer_modulo_by_zero : memref<22xi8>
    %message = memref.cast %message_static : memref<22xi8> to memref<?xi8>
    func.call @__ly_long_raise_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @__ly_long_raise_negative_shift() {
    %class_id = arith.constant 53 : i64
    %length = arith.constant 20 : i64
    %message_static = memref.get_global @__ly_long_msg_negative_shift_count : memref<20xi8>
    %message = memref.cast %message_static : memref<20xi8> to memref<?xi8>
    func.call @__ly_long_raise_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @__ly_long_alloc_raw(%sign: i64, %capacity: i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %layout_int = arith.constant 1 : i64
    %needs_one = arith.cmpi sle, %capacity, %zero : i64
    %alloc_count_i64 = arith.select %needs_one, %one, %capacity : i1, i64
    %alloc_count = arith.index_cast %alloc_count_i64 : i64 to index
    %header = memref.alloc() {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<2xi64>
    %meta = memref.alloc() : memref<2xi64>
    %digits = memref.alloc(%alloc_count) : memref<?xi32>
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
    func.return %header, %meta, %digits : memref<2xi64>, memref<2xi64>, memref<?xi32>
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

  func.func private @__ly_long_copy_with_sign(%sign: i64, %meta_in: memref<2xi64>, %digits_in: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) {
    %count_slot = arith.constant 1 : index
    %count = memref.load %meta_in[%count_slot] : memref<2xi64>
    %header, %meta, %digits = func.call @__ly_long_alloc_raw(%sign, %count) : (i64, i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %count_index = arith.index_cast %count : i64 to index
    scf.for %iv = %c0 to %count_index step %c1 {
      %digit = memref.load %digits_in[%iv] : memref<?xi32>
      memref.store %digit, %digits[%iv] : memref<?xi32>
    }
    func.call @__ly_long_normalize(%meta, %digits, %count) : (memref<2xi64>, memref<?xi32>, i64) -> ()
    func.return %header, %meta, %digits : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func private @__ly_long_copy(%meta_in: memref<2xi64>, %digits_in: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) {
    %sign_slot = arith.constant 0 : index
    %sign = memref.load %meta_in[%sign_slot] : memref<2xi64>
    %header, %meta, %digits = func.call @__ly_long_copy_with_sign(%sign, %meta_in, %digits_in) : (i64, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %header, %meta, %digits : memref<2xi64>, memref<2xi64>, memref<?xi32>
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

  func.func private @__ly_long_add_abs(%sign: i64, %lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) {
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
    %header, %meta, %digits = func.call @__ly_long_alloc_raw(%sign, %capacity) : (i64, i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
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
    func.return %header, %meta, %digits : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func private @__ly_long_sub_abs(%sign: i64, %lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %base = arith.constant 1073741824 : i64
    %count_slot = arith.constant 1 : index
    %lhs_count = memref.load %lhs_meta[%count_slot] : memref<2xi64>
    %rhs_count = memref.load %rhs_meta[%count_slot] : memref<2xi64>
    %header, %meta, %digits = func.call @__ly_long_alloc_raw(%sign, %lhs_count) : (i64, i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
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
    func.return %header, %meta, %digits : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func private @__ly_long_add_signed_general(%lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_effective_sign: i64, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) {
    %zero = arith.constant 0 : i64
    %sign_slot = arith.constant 0 : index
    %lhs_sign = memref.load %lhs_meta[%sign_slot] : memref<2xi64>
    %lhs_zero = arith.cmpi eq, %lhs_sign, %zero : i64
    %result:3 = scf.if %lhs_zero -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) {
      %h, %m, %d = func.call @__ly_long_copy_with_sign(%rhs_effective_sign, %rhs_meta, %rhs_digits) : (i64, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
      scf.yield %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
    } else {
      %rhs_zero = arith.cmpi eq, %rhs_effective_sign, %zero : i64
      %inner:3 = scf.if %rhs_zero -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) {
        %h, %m, %d = func.call @__ly_long_copy(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
        scf.yield %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
      } else {
        %same_sign = arith.cmpi eq, %lhs_sign, %rhs_effective_sign : i64
        %combined:3 = scf.if %same_sign -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) {
          %h, %m, %d = func.call @__ly_long_add_abs(%lhs_sign, %lhs_meta, %lhs_digits, %rhs_meta, %rhs_digits) : (i64, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
          scf.yield %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
        } else {
          %cmp = func.call @__ly_long_abs_compare(%lhs_meta, %lhs_digits, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<?xi32>) -> i64
          %lhs_bigger = arith.cmpi sgt, %cmp, %zero : i64
          %abs_result:3 = scf.if %lhs_bigger -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) {
            %h, %m, %d = func.call @__ly_long_sub_abs(%lhs_sign, %lhs_meta, %lhs_digits, %rhs_meta, %rhs_digits) : (i64, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
            scf.yield %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
          } else {
            %rhs_bigger = arith.cmpi slt, %cmp, %zero : i64
            %rhs_result:3 = scf.if %rhs_bigger -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) {
              %h, %m, %d = func.call @__ly_long_sub_abs(%rhs_effective_sign, %rhs_meta, %rhs_digits, %lhs_meta, %lhs_digits) : (i64, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
              scf.yield %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
            } else {
              %h, %m, %d = func.call @LyLong_FromI64(%zero) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
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
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyLong_FromI64(%value: i64 {ly.runtime.default_i64 = 0 : i64}) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 1 : i64, ly.runtime.contract = "builtins.int", ly.runtime.initializer = "__new__"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %is_cached_zero = arith.cmpi eq, %value, %zero : i64
    cf.cond_br %is_cached_zero, ^cached_zero, ^check_cached_one

  ^cached_zero:
    %zero_header = memref.get_global @__ly_long_zero_header : memref<2xi64>
    %zero_meta = memref.get_global @__ly_long_zero_meta : memref<2xi64>
    %zero_digits_static = memref.get_global @__ly_long_zero_digits : memref<1xi32>
    %zero_digits = memref.cast %zero_digits_static : memref<1xi32> to memref<?xi32>
    func.return %zero_header, %zero_meta, %zero_digits : memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^check_cached_one:
    %is_cached_one = arith.cmpi eq, %value, %one : i64
    cf.cond_br %is_cached_one, ^cached_one, ^check_cached_two

  ^cached_one:
    %one_header = memref.get_global @__ly_long_one_header : memref<2xi64>
    %one_meta = memref.get_global @__ly_long_one_meta : memref<2xi64>
    %one_digits_static = memref.get_global @__ly_long_one_digits : memref<1xi32>
    %one_digits = memref.cast %one_digits_static : memref<1xi32> to memref<?xi32>
    func.return %one_header, %one_meta, %one_digits : memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^check_cached_two:
    %is_cached_two = arith.cmpi eq, %value, %two : i64
    cf.cond_br %is_cached_two, ^cached_two, ^heap

  ^cached_two:
    %two_header = memref.get_global @__ly_long_two_header : memref<2xi64>
    %two_meta = memref.get_global @__ly_long_two_meta : memref<2xi64>
    %two_digits_static = memref.get_global @__ly_long_two_digits : memref<1xi32>
    %two_digits = memref.cast %two_digits_static : memref<1xi32> to memref<?xi32>
    func.return %two_header, %two_meta, %two_digits : memref<2xi64>, memref<2xi64>, memref<?xi32>

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
    %header = memref.alloc() {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<2xi64>
    %meta = memref.alloc() : memref<2xi64>
    %digits = memref.alloc(%alloc_digits) : memref<?xi32>
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
    func.return %header, %meta, %digits : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyLong_AsI64(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %digits: memref<?xi32>) -> i64 attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__int__", ly.runtime.primitive = "unbox.i64"} {
    %result = func.call @__ly_long_view_as_i64(%meta, %digits) : (memref<2xi64>, memref<?xi32>) -> i64
    func.return %result : i64
  }

  func.func @LyLong_AsF64(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %digits: memref<?xi32>) -> f64 attributes {ly.runtime.contract = "builtins.int", ly.runtime.primitive = "unbox.f64"} {
    %value = func.call @LyLong_AsI64(%header, %meta, %digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %float_value = arith.sitofp %value : i64 to f64
    func.return %float_value : f64
  }

  func.func @LyLong_Init(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %digits: memref<?xi32>) attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__init__"} {
    func.return
  }

  func.func @LyLong_Bool(%header: memref<2xi64> {ly.ownership.object_header}, %meta_raw: memref<2xi64>, %digits_raw: memref<?xi32>) -> i1 attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__bool__"} {
    %meta, %digits = func.call @__ly_long_operand_view(%meta_raw, %digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %count_slot = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %count = memref.load %meta[%count_slot] : memref<2xi64>
    %result = arith.cmpi ne, %count, %zero : i64
    func.return %result : i1
  }

  func.func @LyLong_Pos(%header: memref<2xi64> {ly.ownership.object_header}, %meta_raw: memref<2xi64>, %digits_raw: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__pos__"} {
    %meta, %digits = func.call @__ly_long_operand_view(%meta_raw, %digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %h, %m, %d = func.call @__ly_long_copy(%meta, %digits) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyLong_Neg(%header: memref<2xi64> {ly.ownership.object_header}, %meta_raw: memref<2xi64>, %digits_raw: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__neg__"} {
    %meta, %digits = func.call @__ly_long_operand_view(%meta_raw, %digits_raw) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %sign_slot = arith.constant 0 : index
    %zero = arith.constant 0 : i64
    %sign = memref.load %meta[%sign_slot] : memref<2xi64>
    %negated = arith.subi %zero, %sign : i64
    %h, %m, %d = func.call @__ly_long_copy_with_sign(%negated, %meta, %digits) : (i64, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyLong_Invert(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %digits: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__invert__"} {
    %neg_h, %neg_m, %neg_d = func.call @LyLong_Neg(%header, %meta, %digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    %one = arith.constant 1 : i64
    %one_h, %one_m, %one_d = func.call @LyLong_FromI64(%one) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    %result_h, %result_m, %result_d = func.call @LyLong_Sub(%neg_h, %neg_m, %neg_d, %one_h, %one_m, %one_d) : (memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.call @LyLong_DecRef(%neg_h, %neg_m, %neg_d) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> ()
    func.call @LyLong_DecRef(%one_h, %one_m, %one_d) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> ()
    func.return %result_h, %result_m, %result_d : memref<2xi64>, memref<2xi64>, memref<?xi32>
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

  func.func @LyLong_Add(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta_raw: memref<2xi64>, %lhs_digits_raw: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta_raw: memref<2xi64>, %rhs_digits_raw: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__add__"} {
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
    %sh, %sm, %sd = func.call @LyLong_FromI64(%small_sum) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %sh, %sm, %sd : memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^digits:
    %sign_slot = arith.constant 0 : index
    %rhs_sign = memref.load %rhs_meta[%sign_slot] : memref<2xi64>
    %h, %m, %d = func.call @__ly_long_add_signed_general(%lhs_meta, %lhs_digits, %rhs_sign, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>, i64, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyLong_Sub(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta_raw: memref<2xi64>, %rhs_digits_raw: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__sub__"} {
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
    %sh, %sm, %sd = func.call @LyLong_FromI64(%small_diff) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %sh, %sm, %sd : memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^digits:
    %zero = arith.constant 0 : i64
    %sign_slot = arith.constant 0 : index
    %rhs_sign = memref.load %rhs_meta[%sign_slot] : memref<2xi64>
    %rhs_neg = arith.subi %zero, %rhs_sign : i64
    %h, %m, %d = func.call @__ly_long_add_signed_general(%lhs_meta_view, %lhs_digits_view, %rhs_neg, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>, i64, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyLong_Mul(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta_raw: memref<2xi64>, %lhs_digits_raw: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta_raw: memref<2xi64>, %rhs_digits_raw: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__mul__"} {
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
    %sh, %sm, %sd = func.call @LyLong_FromI64(%small_low) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %sh, %sm, %sd : memref<2xi64>, memref<2xi64>, memref<?xi32>

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
      %h, %m, %d = func.call @LyLong_FromI64(%zero) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
      scf.yield %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
    } else {
      %sign = arith.muli %lhs_sign, %rhs_sign : i64
      %sum_count = arith.addi %lhs_count, %rhs_count : i64
      %capacity = arith.addi %sum_count, %one : i64
      %h, %m, %d = func.call @__ly_long_alloc_raw(%sign, %capacity) : (i64, i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
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
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyLong_TrueDiv(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> (memref<2xi64>, memref<1xf64>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__truediv__"} {
    %lhs = func.call @LyLong_AsF64(%lhs_header, %lhs_meta, %lhs_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> f64
    %rhs = func.call @LyLong_AsF64(%rhs_header, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> f64
    %zero = arith.constant 0.0 : f64
    %rhs_nonzero = arith.cmpf one, %rhs, %zero : f64
    %result:2 = scf.if %rhs_nonzero -> (memref<2xi64>, memref<1xf64>) {
      %quotient = arith.divf %lhs, %rhs : f64
      %h, %p = func.call @LyFloat_FromF64(%quotient) : (f64) -> (memref<2xi64>, memref<1xf64>)
      scf.yield %h, %p : memref<2xi64>, memref<1xf64>
    } else {
      func.call @__ly_long_raise_true_div_zero() : () -> ()
      %dummy = arith.constant 0.0 : f64
      %h, %p = func.call @LyFloat_FromF64(%dummy) : (f64) -> (memref<2xi64>, memref<1xf64>)
      scf.yield %h, %p : memref<2xi64>, memref<1xf64>
    }
    func.return %result#0, %result#1 : memref<2xi64>, memref<1xf64>
  }

  func.func @LyLong_FloorDiv(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__floordiv__"} {
    %lhs = func.call @LyLong_AsI64(%lhs_header, %lhs_meta, %lhs_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %rhs = func.call @LyLong_AsI64(%rhs_header, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %rhs_nonzero = arith.cmpi ne, %rhs, %zero : i64
    %result:3 = scf.if %rhs_nonzero -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) {
      %trunc_q = arith.divsi %lhs, %rhs : i64
      %trunc_r = arith.remsi %lhs, %rhs : i64
      %has_remainder = arith.cmpi ne, %trunc_r, %zero : i64
      %lhs_negative = arith.cmpi slt, %lhs, %zero : i64
      %rhs_negative = arith.cmpi slt, %rhs, %zero : i64
      %different_sign = arith.cmpi ne, %lhs_negative, %rhs_negative : i1
      %adjust = arith.andi %has_remainder, %different_sign : i1
      %decremented = arith.subi %trunc_q, %one : i64
      %floor_q = arith.select %adjust, %decremented, %trunc_q : i1, i64
      %h, %m, %d = func.call @LyLong_FromI64(%floor_q) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
      scf.yield %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
    } else {
      func.call @__ly_long_raise_floor_div_zero() : () -> ()
      %h, %m, %d = func.call @LyLong_FromI64(%zero) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
      scf.yield %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
    }
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyLong_Mod(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__mod__"} {
    %lhs = func.call @LyLong_AsI64(%lhs_header, %lhs_meta, %lhs_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %rhs = func.call @LyLong_AsI64(%rhs_header, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %zero = arith.constant 0 : i64
    %rhs_nonzero = arith.cmpi ne, %rhs, %zero : i64
    %result:3 = scf.if %rhs_nonzero -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) {
      %trunc_r = arith.remsi %lhs, %rhs : i64
      %has_remainder = arith.cmpi ne, %trunc_r, %zero : i64
      %remainder_negative = arith.cmpi slt, %trunc_r, %zero : i64
      %rhs_negative = arith.cmpi slt, %rhs, %zero : i64
      %different_sign = arith.cmpi ne, %remainder_negative, %rhs_negative : i1
      %adjust = arith.andi %has_remainder, %different_sign : i1
      %adjusted = arith.addi %trunc_r, %rhs : i64
      %mod = arith.select %adjust, %adjusted, %trunc_r : i1, i64
      %h, %m, %d = func.call @LyLong_FromI64(%mod) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
      scf.yield %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
    } else {
      func.call @__ly_long_raise_mod_zero() : () -> ()
      %h, %m, %d = func.call @LyLong_FromI64(%zero) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
      scf.yield %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
    }
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyLong_BitAnd(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__and__"} {
    %lhs = func.call @LyLong_AsI64(%lhs_header, %lhs_meta, %lhs_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %rhs = func.call @LyLong_AsI64(%rhs_header, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %value = arith.andi %lhs, %rhs : i64
    %h, %m, %d = func.call @LyLong_FromI64(%value) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyLong_BitOr(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__or__"} {
    %lhs = func.call @LyLong_AsI64(%lhs_header, %lhs_meta, %lhs_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %rhs = func.call @LyLong_AsI64(%rhs_header, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %value = arith.ori %lhs, %rhs : i64
    %h, %m, %d = func.call @LyLong_FromI64(%value) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyLong_BitXor(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__xor__"} {
    %lhs = func.call @LyLong_AsI64(%lhs_header, %lhs_meta, %lhs_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %rhs = func.call @LyLong_AsI64(%rhs_header, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %value = arith.xori %lhs, %rhs : i64
    %h, %m, %d = func.call @LyLong_FromI64(%value) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyLong_LShift(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__lshift__"} {
    %lhs = func.call @LyLong_AsI64(%lhs_header, %lhs_meta, %lhs_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %rhs = func.call @LyLong_AsI64(%rhs_header, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %zero = arith.constant 0 : i64
    %nonnegative = arith.cmpi sge, %rhs, %zero : i64
    %result:3 = scf.if %nonnegative -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) {
      %value = arith.shli %lhs, %rhs : i64
      %h, %m, %d = func.call @LyLong_FromI64(%value) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
      scf.yield %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
    } else {
      func.call @__ly_long_raise_negative_shift() : () -> ()
      %h, %m, %d = func.call @LyLong_FromI64(%zero) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
      scf.yield %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
    }
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyLong_RShift(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__rshift__"} {
    %lhs = func.call @LyLong_AsI64(%lhs_header, %lhs_meta, %lhs_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %rhs = func.call @LyLong_AsI64(%rhs_header, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %zero = arith.constant 0 : i64
    %nonnegative = arith.cmpi sge, %rhs, %zero : i64
    %result:3 = scf.if %nonnegative -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) {
      %value = arith.shrsi %lhs, %rhs : i64
      %h, %m, %d = func.call @LyLong_FromI64(%value) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
      scf.yield %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
    } else {
      func.call @__ly_long_raise_negative_shift() : () -> ()
      %h, %m, %d = func.call @LyLong_FromI64(%zero) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
      scf.yield %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
    }
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyLong_Round(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %digits: memref<?xi32>, %ndigits: i64 {ly.runtime.default_i64 = 0 : i64}) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__round__"} {
    %value = func.call @LyLong_AsI64(%header, %meta, %digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %zero = arith.constant 0 : i64
    %round_to_integer = arith.cmpi sge, %ndigits, %zero : i64
    %rounded = scf.if %round_to_integer -> (i64) {
      scf.yield %value : i64
    } else {
      %one = arith.constant 1 : i64
      %two = arith.constant 2 : i64
      %ten = arith.constant 10 : i64
      %eighteen = arith.constant 18 : i64
      %places = arith.subi %zero, %ndigits : i64
      %too_large = arith.cmpi sgt, %places, %eighteen : i64
      %limited_places = arith.select %too_large, %eighteen, %places : i1, i64
      %upper = arith.index_cast %limited_places : i64 to index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %scale = scf.for %i = %c0 to %upper step %c1 iter_args(%current = %one) -> (i64) {
        %next = arith.muli %current, %ten : i64
        scf.yield %next : i64
      }
      %negative = arith.cmpi slt, %value, %zero : i64
      %negated = arith.subi %zero, %value : i64
      %abs_value = arith.select %negative, %negated, %value : i1, i64
      %quotient = arith.divsi %abs_value, %scale : i64
      %remainder = arith.remsi %abs_value, %scale : i64
      %twice_remainder = arith.muli %remainder, %two : i64
      %greater_than_half = arith.cmpi sgt, %twice_remainder, %scale : i64
      %exactly_half = arith.cmpi eq, %twice_remainder, %scale : i64
      %low_bit = arith.andi %quotient, %one : i64
      %quotient_odd = arith.cmpi ne, %low_bit, %zero : i64
      %tie_rounds_up = arith.andi %exactly_half, %quotient_odd : i1
      %rounds_up = arith.ori %greater_than_half, %tie_rounds_up : i1
      %incremented = arith.addi %quotient, %one : i64
      %rounded_quotient = arith.select %rounds_up, %incremented, %quotient : i1, i64
      %rounded_abs = arith.muli %rounded_quotient, %scale : i64
      %rounded_neg = arith.subi %zero, %rounded_abs : i64
      %signed = arith.select %negative, %rounded_neg, %rounded_abs : i1, i64
      scf.yield %signed : i64
    }
    %h, %m, %d = func.call @LyLong_FromI64(%rounded) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyLong_Compare(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta_raw: memref<2xi64>, %lhs_digits_raw: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta_raw: memref<2xi64>, %rhs_digits_raw: memref<?xi32>) -> i64 attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__richcompare__"} {
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

  func.func @LyLong_EqBool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> i1 attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__eq__"} {
    %cmp = func.call @LyLong_Compare(%lhs_header, %lhs_meta, %lhs_digits, %rhs_header, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %zero = arith.constant 0 : i64
    %result = arith.cmpi eq, %cmp, %zero : i64
    func.return %result : i1
  }

  func.func @LyLong_NeBool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> i1 attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__ne__"} {
    %cmp = func.call @LyLong_Compare(%lhs_header, %lhs_meta, %lhs_digits, %rhs_header, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %zero = arith.constant 0 : i64
    %result = arith.cmpi ne, %cmp, %zero : i64
    func.return %result : i1
  }

  func.func @LyLong_LtBool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> i1 attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__lt__"} {
    %cmp = func.call @LyLong_Compare(%lhs_header, %lhs_meta, %lhs_digits, %rhs_header, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %zero = arith.constant 0 : i64
    %result = arith.cmpi slt, %cmp, %zero : i64
    func.return %result : i1
  }

  func.func @LyLong_LeBool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> i1 attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__le__"} {
    %cmp = func.call @LyLong_Compare(%lhs_header, %lhs_meta, %lhs_digits, %rhs_header, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %zero = arith.constant 0 : i64
    %result = arith.cmpi sle, %cmp, %zero : i64
    func.return %result : i1
  }

  func.func @LyLong_GtBool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> i1 attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__gt__"} {
    %cmp = func.call @LyLong_Compare(%lhs_header, %lhs_meta, %lhs_digits, %rhs_header, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %zero = arith.constant 0 : i64
    %result = arith.cmpi sgt, %cmp, %zero : i64
    func.return %result : i1
  }

  func.func @LyLong_GeBool(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta: memref<2xi64>, %rhs_digits: memref<?xi32>) -> i1 attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__ge__"} {
    %cmp = func.call @LyLong_Compare(%lhs_header, %lhs_meta, %lhs_digits, %rhs_header, %rhs_meta, %rhs_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %zero = arith.constant 0 : i64
    %result = arith.cmpi sge, %cmp, %zero : i64
    func.return %result : i1
  }

  func.func @LyLong_Repr(%header: memref<2xi64> {ly.ownership.object_header}, %meta_raw: memref<2xi64>, %digits_raw: memref<?xi32>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__repr__"} {
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
      %h, %b = func.call @LyUnicode_FromBytes(%buffer, %start, %length) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
      memref.dealloc %buffer : memref<?xi8>
      memref.dealloc %tmp : memref<?xi32>
      scf.yield %h, %b : memref<2xi64>, memref<?xi8>
    }
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }
}
