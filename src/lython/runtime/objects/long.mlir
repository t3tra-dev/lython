module {
  func.func private @LyLong_DecRef(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %digits: memref<?xi32>) attributes {ly.ownership.release_args = [0]}
  func.func private @LyUnicode_FromBytes(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0]}
  func.func private @LyUnicode_FromI64(%value: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0]}


  // Builds the three descriptor parts of an inline tagged long
  // (rfc/tagged-long.md). The tagged word (payload << 1 | 1) sits in the
  // pointer lanes of every part; nothing may dereference them. Genuine
  // allocations are at least 8-byte aligned, so bit 0 discriminates.
  func.func private @__ly_long_make_tagged(%payload: i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) {
    %one = arith.constant 1 : i64
    %shifted = arith.shli %payload, %one : i64
    %word = arith.ori %shifted, %one : i64
    %ptr = llvm.inttoptr %word : i64 to !llvm.ptr
    %zero_l = llvm.mlir.constant(0 : i64) : i64
    %one_l = llvm.mlir.constant(1 : i64) : i64
    %two_l = llvm.mlir.constant(2 : i64) : i64
    %undef = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %p0 = llvm.insertvalue %ptr, %undef[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %p1 = llvm.insertvalue %ptr, %p0[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %p2 = llvm.insertvalue %zero_l, %p1[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %w3 = llvm.insertvalue %two_l, %p2[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %w4 = llvm.insertvalue %one_l, %w3[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %header = builtin.unrealized_conversion_cast %w4 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<2xi64> {ly.runtime.descriptor_bridge}
    %meta = builtin.unrealized_conversion_cast %w4 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<2xi64> {ly.runtime.descriptor_bridge}
    %d3 = llvm.insertvalue %zero_l, %p2[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %d4 = llvm.insertvalue %one_l, %d3[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %digits = builtin.unrealized_conversion_cast %d4 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xi32> {ly.runtime.descriptor_bridge}
    func.return %header, %meta, %digits : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  // Selects the digit view of a long operand. Tagged payloads are
  // materialized into caller-provided stack scratch; heap operands pass
  // through untouched. The scratch never escapes the caller frame, and the
  // digit algorithms below stay tag-free.
  func.func private @__ly_long_operand_view(%header: memref<2xi64>, %meta: memref<2xi64>, %digits: memref<?xi32>, %scratch_meta: memref<2xi64>, %scratch_digits: memref<3xi32>) -> (memref<2xi64>, memref<?xi32>) {
    %tag_one = arith.constant 1 : i64
    %ptr = memref.extract_aligned_pointer_as_index %header : memref<2xi64> -> index
    %bits = arith.index_cast %ptr : index to i64
    %bit = arith.andi %bits, %tag_one : i64
    %tagged = arith.cmpi eq, %bit, %tag_one : i64
    cf.cond_br %tagged, ^spill, ^pass

  ^pass:
    func.return %meta, %digits : memref<2xi64>, memref<?xi32>

  ^spill:
    %payload = arith.shrsi %bits, %tag_one : i64
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %is_zero = arith.cmpi eq, %payload, %zero : i64
    %is_negative = arith.cmpi slt, %payload, %zero : i64
    %neg_one = arith.constant -1 : i64
    %signed_sign = arith.select %is_negative, %neg_one, %one : i1, i64
    %sign = arith.select %is_zero, %zero, %signed_sign : i1, i64
    %negated = arith.subi %zero, %payload : i64
    %abs_value = arith.select %is_negative, %negated, %payload : i1, i64
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
    %sign_slot = arith.constant 0 : index
    %count_slot = arith.constant 1 : index
    memref.store %sign, %scratch_meta[%sign_slot] : memref<2xi64>
    memref.store %ndigits, %scratch_meta[%count_slot] : memref<2xi64>
    %d0 = arith.trunci %digit0_i64 : i64 to i32
    %d1 = arith.trunci %digit1_i64 : i64 to i32
    %d2 = arith.trunci %digit2_i64 : i64 to i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    memref.store %d0, %scratch_digits[%c0] : memref<3xi32>
    memref.store %d1, %scratch_digits[%c1] : memref<3xi32>
    memref.store %d2, %scratch_digits[%c2] : memref<3xi32>
    %digits_view = memref.cast %scratch_digits : memref<3xi32> to memref<?xi32>
    func.return %scratch_meta, %digits_view : memref<2xi64>, memref<?xi32>
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

  func.func @LyLong_FromI64(%value: i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0]} {
    // Values that round-trip a 1-bit shift fit the 63-bit tagged payload and
    // never allocate (rfc/tagged-long.md). This also subsumes the small-int
    // cache.
    %tag_one = arith.constant 1 : i64
    %fits_shifted = arith.shli %value, %tag_one : i64
    %fits_back = arith.shrsi %fits_shifted, %tag_one : i64
    %fits = arith.cmpi eq, %fits_back, %value : i64
    cf.cond_br %fits, ^tagged, ^alloc

  ^tagged:
    %th, %tm, %td = func.call @__ly_long_make_tagged(%value) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %th, %tm, %td : memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^alloc:
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
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
    %header, %meta, %digits = func.call @__ly_long_alloc_raw(%sign, %ndigits) : (i64, i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
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
    func.call @__ly_long_normalize(%meta, %digits, %ndigits) : (memref<2xi64>, memref<?xi32>, i64) -> ()
    func.return %header, %meta, %digits : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyLong_AsI64(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %digits: memref<?xi32>) -> i64 {
    %tag_one = arith.constant 1 : i64
    %hdr_ptr = memref.extract_aligned_pointer_as_index %header : memref<2xi64> -> index
    %hdr_bits = arith.index_cast %hdr_ptr : index to i64
    %hdr_tagbit = arith.andi %hdr_bits, %tag_one : i64
    %hdr_tagged = arith.cmpi eq, %hdr_tagbit, %tag_one : i64
    cf.cond_br %hdr_tagged, ^tagged, ^heap

  ^tagged:
    %payload = arith.shrsi %hdr_bits, %tag_one : i64
    func.return %payload : i64

  ^heap:
    %result = func.call @__ly_long_view_as_i64(%meta, %digits) : (memref<2xi64>, memref<?xi32>) -> i64
    func.return %result : i64
  }

  // Reads a (meta, digits) view whose magnitude fits i64 (at most two
  // 30-bit limbs always does). Used by AsI64 and by the small-operand fast
  // paths of the arithmetic entry points.
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

  func.func @LyLong_Add(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta_raw: memref<2xi64>, %lhs_digits_raw: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta_raw: memref<2xi64>, %rhs_digits_raw: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0]} {
    %tag_one = arith.constant 1 : i64
    %lhs_ptr = memref.extract_aligned_pointer_as_index %lhs_header : memref<2xi64> -> index
    %lhs_bits = arith.index_cast %lhs_ptr : index to i64
    %lhs_tagbit = arith.andi %lhs_bits, %tag_one : i64
    %lhs_tagged = arith.cmpi eq, %lhs_tagbit, %tag_one : i64
    %rhs_ptr = memref.extract_aligned_pointer_as_index %rhs_header : memref<2xi64> -> index
    %rhs_bits = arith.index_cast %rhs_ptr : index to i64
    %rhs_tagbit = arith.andi %rhs_bits, %tag_one : i64
    %rhs_tagged = arith.cmpi eq, %rhs_tagbit, %tag_one : i64
    %both_tagged = arith.andi %lhs_tagged, %rhs_tagged : i1
    %lhs_scratch_meta = memref.alloca() : memref<2xi64>
    %lhs_scratch_digits = memref.alloca() : memref<3xi32>
    %rhs_scratch_meta = memref.alloca() : memref<2xi64>
    %rhs_scratch_digits = memref.alloca() : memref<3xi32>
    cf.cond_br %both_tagged, ^fast, ^general

  ^fast:
    // Two 63-bit payloads cannot wrap i64; FromI64 re-tags when it fits.
    %fast_a = arith.shrsi %lhs_bits, %tag_one : i64
    %fast_b = arith.shrsi %rhs_bits, %tag_one : i64
    %fast_sum = arith.addi %fast_a, %fast_b : i64
    %fh, %fm, %fd = func.call @LyLong_FromI64(%fast_sum) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %fh, %fm, %fd : memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^general:
    %lhs_meta, %lhs_digits = func.call @__ly_long_operand_view(%lhs_header, %lhs_meta_raw, %lhs_digits_raw, %lhs_scratch_meta, %lhs_scratch_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<3xi32>) -> (memref<2xi64>, memref<?xi32>)
    %rhs_meta, %rhs_digits = func.call @__ly_long_operand_view(%rhs_header, %rhs_meta_raw, %rhs_digits_raw, %rhs_scratch_meta, %rhs_scratch_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<3xi32>) -> (memref<2xi64>, memref<?xi32>)
    %small_two = arith.constant 2 : i64
    %small_count_slot = arith.constant 1 : index
    %lhs_view_count = memref.load %lhs_meta[%small_count_slot] : memref<2xi64>
    %rhs_view_count = memref.load %rhs_meta[%small_count_slot] : memref<2xi64>
    %lhs_small = arith.cmpi sle, %lhs_view_count, %small_two : i64
    %rhs_small = arith.cmpi sle, %rhs_view_count, %small_two : i64
    %both_small = arith.andi %lhs_small, %rhs_small : i1
    cf.cond_br %both_small, ^small, ^digits

  ^small:
    // At most two 30-bit limbs per side: compute natively and re-enter the
    // tagged domain through FromI64. This absorbs heap-immortal literals
    // into tagged values after a single operation.
    %small_a = func.call @__ly_long_view_as_i64(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %small_b = func.call @__ly_long_view_as_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %small_sum = arith.addi %small_a, %small_b : i64
    %sh, %sm, %sd = func.call @LyLong_FromI64(%small_sum) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %sh, %sm, %sd : memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^digits:
    %zero = arith.constant 0 : i64
    %sign_slot = arith.constant 0 : index
    %lhs_sign = memref.load %lhs_meta[%sign_slot] : memref<2xi64>
    %rhs_sign = memref.load %rhs_meta[%sign_slot] : memref<2xi64>
    %lhs_zero = arith.cmpi eq, %lhs_sign, %zero : i64
    %result:3 = scf.if %lhs_zero -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) {
      %h, %m, %d = func.call @__ly_long_copy(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
      scf.yield %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
    } else {
      %rhs_zero = arith.cmpi eq, %rhs_sign, %zero : i64
      %inner:3 = scf.if %rhs_zero -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) {
        %h, %m, %d = func.call @__ly_long_copy(%lhs_meta, %lhs_digits) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
        scf.yield %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
      } else {
        %same_sign = arith.cmpi eq, %lhs_sign, %rhs_sign : i64
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
              %h, %m, %d = func.call @__ly_long_sub_abs(%rhs_sign, %rhs_meta, %rhs_digits, %lhs_meta, %lhs_digits) : (i64, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
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

  func.func @LyLong_Sub(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta: memref<2xi64>, %lhs_digits: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta_raw: memref<2xi64>, %rhs_digits_raw: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0]} {
    %tag_one = arith.constant 1 : i64
    %lhs_ptr = memref.extract_aligned_pointer_as_index %lhs_header : memref<2xi64> -> index
    %lhs_bits = arith.index_cast %lhs_ptr : index to i64
    %lhs_tagbit = arith.andi %lhs_bits, %tag_one : i64
    %lhs_tagged = arith.cmpi eq, %lhs_tagbit, %tag_one : i64
    %rhs_ptr = memref.extract_aligned_pointer_as_index %rhs_header : memref<2xi64> -> index
    %rhs_bits = arith.index_cast %rhs_ptr : index to i64
    %rhs_tagbit = arith.andi %rhs_bits, %tag_one : i64
    %rhs_tagged = arith.cmpi eq, %rhs_tagbit, %tag_one : i64
    %both_tagged = arith.andi %lhs_tagged, %rhs_tagged : i1
    %lhs_scratch_meta = memref.alloca() : memref<2xi64>
    %lhs_scratch_digits = memref.alloca() : memref<3xi32>
    %rhs_scratch_meta = memref.alloca() : memref<2xi64>
    %rhs_scratch_digits = memref.alloca() : memref<3xi32>
    cf.cond_br %both_tagged, ^fast, ^general

  ^fast:
    %fast_a = arith.shrsi %lhs_bits, %tag_one : i64
    %fast_b = arith.shrsi %rhs_bits, %tag_one : i64
    %fast_diff = arith.subi %fast_a, %fast_b : i64
    %fh, %fm, %fd = func.call @LyLong_FromI64(%fast_diff) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %fh, %fm, %fd : memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^general:
    %lhs_meta_view, %lhs_digits_view = func.call @__ly_long_operand_view(%lhs_header, %lhs_meta, %lhs_digits, %lhs_scratch_meta, %lhs_scratch_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<3xi32>) -> (memref<2xi64>, memref<?xi32>)
    %rhs_meta, %rhs_digits = func.call @__ly_long_operand_view(%rhs_header, %rhs_meta_raw, %rhs_digits_raw, %rhs_scratch_meta, %rhs_scratch_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<3xi32>) -> (memref<2xi64>, memref<?xi32>)
    %small_two = arith.constant 2 : i64
    %small_count_slot = arith.constant 1 : index
    %lhs_view_count = memref.load %lhs_meta_view[%small_count_slot] : memref<2xi64>
    %rhs_view_count = memref.load %rhs_meta[%small_count_slot] : memref<2xi64>
    %lhs_small = arith.cmpi sle, %lhs_view_count, %small_two : i64
    %rhs_small = arith.cmpi sle, %rhs_view_count, %small_two : i64
    %both_small = arith.andi %lhs_small, %rhs_small : i1
    cf.cond_br %both_small, ^small, ^digits

  ^small:
    %small_a = func.call @__ly_long_view_as_i64(%lhs_meta_view, %lhs_digits_view) : (memref<2xi64>, memref<?xi32>) -> i64
    %small_b = func.call @__ly_long_view_as_i64(%rhs_meta, %rhs_digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %small_diff = arith.subi %small_a, %small_b : i64
    %sh, %sm, %sd = func.call @LyLong_FromI64(%small_diff) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %sh, %sm, %sd : memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^digits:
    %zero = arith.constant 0 : i64
    %neg_one = arith.constant -1 : i64
    %sign_slot = arith.constant 0 : index
    %rhs_sign = memref.load %rhs_meta[%sign_slot] : memref<2xi64>
    %rhs_neg = arith.muli %rhs_sign, %neg_one : i64
    %h, %m, %d = func.call @__ly_long_copy_with_sign(%rhs_neg, %rhs_meta, %rhs_digits) : (i64, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    %result_h, %result_m, %result_d = func.call @LyLong_Add(%lhs_header, %lhs_meta, %lhs_digits, %h, %m, %d) : (memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.call @LyLong_DecRef(%h, %m, %d) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> ()
    func.return %result_h, %result_m, %result_d : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyLong_Mul(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta_raw: memref<2xi64>, %lhs_digits_raw: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta_raw: memref<2xi64>, %rhs_digits_raw: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0]} {
    %tag_one = arith.constant 1 : i64
    %lhs_ptr = memref.extract_aligned_pointer_as_index %lhs_header : memref<2xi64> -> index
    %lhs_bits = arith.index_cast %lhs_ptr : index to i64
    %lhs_tagbit = arith.andi %lhs_bits, %tag_one : i64
    %lhs_tagged = arith.cmpi eq, %lhs_tagbit, %tag_one : i64
    %rhs_ptr = memref.extract_aligned_pointer_as_index %rhs_header : memref<2xi64> -> index
    %rhs_bits = arith.index_cast %rhs_ptr : index to i64
    %rhs_tagbit = arith.andi %rhs_bits, %tag_one : i64
    %rhs_tagged = arith.cmpi eq, %rhs_tagbit, %tag_one : i64
    %both_tagged = arith.andi %lhs_tagged, %rhs_tagged : i1
    %lhs_scratch_meta = memref.alloca() : memref<2xi64>
    %lhs_scratch_digits = memref.alloca() : memref<3xi32>
    %rhs_scratch_meta = memref.alloca() : memref<2xi64>
    %rhs_scratch_digits = memref.alloca() : memref<3xi32>
    cf.cond_br %both_tagged, ^fast, ^general

  ^fast:
    %fast_a = arith.shrsi %lhs_bits, %tag_one : i64
    %fast_b = arith.shrsi %rhs_bits, %tag_one : i64
    %fast_low, %fast_high = arith.mulsi_extended %fast_a, %fast_b : i64
    %c63 = arith.constant 63 : i64
    %fast_sign_ext = arith.shrsi %fast_low, %c63 : i64
    %fits_i64 = arith.cmpi eq, %fast_high, %fast_sign_ext : i64
    cf.cond_br %fits_i64, ^fast_fit, ^general

  ^fast_fit:
    %fh, %fm, %fd = func.call @LyLong_FromI64(%fast_low) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %fh, %fm, %fd : memref<2xi64>, memref<2xi64>, memref<?xi32>

  ^general:
    %lhs_meta, %lhs_digits = func.call @__ly_long_operand_view(%lhs_header, %lhs_meta_raw, %lhs_digits_raw, %lhs_scratch_meta, %lhs_scratch_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<3xi32>) -> (memref<2xi64>, memref<?xi32>)
    %rhs_meta, %rhs_digits = func.call @__ly_long_operand_view(%rhs_header, %rhs_meta_raw, %rhs_digits_raw, %rhs_scratch_meta, %rhs_scratch_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<3xi32>) -> (memref<2xi64>, memref<?xi32>)
    %small_two = arith.constant 2 : i64
    %small_count_slot = arith.constant 1 : index
    %lhs_view_count = memref.load %lhs_meta[%small_count_slot] : memref<2xi64>
    %rhs_view_count = memref.load %rhs_meta[%small_count_slot] : memref<2xi64>
    %lhs_small = arith.cmpi sle, %lhs_view_count, %small_two : i64
    %rhs_small = arith.cmpi sle, %rhs_view_count, %small_two : i64
    %both_small = arith.andi %lhs_small, %rhs_small : i1
    cf.cond_br %both_small, ^small, ^digits

  ^small:
    // At most two 30-bit limbs per side: compute natively and re-enter the
    // tagged domain through FromI64. This absorbs heap-immortal literals
    // into tagged values after a single operation.
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

  func.func @LyLong_Compare(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_meta_raw: memref<2xi64>, %lhs_digits_raw: memref<?xi32>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_meta_raw: memref<2xi64>, %rhs_digits_raw: memref<?xi32>) -> i64 {
    %tag_one = arith.constant 1 : i64
    %lhs_ptr = memref.extract_aligned_pointer_as_index %lhs_header : memref<2xi64> -> index
    %lhs_bits = arith.index_cast %lhs_ptr : index to i64
    %lhs_tagbit = arith.andi %lhs_bits, %tag_one : i64
    %lhs_tagged = arith.cmpi eq, %lhs_tagbit, %tag_one : i64
    %rhs_ptr = memref.extract_aligned_pointer_as_index %rhs_header : memref<2xi64> -> index
    %rhs_bits = arith.index_cast %rhs_ptr : index to i64
    %rhs_tagbit = arith.andi %rhs_bits, %tag_one : i64
    %rhs_tagged = arith.cmpi eq, %rhs_tagbit, %tag_one : i64
    %both_tagged = arith.andi %lhs_tagged, %rhs_tagged : i1
    %lhs_scratch_meta = memref.alloca() : memref<2xi64>
    %lhs_scratch_digits = memref.alloca() : memref<3xi32>
    %rhs_scratch_meta = memref.alloca() : memref<2xi64>
    %rhs_scratch_digits = memref.alloca() : memref<3xi32>
    cf.cond_br %both_tagged, ^fast, ^general

  ^fast:
    %fast_a = arith.shrsi %lhs_bits, %tag_one : i64
    %fast_b = arith.shrsi %rhs_bits, %tag_one : i64
    %fast_zero = arith.constant 0 : i64
    %fast_one = arith.constant 1 : i64
    %fast_neg_one = arith.constant -1 : i64
    %fast_gt = arith.cmpi sgt, %fast_a, %fast_b : i64
    %fast_lt = arith.cmpi slt, %fast_a, %fast_b : i64
    %fast_pos = arith.select %fast_gt, %fast_one, %fast_zero : i1, i64
    %fast_cmp = arith.select %fast_lt, %fast_neg_one, %fast_pos : i1, i64
    func.return %fast_cmp : i64

  ^general:
    %lhs_meta, %lhs_digits = func.call @__ly_long_operand_view(%lhs_header, %lhs_meta_raw, %lhs_digits_raw, %lhs_scratch_meta, %lhs_scratch_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<3xi32>) -> (memref<2xi64>, memref<?xi32>)
    %rhs_meta, %rhs_digits = func.call @__ly_long_operand_view(%rhs_header, %rhs_meta_raw, %rhs_digits_raw, %rhs_scratch_meta, %rhs_scratch_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<3xi32>) -> (memref<2xi64>, memref<?xi32>)
    %small_two = arith.constant 2 : i64
    %small_count_slot = arith.constant 1 : index
    %lhs_view_count = memref.load %lhs_meta[%small_count_slot] : memref<2xi64>
    %rhs_view_count = memref.load %rhs_meta[%small_count_slot] : memref<2xi64>
    %lhs_small = arith.cmpi sle, %lhs_view_count, %small_two : i64
    %rhs_small = arith.cmpi sle, %rhs_view_count, %small_two : i64
    %both_small = arith.andi %lhs_small, %rhs_small : i1
    cf.cond_br %both_small, ^small, ^digits

  ^small:
    // At most two 30-bit limbs per side: compute natively and re-enter the
    // tagged domain through FromI64. This absorbs heap-immortal literals
    // into tagged values after a single operation.
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

  func.func @LyLong_Repr(%header: memref<2xi64> {ly.ownership.object_header}, %meta_raw: memref<2xi64>, %digits_raw: memref<?xi32>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0]} {
    %repr_scratch_meta = memref.alloca() : memref<2xi64>
    %repr_scratch_digits = memref.alloca() : memref<3xi32>
    %meta, %digits = func.call @__ly_long_operand_view(%header, %meta_raw, %digits_raw, %repr_scratch_meta, %repr_scratch_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>, memref<2xi64>, memref<3xi32>) -> (memref<2xi64>, memref<?xi32>)
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
