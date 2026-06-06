module {
  func.func @LyLong_FromI64(%value: i64) -> (memref<2xi64>, memref<2xi8>, memref<3xi32>) attributes {ly.ownership.owned_results = [0]} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %layout_int = arith.constant 1 : i64
    %is_zero = arith.cmpi eq, %value, %zero : i64
    %is_negative = arith.cmpi slt, %value, %zero : i64
    %negative_one = arith.constant -1 : i64
    %positive_one = arith.constant 1 : i64
    %signed_sign = arith.select %is_negative, %negative_one, %positive_one : i64
    %sign = arith.select %is_zero, %zero, %signed_sign : i64
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
    %digit0 = arith.trunci %digit0_i64 : i64 to i32
    %digit1 = arith.trunci %digit1_i64 : i64 to i32
    %digit2 = arith.trunci %digit2_i64 : i64 to i32
    %one_digit = arith.constant 1 : i64
    %two_digits = arith.constant 2 : i64
    %three_digits = arith.constant 3 : i64
    %digit1_nonzero = arith.cmpi ne, %digit1_i64, %zero : i64
    %digit2_nonzero = arith.cmpi ne, %digit2_i64, %zero : i64
    %one_or_two = arith.select %digit1_nonzero, %two_digits, %one_digit : i64
    %nonzero_digits = arith.select %digit2_nonzero, %three_digits, %one_or_two : i64
    %ndigits = arith.select %is_zero, %zero, %nonzero_digits : i64
    %header = memref.alloc() {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<2xi64>
    %meta = memref.alloc() : memref<2xi8>
    %digits = memref.alloc() : memref<3xi32>
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    memref.store %one, %header[%refcount_slot] : memref<2xi64>
    memref.store %layout_int, %header[%layout_slot] : memref<2xi64>
    %sign_slot = arith.constant 0 : index
    %digit_count_slot = arith.constant 1 : index
    %digit0_slot = arith.constant 0 : index
    %digit1_slot = arith.constant 1 : index
    %digit2_slot = arith.constant 2 : index
    %sign_i8 = arith.trunci %sign : i64 to i8
    %ndigits_i8 = arith.trunci %ndigits : i64 to i8
    memref.store %sign_i8, %meta[%sign_slot] : memref<2xi8>
    memref.store %ndigits_i8, %meta[%digit_count_slot] : memref<2xi8>
    memref.store %digit0, %digits[%digit0_slot] : memref<3xi32>
    memref.store %digit1, %digits[%digit1_slot] : memref<3xi32>
    memref.store %digit2, %digits[%digit2_slot] : memref<3xi32>
    func.return %header, %meta, %digits : memref<2xi64>, memref<2xi8>, memref<3xi32>
  }

  func.func @LyLong_AsI64(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi8>, %digits: memref<3xi32>) -> i64 {
    %sign_slot = arith.constant 0 : index
    %digit_count_slot = arith.constant 1 : index
    %sign_i8 = memref.load %meta[%sign_slot] : memref<2xi8>
    %ndigits_i8 = memref.load %meta[%digit_count_slot] : memref<2xi8>
    %sign = arith.extsi %sign_i8 : i8 to i64
    %ndigits = arith.extui %ndigits_i8 : i8 to i64
    %digit0_index = arith.constant 0 : index
    %digit1_index = arith.constant 1 : index
    %digit2_index = arith.constant 2 : index
    %digit0_i32 = memref.load %digits[%digit0_index] : memref<3xi32>
    %digit1_i32 = memref.load %digits[%digit1_index] : memref<3xi32>
    %digit2_i32 = memref.load %digits[%digit2_index] : memref<3xi32>
    %digit0 = arith.extui %digit0_i32 : i32 to i64
    %digit1 = arith.extui %digit1_i32 : i32 to i64
    %digit2 = arith.extui %digit2_i32 : i32 to i64
    %zero = arith.constant 0 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %shift30 = arith.constant 30 : i64
    %shift60 = arith.constant 60 : i64
    %has_digit1 = arith.cmpi uge, %ndigits, %two : i64
    %has_digit2 = arith.cmpi uge, %ndigits, %three : i64
    %digit1_shifted = arith.shli %digit1, %shift30 : i64
    %with_digit1_full = arith.ori %digit0, %digit1_shifted : i64
    %with_digit1 = arith.select %has_digit1, %with_digit1_full, %digit0 : i64
    %digit2_shifted = arith.shli %digit2, %shift60 : i64
    %magnitude_full = arith.ori %with_digit1, %digit2_shifted : i64
    %magnitude = arith.select %has_digit2, %magnitude_full, %with_digit1 : i64
    %negated = arith.subi %zero, %magnitude : i64
    %is_negative = arith.cmpi slt, %sign, %zero : i64
    %result = arith.select %is_negative, %negated, %magnitude : i64
    func.return %result : i64
  }
}
