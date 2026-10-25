// Float to decimal digits -- what CPython's Python/dtoa.c (_Py_dg_dtoa) does,
// done here by a Steele-White / Burger-Dybvig digit generator over small
// bignums (`__ly_d4_*`): the shortest round-tripping digits (repr) and a
// counted number of digits (format precision, round).

module {
  // ===== impls: float-to-decimal (Dragon4) =====
  // Big scratch integers for the Steele-White/Burger-Dybvig digit generator:
  // little-endian base-10^9 limbs in i32 buffers, the live limb count travels
  // in registers, zero is length 0. Decimal limbs (not binary) because the
  // generator's only expensive steps are *10 scaling and radix-10 digit
  // extraction, which decimal limbs make carry-free scans.

  func.func private @__ly_d4_mul_small(%buf: memref<?xi32>, %len: i64, %m: i64) -> i64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %base = arith.constant 1000000000 : i64
    %len_idx = arith.index_cast %len : i64 to index
    %carry_final = scf.for %i = %c0 to %len_idx step %c1 iter_args(%carry = %zero) -> (i64) {
      %limb32 = memref.load %buf[%i] : memref<?xi32>
      %limb = arith.extui %limb32 : i32 to i64
      %prod = arith.muli %limb, %m : i64
      %acc = arith.addi %prod, %carry : i64
      %digit = arith.remui %acc, %base : i64
      %ncarry = arith.divui %acc, %base : i64
      %digit32 = arith.trunci %digit : i64 to i32
      memref.store %digit32, %buf[%i] : memref<?xi32>
      scf.yield %ncarry : i64
    }
    %res:2 = scf.while (%carry = %carry_final, %l = %len) : (i64, i64) -> (i64, i64) {
      %more = arith.cmpi ne, %carry, %zero : i64
      scf.condition(%more) %carry, %l : i64, i64
    } do {
    ^bb0(%carry: i64, %l: i64):
      %pos = arith.index_cast %l : i64 to index
      %digit = arith.remui %carry, %base : i64
      %ncarry = arith.divui %carry, %base : i64
      %digit32 = arith.trunci %digit : i64 to i32
      memref.store %digit32, %buf[%pos] : memref<?xi32>
      %nl = arith.addi %l, %one : i64
      scf.yield %ncarry, %nl : i64, i64
    }
    func.return %res#1 : i64
  }

  func.func private @__ly_d4_mul_pow2(%buf: memref<?xi32>, %len: i64, %k: i64) -> i64 {
    %zero = arith.constant 0 : i64
    %chunk = arith.constant 29 : i64
    %one = arith.constant 1 : i64
    %res:2 = scf.while (%remaining = %k, %l = %len) : (i64, i64) -> (i64, i64) {
      %more = arith.cmpi sgt, %remaining, %zero : i64
      scf.condition(%more) %remaining, %l : i64, i64
    } do {
    ^bb0(%remaining: i64, %l: i64):
      %step = arith.minsi %remaining, %chunk : i64
      %factor = arith.shli %one, %step : i64
      %nl = func.call @__ly_d4_mul_small(%buf, %l, %factor) : (memref<?xi32>, i64, i64) -> i64
      %nrem = arith.subi %remaining, %step : i64
      scf.yield %nrem, %nl : i64, i64
    }
    func.return %res#1 : i64
  }

  func.func private @__ly_d4_mul_pow10(%buf: memref<?xi32>, %len: i64, %k: i64) -> i64 {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %nine = arith.constant 9 : i64
    %ten = arith.constant 10 : i64
    %base = arith.constant 1000000000 : i64
    %big:2 = scf.while (%remaining = %k, %l = %len) : (i64, i64) -> (i64, i64) {
      %more = arith.cmpi sge, %remaining, %nine : i64
      scf.condition(%more) %remaining, %l : i64, i64
    } do {
    ^bb0(%remaining: i64, %l: i64):
      %nl = func.call @__ly_d4_mul_small(%buf, %l, %base) : (memref<?xi32>, i64, i64) -> i64
      %nrem = arith.subi %remaining, %nine : i64
      scf.yield %nrem, %nl : i64, i64
    }
    %fin:2 = scf.while (%remaining = %big#0, %l = %big#1) : (i64, i64) -> (i64, i64) {
      %more = arith.cmpi sgt, %remaining, %zero : i64
      scf.condition(%more) %remaining, %l : i64, i64
    } do {
    ^bb0(%remaining: i64, %l: i64):
      %nl = func.call @__ly_d4_mul_small(%buf, %l, %ten) : (memref<?xi32>, i64, i64) -> i64
      %nrem = arith.subi %remaining, %one : i64
      scf.yield %nrem, %nl : i64, i64
    }
    func.return %fin#1 : i64
  }

  func.func private @__ly_d4_cmp(%a: memref<?xi32>, %alen: i64, %b: memref<?xi32>, %blen: i64) -> i64 {
    %zero = arith.constant 0 : i64
    %minus = arith.constant -1 : i64
    %plus = arith.constant 1 : i64
    %len_ne = arith.cmpi ne, %alen, %blen : i64
    %len_lt = arith.cmpi slt, %alen, %blen : i64
    %len_result = arith.select %len_lt, %minus, %plus : i64
    %result = scf.if %len_ne -> (i64) {
      scf.yield %len_result : i64
    } else {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %n = arith.index_cast %alen : i64 to index
      %scan:2 = scf.while (%acc = %zero, %i = %c0) : (i64, index) -> (i64, index) {
        %in_range = arith.cmpi slt, %i, %n : index
        %decided = arith.cmpi ne, %acc, %zero : i64
        %open_true = arith.constant true
        %open = arith.xori %decided, %open_true : i1
        %go = arith.andi %in_range, %open : i1
        scf.condition(%go) %acc, %i : i64, index
      } do {
      ^bb0(%acc: i64, %i: index):
        %ip1 = arith.addi %i, %c1 : index
        %rev = arith.subi %n, %ip1 : index
        %av32 = memref.load %a[%rev] : memref<?xi32>
        %bv32 = memref.load %b[%rev] : memref<?xi32>
        %av = arith.extui %av32 : i32 to i64
        %bv = arith.extui %bv32 : i32 to i64
        %lt = arith.cmpi ult, %av, %bv : i64
        %gt = arith.cmpi ugt, %av, %bv : i64
        %gtv = arith.select %gt, %plus, %zero : i64
        %this = arith.select %lt, %minus, %gtv : i64
        %iv_next = arith.addi %i, %c1 : index
        scf.yield %this, %iv_next : i64, index
      }
      scf.yield %scan#0 : i64
    }
    func.return %result : i64
  }

  // a -= b; requires a >= b. Returns the normalized length.
  func.func private @__ly_d4_sub_inplace(%a: memref<?xi32>, %alen: i64, %b: memref<?xi32>, %blen: i64) -> i64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %base = arith.constant 1000000000 : i64
    %z32 = arith.constant 0 : i32
    %n = arith.index_cast %alen : i64 to index
    %blen_idx = arith.index_cast %blen : i64 to index
    %borrow_final = scf.for %i = %c0 to %n step %c1 iter_args(%borrow = %zero) -> (i64) {
      %av32 = memref.load %a[%i] : memref<?xi32>
      %av = arith.extui %av32 : i32 to i64
      %inb = arith.cmpi ult, %i, %blen_idx : index
      %bv = scf.if %inb -> (i64) {
        %bv32 = memref.load %b[%i] : memref<?xi32>
        %bvv = arith.extui %bv32 : i32 to i64
        scf.yield %bvv : i64
      } else {
        scf.yield %zero : i64
      }
      %sub = arith.subi %av, %bv : i64
      %d0 = arith.subi %sub, %borrow : i64
      %negd = arith.cmpi slt, %d0, %zero : i64
      %dfix = arith.addi %d0, %base : i64
      %d = arith.select %negd, %dfix, %d0 : i64
      %nborrow = arith.select %negd, %one, %zero : i64
      %d32 = arith.trunci %d : i64 to i32
      memref.store %d32, %a[%i] : memref<?xi32>
      scf.yield %nborrow : i64
    }
    %newlen = scf.while (%l = %alen) : (i64) -> i64 {
      %pos = arith.cmpi sgt, %l, %zero : i64
      %top_zero = scf.if %pos -> (i1) {
        %lm1 = arith.subi %l, %one : i64
        %idx = arith.index_cast %lm1 : i64 to index
        %v = memref.load %a[%idx] : memref<?xi32>
        %is0 = arith.cmpi eq, %v, %z32 : i32
        scf.yield %is0 : i1
      } else {
        %false = arith.constant false
        scf.yield %false : i1
      }
      scf.condition(%top_zero) %l : i64
    } do {
    ^bb0(%l: i64):
      %lm1 = arith.subi %l, %one : i64
      scf.yield %lm1 : i64
    }
    func.return %newlen : i64
  }

  func.func private @__ly_d4_copy(%src: memref<?xi32>, %len: i64, %dst: memref<?xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %n = arith.index_cast %len : i64 to index
    scf.for %i = %c0 to %n step %c1 {
      %v = memref.load %src[%i] : memref<?xi32>
      memref.store %v, %dst[%i] : memref<?xi32>
    }
    func.return
  }

  // a += b in place; a's buffer must have room for one extra limb.
  func.func private @__ly_d4_add_inplace(%a: memref<?xi32>, %alen: i64, %b: memref<?xi32>, %blen: i64) -> i64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %base = arith.constant 1000000000 : i64
    %maxlen = arith.maxsi %alen, %blen : i64
    %n = arith.index_cast %maxlen : i64 to index
    %alen_idx = arith.index_cast %alen : i64 to index
    %blen_idx = arith.index_cast %blen : i64 to index
    %carry_final = scf.for %i = %c0 to %n step %c1 iter_args(%carry = %zero) -> (i64) {
      %ina = arith.cmpi ult, %i, %alen_idx : index
      %av = scf.if %ina -> (i64) {
        %av32 = memref.load %a[%i] : memref<?xi32>
        %avv = arith.extui %av32 : i32 to i64
        scf.yield %avv : i64
      } else {
        scf.yield %zero : i64
      }
      %inb = arith.cmpi ult, %i, %blen_idx : index
      %bv = scf.if %inb -> (i64) {
        %bv32 = memref.load %b[%i] : memref<?xi32>
        %bvv = arith.extui %bv32 : i32 to i64
        scf.yield %bvv : i64
      } else {
        scf.yield %zero : i64
      }
      %s0 = arith.addi %av, %bv : i64
      %s = arith.addi %s0, %carry : i64
      %overflow = arith.cmpi sge, %s, %base : i64
      %sfix = arith.subi %s, %base : i64
      %d = arith.select %overflow, %sfix, %s : i64
      %ncarry = arith.select %overflow, %one, %zero : i64
      %d32 = arith.trunci %d : i64 to i32
      memref.store %d32, %a[%i] : memref<?xi32>
      scf.yield %ncarry : i64
    }
    %has_carry = arith.cmpi ne, %carry_final, %zero : i64
    %newlen = scf.if %has_carry -> (i64) {
      %one32 = arith.constant 1 : i32
      memref.store %one32, %a[%n] : memref<?xi32>
      %nl = arith.addi %maxlen, %one : i64
      scf.yield %nl : i64
    } else {
      scf.yield %maxlen : i64
    }
    func.return %newlen : i64
  }

  func.func private @__ly_d4_from_u64(%buf: memref<?xi32>, %v: i64) -> i64 {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %base = arith.constant 1000000000 : i64
    %res:2 = scf.while (%rest = %v, %l = %zero) : (i64, i64) -> (i64, i64) {
      %more = arith.cmpi ne, %rest, %zero : i64
      scf.condition(%more) %rest, %l : i64, i64
    } do {
    ^bb0(%rest: i64, %l: i64):
      %pos = arith.index_cast %l : i64 to index
      %digit = arith.remui %rest, %base : i64
      %digit32 = arith.trunci %digit : i64 to i32
      memref.store %digit32, %buf[%pos] : memref<?xi32>
      %nrest = arith.divui %rest, %base : i64
      %nl = arith.addi %l, %one : i64
      scf.yield %nrest, %nl : i64, i64
    }
    func.return %res#1 : i64
  }

  // The Burger-Dybvig "high" stopping test: scratch := (r + mplus) [* 10],
  // compared against s. Inclusive (>=) when the mantissa is even, because
  // round-to-nearest-even lets the boundary itself round back.
  func.func private @__ly_d4_high(%r: memref<?xi32>, %rlen: i64, %mp: memref<?xi32>, %mplen: i64, %s: memref<?xi32>, %slen: i64, %scratch: memref<?xi32>, %even: i1, %extra10: i1) -> i1 {
    %zero = arith.constant 0 : i64
    %ten = arith.constant 10 : i64
    func.call @__ly_d4_copy(%r, %rlen, %scratch) : (memref<?xi32>, i64, memref<?xi32>) -> ()
    %tlen0 = func.call @__ly_d4_add_inplace(%scratch, %rlen, %mp, %mplen) : (memref<?xi32>, i64, memref<?xi32>, i64) -> i64
    %tlen = scf.if %extra10 -> (i64) {
      %scaled = func.call @__ly_d4_mul_small(%scratch, %tlen0, %ten) : (memref<?xi32>, i64, i64) -> i64
      scf.yield %scaled : i64
    } else {
      scf.yield %tlen0 : i64
    }
    %cmp = func.call @__ly_d4_cmp(%scratch, %tlen, %s, %slen) : (memref<?xi32>, i64, memref<?xi32>, i64) -> i64
    %ge = arith.cmpi sge, %cmp, %zero : i64
    %gt = arith.cmpi sgt, %cmp, %zero : i64
    %result = arith.select %even, %ge, %gt : i1
    func.return %result : i1
  }

  // Increment the ASCII digit string's last digit, propagating the carry.
  // Returns (new count, decimal-point bump). Positions that carried out
  // became '0' and are trimmed; an all-nines overflow leaves the single
  // digit "1" one decimal place higher.
  func.func private @__ly_d4_round_up(%digits: memref<?xi8>, %count: i64) -> (i64, i64) {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %nine_ch = arith.constant 57 : i8
    %zero_ch = arith.constant 48 : i8
    %one_ch = arith.constant 49 : i8
    %one_i8 = arith.constant 1 : i8
    %c0 = arith.constant 0 : index
    %j_final = scf.while (%j = %count) : (i64) -> i64 {
      // Loop while there is a position left and it holds '9'.
      %jm1 = arith.subi %j, %one : i64
      %has = arith.cmpi sgt, %j, %zero : i64
      %is_nine = scf.if %has -> (i1) {
        %idx = arith.index_cast %jm1 : i64 to index
        %ch = memref.load %digits[%idx] : memref<?xi8>
        %nine = arith.cmpi eq, %ch, %nine_ch : i8
        scf.yield %nine : i1
      } else {
        %false = arith.constant false
        scf.yield %false : i1
      }
      scf.condition(%is_nine) %j : i64
    } do {
    ^bb0(%j: i64):
      %jm1 = arith.subi %j, %one : i64
      %idx = arith.index_cast %jm1 : i64 to index
      memref.store %zero_ch, %digits[%idx] : memref<?xi8>
      scf.yield %jm1 : i64
    }
    %overflowed = arith.cmpi eq, %j_final, %zero : i64
    %res:2 = scf.if %overflowed -> (i64, i64) {
      memref.store %one_ch, %digits[%c0] : memref<?xi8>
      scf.yield %one, %one : i64, i64
    } else {
      %jm1 = arith.subi %j_final, %one : i64
      %idx = arith.index_cast %jm1 : i64 to index
      %ch = memref.load %digits[%idx] : memref<?xi8>
      %chp1 = arith.addi %ch, %one_i8 : i8
      memref.store %chp1, %digits[%idx] : memref<?xi8>
      scf.yield %j_final, %zero : i64, i64
    }
    func.return %res#0, %res#1 : i64, i64
  }

  // Shortest round-trip digits for a positive finite double (Burger-Dybvig
  // with exact big-integer state; the log10 estimate is repaired by the two
  // fixup loops, so estimate error only costs an extra *10 pass). Returns
  // (digit count, decimal exponent k): value = 0.d1d2...dn * 10^k.
  func.func private @__ly_dtoa_shortest(%abs: f64, %digits_out: memref<?xi8>) -> (i64, i64) {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %four = arith.constant 4 : i64
    %ten = arith.constant 10 : i64
    %ascii_zero = arith.constant 48 : i64
    %false = arith.constant false
    %true = arith.constant true

    %bits = arith.bitcast %abs : f64 to i64
    %mant_mask = arith.constant 4503599627370495 : i64
    %mant = arith.andi %bits, %mant_mask : i64
    %c52 = arith.constant 52 : i64
    %shifted = arith.shrui %bits, %c52 : i64
    %exp_mask = arith.constant 2047 : i64
    %biased = arith.andi %shifted, %exp_mask : i64
    %is_denormal = arith.cmpi eq, %biased, %zero : i64
    %implicit = arith.constant 4503599627370496 : i64
    %fnorm = arith.addi %mant, %implicit : i64
    %f = arith.select %is_denormal, %mant, %fnorm : i64
    %c1075 = arith.constant 1075 : i64
    %enorm = arith.subi %biased, %c1075 : i64
    %c_min_e = arith.constant -1074 : i64
    %e = arith.select %is_denormal, %c_min_e, %enorm : i64
    %f_low_bit = arith.andi %f, %one : i64
    %even = arith.cmpi eq, %f_low_bit, %zero : i64
    %mant_zero = arith.cmpi eq, %mant, %zero : i64
    %biased_gt1 = arith.cmpi sgt, %biased, %one : i64
    %boundary = arith.andi %mant_zero, %biased_gt1 : i1

    %r_store = memref.alloca() : memref<48xi32>
    %s_store = memref.alloca() : memref<48xi32>
    %mp_store = memref.alloca() : memref<48xi32>
    %mm_store = memref.alloca() : memref<48xi32>
    %t_store = memref.alloca() : memref<48xi32>
    %r = memref.cast %r_store : memref<48xi32> to memref<?xi32>
    %s = memref.cast %s_store : memref<48xi32> to memref<?xi32>
    %mp = memref.cast %mp_store : memref<48xi32> to memref<?xi32>
    %mm = memref.cast %mm_store : memref<48xi32> to memref<?xi32>
    %t = memref.cast %t_store : memref<48xi32> to memref<?xi32>

    %e_neg = arith.cmpi slt, %e, %zero : i64
    %lens:4 = scf.if %e_neg -> (i64, i64, i64, i64) {
      %rl0 = func.call @__ly_d4_from_u64(%r, %f) : (memref<?xi32>, i64) -> i64
      %rshift = arith.select %boundary, %two, %one : i64
      %rl = func.call @__ly_d4_mul_pow2(%r, %rl0, %rshift) : (memref<?xi32>, i64, i64) -> i64
      %sl0 = func.call @__ly_d4_from_u64(%s, %one) : (memref<?xi32>, i64) -> i64
      %sbase = arith.select %boundary, %two, %one : i64
      %sexp = arith.subi %sbase, %e : i64
      %sl = func.call @__ly_d4_mul_pow2(%s, %sl0, %sexp) : (memref<?xi32>, i64, i64) -> i64
      %mpv = arith.select %boundary, %two, %one : i64
      %mpl = func.call @__ly_d4_from_u64(%mp, %mpv) : (memref<?xi32>, i64) -> i64
      %mml = func.call @__ly_d4_from_u64(%mm, %one) : (memref<?xi32>, i64) -> i64
      scf.yield %rl, %sl, %mpl, %mml : i64, i64, i64, i64
    } else {
      %rl0 = func.call @__ly_d4_from_u64(%r, %f) : (memref<?xi32>, i64) -> i64
      %rextra = arith.select %boundary, %two, %one : i64
      %rexp = arith.addi %e, %rextra : i64
      %rl = func.call @__ly_d4_mul_pow2(%r, %rl0, %rexp) : (memref<?xi32>, i64, i64) -> i64
      %sv = arith.select %boundary, %four, %two : i64
      %sl = func.call @__ly_d4_from_u64(%s, %sv) : (memref<?xi32>, i64) -> i64
      %mpl0 = func.call @__ly_d4_from_u64(%mp, %one) : (memref<?xi32>, i64) -> i64
      %mpextra = arith.select %boundary, %one, %zero : i64
      %mpexp = arith.addi %e, %mpextra : i64
      %mpl = func.call @__ly_d4_mul_pow2(%mp, %mpl0, %mpexp) : (memref<?xi32>, i64, i64) -> i64
      %mml0 = func.call @__ly_d4_from_u64(%mm, %one) : (memref<?xi32>, i64) -> i64
      %mml = func.call @__ly_d4_mul_pow2(%mm, %mml0, %e) : (memref<?xi32>, i64, i64) -> i64
      scf.yield %rl, %sl, %mpl, %mml : i64, i64, i64, i64
    }

    %l10 = math.log10 %abs : f64
    %flr = math.floor %l10 : f64
    %k0f = arith.fptosi %flr : f64 to i64
    %k0 = arith.addi %k0f, %one : i64
    %k0_neg = arith.cmpi slt, %k0, %zero : i64
    %lens2:4 = scf.if %k0_neg -> (i64, i64, i64, i64) {
      %n10 = arith.subi %zero, %k0 : i64
      %rl = func.call @__ly_d4_mul_pow10(%r, %lens#0, %n10) : (memref<?xi32>, i64, i64) -> i64
      %mpl = func.call @__ly_d4_mul_pow10(%mp, %lens#2, %n10) : (memref<?xi32>, i64, i64) -> i64
      %mml = func.call @__ly_d4_mul_pow10(%mm, %lens#3, %n10) : (memref<?xi32>, i64, i64) -> i64
      scf.yield %rl, %lens#1, %mpl, %mml : i64, i64, i64, i64
    } else {
      %sl = func.call @__ly_d4_mul_pow10(%s, %lens#1, %k0) : (memref<?xi32>, i64, i64) -> i64
      scf.yield %lens#0, %sl, %lens#2, %lens#3 : i64, i64, i64, i64
    }

    // Fixup 1: while the value's upper bound reaches 10^k, grow S.
    %fix1:2 = scf.while (%k = %k0, %sl = %lens2#1) : (i64, i64) -> (i64, i64) {
      %hi = func.call @__ly_d4_high(%r, %lens2#0, %mp, %lens2#2, %s, %sl, %t, %even, %false) : (memref<?xi32>, i64, memref<?xi32>, i64, memref<?xi32>, i64, memref<?xi32>, i1, i1) -> i1
      scf.condition(%hi) %k, %sl : i64, i64
    } do {
    ^bb0(%k: i64, %sl: i64):
      %nsl = func.call @__ly_d4_mul_small(%s, %sl, %ten) : (memref<?xi32>, i64, i64) -> i64
      %nk = arith.addi %k, %one : i64
      scf.yield %nk, %nsl : i64, i64
    }
    // Fixup 2: while even the *10-scaled upper bound stays under S, shrink k.
    %fix2:4 = scf.while (%k = %fix1#0, %rl = %lens2#0, %mpl = %lens2#2, %mml = %lens2#3) : (i64, i64, i64, i64) -> (i64, i64, i64, i64) {
      %hi = func.call @__ly_d4_high(%r, %rl, %mp, %mpl, %s, %fix1#1, %t, %even, %true) : (memref<?xi32>, i64, memref<?xi32>, i64, memref<?xi32>, i64, memref<?xi32>, i1, i1) -> i1
      %lo = arith.xori %hi, %true : i1
      scf.condition(%lo) %k, %rl, %mpl, %mml : i64, i64, i64, i64
    } do {
    ^bb0(%k: i64, %rl: i64, %mpl: i64, %mml: i64):
      %nrl = func.call @__ly_d4_mul_small(%r, %rl, %ten) : (memref<?xi32>, i64, i64) -> i64
      %nmpl = func.call @__ly_d4_mul_small(%mp, %mpl, %ten) : (memref<?xi32>, i64, i64) -> i64
      %nmml = func.call @__ly_d4_mul_small(%mm, %mml, %ten) : (memref<?xi32>, i64, i64) -> i64
      %nk = arith.subi %k, %one : i64
      scf.yield %nk, %nrl, %nmpl, %nmml : i64, i64, i64, i64
    }
    cf.br ^digit_loop(%fix2#1, %fix2#2, %fix2#3, %zero : i64, i64, i64, i64)

  ^digit_loop(%rl_in: i64, %mpl_in: i64, %mml_in: i64, %count: i64):
    %rl_a = func.call @__ly_d4_mul_small(%r, %rl_in, %ten) : (memref<?xi32>, i64, i64) -> i64
    %mpl_a = func.call @__ly_d4_mul_small(%mp, %mpl_in, %ten) : (memref<?xi32>, i64, i64) -> i64
    %mml_a = func.call @__ly_d4_mul_small(%mm, %mml_in, %ten) : (memref<?xi32>, i64, i64) -> i64
    %ext:2 = scf.while (%d = %zero, %rl = %rl_a) : (i64, i64) -> (i64, i64) {
      %cmp = func.call @__ly_d4_cmp(%r, %rl, %s, %fix1#1) : (memref<?xi32>, i64, memref<?xi32>, i64) -> i64
      %ge = arith.cmpi sge, %cmp, %zero : i64
      scf.condition(%ge) %d, %rl : i64, i64
    } do {
    ^bb0(%d: i64, %rl: i64):
      %nrl = func.call @__ly_d4_sub_inplace(%r, %rl, %s, %fix1#1) : (memref<?xi32>, i64, memref<?xi32>, i64) -> i64
      %nd = arith.addi %d, %one : i64
      scf.yield %nd, %nrl : i64, i64
    }
    %cl = func.call @__ly_d4_cmp(%r, %ext#1, %mm, %mml_a) : (memref<?xi32>, i64, memref<?xi32>, i64) -> i64
    %cl_le = arith.cmpi sle, %cl, %zero : i64
    %cl_lt = arith.cmpi slt, %cl, %zero : i64
    %low = arith.select %even, %cl_le, %cl_lt : i1
    %high = func.call @__ly_d4_high(%r, %ext#1, %mp, %mpl_a, %s, %fix1#1, %t, %even, %false) : (memref<?xi32>, i64, memref<?xi32>, i64, memref<?xi32>, i64, memref<?xi32>, i1, i1) -> i1
    %stop = arith.ori %low, %high : i1
    cf.cond_br %stop, ^finalize(%ext#0, %ext#1, %low, %high, %count : i64, i64, i1, i1, i64), ^emit(%ext#0, %ext#1, %mpl_a, %mml_a, %count : i64, i64, i64, i64, i64)

  ^emit(%d_emit: i64, %rl_emit: i64, %mpl_emit: i64, %mml_emit: i64, %count_emit: i64):
    %ch_i64 = arith.addi %d_emit, %ascii_zero : i64
    %ch = arith.trunci %ch_i64 : i64 to i8
    %pos = arith.index_cast %count_emit : i64 to index
    memref.store %ch, %digits_out[%pos] : memref<?xi8>
    %ncount = arith.addi %count_emit, %one : i64
    cf.br ^digit_loop(%rl_emit, %mpl_emit, %mml_emit, %ncount : i64, i64, i64, i64)

  ^finalize(%d_fin: i64, %rl_fin: i64, %low_fin: i1, %high_fin: i1, %count_fin: i64):
    // Store the candidate digit, then decide whether it rounds up: high-only
    // always rounds up; low&high picks the nearest side of the scaled value
    // (2R vs S), breaking the exact tie toward the even digit.
    %fin_ch_i64 = arith.addi %d_fin, %ascii_zero : i64
    %fin_ch = arith.trunci %fin_ch_i64 : i64 to i8
    %fin_pos = arith.index_cast %count_fin : i64 to index
    memref.store %fin_ch, %digits_out[%fin_pos] : memref<?xi8>
    %count1 = arith.addi %count_fin, %one : i64
    %both = arith.andi %low_fin, %high_fin : i1
    %roundup = scf.if %both -> (i1) {
      func.call @__ly_d4_copy(%r, %rl_fin, %t) : (memref<?xi32>, i64, memref<?xi32>) -> ()
      %tlen = func.call @__ly_d4_mul_small(%t, %rl_fin, %two) : (memref<?xi32>, i64, i64) -> i64
      %cmp2 = func.call @__ly_d4_cmp(%t, %tlen, %s, %fix1#1) : (memref<?xi32>, i64, memref<?xi32>, i64) -> i64
      %gt = arith.cmpi sgt, %cmp2, %zero : i64
      %eq = arith.cmpi eq, %cmp2, %zero : i64
      %d_odd_bit = arith.andi %d_fin, %one : i64
      %d_odd = arith.cmpi ne, %d_odd_bit, %zero : i64
      %tie_up = arith.andi %eq, %d_odd : i1
      %up = arith.ori %gt, %tie_up : i1
      scf.yield %up : i1
    } else {
      %not_low = arith.xori %low_fin, %true : i1
      %up = arith.andi %high_fin, %not_low : i1
      scf.yield %up : i1
    }
    %final:2 = scf.if %roundup -> (i64, i64) {
      %rounded:2 = func.call @__ly_d4_round_up(%digits_out, %count1) : (memref<?xi8>, i64) -> (i64, i64)
      %nk = arith.addi %fix2#0, %rounded#1 : i64
      scf.yield %rounded#0, %nk : i64, i64
    } else {
      scf.yield %count1, %fix2#0 : i64, i64
    }
    func.return %final#0, %final#1 : i64, i64
  }

  // CPython's tens[] and bigtens[] (Python/dtoa.c).
  memref.global "private" constant @__ly_dtoa_tens : memref<23xf64> = dense<[1.0e0, 1.0e1, 1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e7, 1.0e8, 1.0e9, 1.0e10, 1.0e11, 1.0e12, 1.0e13, 1.0e14, 1.0e15, 1.0e16, 1.0e17, 1.0e18, 1.0e19, 1.0e20, 1.0e21, 1.0e22]>
  memref.global "private" constant @__ly_dtoa_bigtens : memref<5xf64> = dense<[1.0e16, 1.0e32, 1.0e64, 1.0e128, 1.0e256]>

  func.func private @__ly_dtoa_ten_to(%k: i64) -> f64 {
    %table = memref.get_global @__ly_dtoa_tens : memref<23xf64>
    %index = arith.index_cast %k : i64 to index
    %value = memref.load %table[%index] : memref<23xf64>
    func.return %value : f64
  }

  // Trailing zeros of the digits [0, count) dropped.
  func.func private @__ly_dtoa_trim(%digits: memref<?xi8>, %count: i64) -> i64 {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %zero_ch = arith.constant 48 : i8
    %false = arith.constant false
    %trimmed = scf.while (%c = %count) : (i64) -> i64 {
      %has = arith.cmpi sgt, %c, %zero : i64
      %is_zero_ch = scf.if %has -> (i1) {
        %cm1 = arith.subi %c, %one : i64
        %idx = arith.index_cast %cm1 : i64 to index
        %ch = memref.load %digits[%idx] : memref<?xi8>
        %z = arith.cmpi eq, %ch, %zero_ch : i8
        scf.yield %z : i1
      } else {
        scf.yield %false : i1
      }
      scf.condition(%is_zero_ch) %c : i64
    } do {
    ^bb0(%c: i64):
      %cm1 = arith.subi %c, %one : i64
      scf.yield %cm1 : i64
    }
    func.return %trimmed : i64
  }

  // The two ways _Py_dg_dtoa answers modes 2 and 3 without bignums, ported:
  // floating-point digit generation with a running error bound when the
  // digit count is at most Quick_max (14), and exact double arithmetic for an
  // integer below 10^15 (Int_max). Answers (ok, digit count, decimal point)
  // in __ly_dtoa_counted's terms, ok false when neither applies or the error
  // bound cannot decide the last digit -- then the bignum path answers, as
  // CPython's does at its `fast_failed`. A positive finite `u`.
  // ⛔ Not log10 for the decimal exponent: libm's may round 999.9999999999
  // up to 3.0, and the floating-point path relies on CPython's estimate
  // never being too small.
  func.func private @__ly_dtoa_quick(%u: f64, %fixed_mode: i1, %ndigits: i64, %digits_out: memref<?xi8>) -> (i1, i64, i64) {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %ten = arith.constant 10 : i64
    %fifteen = arith.constant 15 : i64
    %sixteen = arith.constant 16 : i64
    %four = arith.constant 4 : i64
    %quick_max = arith.constant 14 : i64
    %ten_pmax = arith.constant 22 : i64
    %ascii_zero = arith.constant 48 : i64
    %false = arith.constant false
    %true = arith.constant true
    %zero_f = arith.constant 0.0 : f64
    %half_f = arith.constant 0.5 : f64
    %one_f = arith.constant 1.0 : f64
    %five_f = arith.constant 5.0 : f64
    %seven_f = arith.constant 7.0 : f64
    %ten_f = arith.constant 10.0 : f64
    %bits = arith.bitcast %u : f64 to i64
    %c52 = arith.constant 52 : i64
    %exp_mask = arith.constant 2047 : i64
    %shifted = arith.shrui %bits, %c52 : i64
    %biased = arith.andi %shifted, %exp_mask : i64
    %denormal = arith.cmpi eq, %biased, %zero : i64
    cf.cond_br %denormal, ^fail, ^estimate

  ^estimate:
    // k = floor(log10(u)), possibly one too large but never too small: the
    // first-order expansion of log10 about 1.5, with the exponent's share.
    %frac_mask = arith.constant 4503599627370495 : i64
    %one_exponent = arith.constant 4607182418800017408 : i64
    %frac = arith.andi %bits, %frac_mask : i64
    %d2_bits = arith.ori %frac, %one_exponent : i64
    %d2 = arith.bitcast %d2_bits : i64 to f64
    %bias = arith.constant 1023 : i64
    %i_exp = arith.subi %biased, %bias : i64
    %i_exp_f = arith.sitofp %i_exp : i64 to f64
    %one_half = arith.constant 1.5 : f64
    %slope = arith.constant 0.289529654602168 : f64
    %intercept = arith.constant 0.1760912590558 : f64
    %log10_2 = arith.constant 0.301029995663981 : f64
    %d2_off = arith.subf %d2, %one_half : f64
    %t0 = arith.mulf %d2_off, %slope : f64
    %t1 = arith.addf %t0, %intercept : f64
    %t2 = arith.mulf %i_exp_f, %log10_2 : f64
    %ds_est = arith.addf %t1, %t2 : f64
    %k_trunc = arith.fptosi %ds_est : f64 to i64
    %k_trunc_f = arith.sitofp %k_trunc : i64 to f64
    %ds_negative = arith.cmpf olt, %ds_est, %zero_f : f64
    %ds_fractional = arith.cmpf one, %ds_est, %k_trunc_f : f64
    %round_down = arith.andi %ds_negative, %ds_fractional : i1
    %k_dec = arith.subi %k_trunc, %one : i64
    %k_floor = arith.select %round_down, %k_dec, %k_trunc : i64
    %k_nonneg = arith.cmpi sge, %k_floor, %zero : i64
    %k_small = arith.cmpi sle, %k_floor, %ten_pmax : i64
    %in_table = arith.andi %k_nonneg, %k_small : i1
    %k = scf.if %in_table -> (i64) {
      %power = func.call @__ly_dtoa_ten_to(%k_floor) : (i64) -> f64
      %below = arith.cmpf olt, %u, %power : f64
      %k_less = arith.subi %k_floor, %one : i64
      %exact = arith.select %below, %k_less, %k_floor : i64
      scf.yield %exact : i64
    } else {
      scf.yield %k_floor : i64
    }
    %k_check = arith.xori %in_table, %true : i1
    // mode 2: max(1, ndigits) significant digits; mode 3: ndigits past the
    // point, so ndigits + k + 1 digits in all.
    %fixed_count = arith.addi %ndigits, %k : i64
    %fixed_ilim = arith.addi %fixed_count, %one : i64
    %sig = arith.maxsi %ndigits, %one : i64
    %ilim = arith.select %fixed_mode, %fixed_ilim, %sig : i64
    %ilim1 = arith.select %fixed_mode, %fixed_count, %sig : i64
    %ilim_nonneg = arith.cmpi sge, %ilim, %zero : i64
    %ilim_small = arith.cmpi sle, %ilim, %quick_max : i64
    %try_quick = arith.andi %ilim_nonneg, %ilim_small : i1
    cf.cond_br %try_quick, ^scale, ^small_integer

  ^scale:
    // u / 10^k, with ieps counting the roundings that took.
    %bigtens = memref.get_global @__ly_dtoa_bigtens : memref<5xf64>
    %k_positive = arith.cmpi sgt, %k, %zero : i64
    %k_negative = arith.cmpi slt, %k, %zero : i64
    %scaled:2 = scf.if %k_positive -> (f64, i64) {
      %low = arith.andi %k, %fifteen : i64
      %ds0 = func.call @__ly_dtoa_ten_to(%low) : (i64) -> f64
      %j0 = arith.shrui %k, %four : i64
      %bletch_bit = arith.andi %j0, %sixteen : i64
      %bletch = arith.cmpi ne, %bletch_bit, %zero : i64
      %j_masked = arith.andi %j0, %fifteen : i64
      %j = arith.select %bletch, %j_masked, %j0 : i64
      %c4i = arith.constant 4 : index
      %top = memref.load %bigtens[%c4i] : memref<5xf64>
      %u_pre = arith.divf %u, %top : f64
      %u1 = arith.select %bletch, %u_pre, %u : f64
      %three = arith.constant 3 : i64
      %ieps0 = arith.select %bletch, %three, %two : i64
      %walk:4 = scf.while (%jj = %j, %ii = %zero, %ds = %ds0, %ieps = %ieps0) : (i64, i64, f64, i64) -> (i64, i64, f64, i64) {
        %more = arith.cmpi ne, %jj, %zero : i64
        scf.condition(%more) %jj, %ii, %ds, %ieps : i64, i64, f64, i64
      } do {
      ^bb0(%jj: i64, %ii: i64, %ds: f64, %ieps: i64):
        %bit = arith.andi %jj, %one : i64
        %set = arith.cmpi ne, %bit, %zero : i64
        %index = arith.index_cast %ii : i64 to index
        %big = memref.load %bigtens[%index] : memref<5xf64>
        %ds_mul = arith.mulf %ds, %big : f64
        %ds_next = arith.select %set, %ds_mul, %ds : f64
        %ieps_inc = arith.addi %ieps, %one : i64
        %ieps_next = arith.select %set, %ieps_inc, %ieps : i64
        %jj_next = arith.shrui %jj, %one : i64
        %ii_next = arith.addi %ii, %one : i64
        scf.yield %jj_next, %ii_next, %ds_next, %ieps_next : i64, i64, f64, i64
      }
      %u2 = arith.divf %u1, %walk#2 : f64
      scf.yield %u2, %walk#3 : f64, i64
    } else {
      %neg:2 = scf.if %k_negative -> (f64, i64) {
        %j1 = arith.subi %zero, %k : i64
        %low = arith.andi %j1, %fifteen : i64
        %scale0 = func.call @__ly_dtoa_ten_to(%low) : (i64) -> f64
        %u1 = arith.mulf %u, %scale0 : f64
        %j = arith.shrui %j1, %four : i64
        %walk:4 = scf.while (%jj = %j, %ii = %zero, %uu = %u1, %ieps = %two) : (i64, i64, f64, i64) -> (i64, i64, f64, i64) {
          %more = arith.cmpi ne, %jj, %zero : i64
          scf.condition(%more) %jj, %ii, %uu, %ieps : i64, i64, f64, i64
        } do {
        ^bb0(%jj: i64, %ii: i64, %uu: f64, %ieps: i64):
          %bit = arith.andi %jj, %one : i64
          %set = arith.cmpi ne, %bit, %zero : i64
          %index = arith.index_cast %ii : i64 to index
          %big = memref.load %bigtens[%index] : memref<5xf64>
          %uu_mul = arith.mulf %uu, %big : f64
          %uu_next = arith.select %set, %uu_mul, %uu : f64
          %ieps_inc = arith.addi %ieps, %one : i64
          %ieps_next = arith.select %set, %ieps_inc, %ieps : i64
          %jj_next = arith.shrui %jj, %one : i64
          %ii_next = arith.addi %ii, %one : i64
          scf.yield %jj_next, %ii_next, %uu_next, %ieps_next : i64, i64, f64, i64
        }
        scf.yield %walk#2, %walk#3 : f64, i64
      } else {
        scf.yield %u, %two : f64, i64
      }
      scf.yield %neg#0, %neg#1 : f64, i64
    }
    // An estimate one too large leaves u below 1: take one digit fewer.
    %below_one = arith.cmpf olt, %scaled#0, %one_f : f64
    %some_digits = arith.cmpi sgt, %ilim, %zero : i64
    %fix0 = arith.andi %k_check, %below_one : i1
    %fix = arith.andi %fix0, %some_digits : i1
    %ilim1_empty = arith.cmpi sle, %ilim1, %zero : i64
    %fix_fails = arith.andi %fix, %ilim1_empty : i1
    cf.cond_br %fix_fails, ^small_integer, ^fixed_up

  ^fixed_up:
    %u_times_ten = arith.mulf %scaled#0, %ten_f : f64
    %uq = arith.select %fix, %u_times_ten, %scaled#0 : f64
    %ilim_q = arith.select %fix, %ilim1, %ilim : i64
    %k_less = arith.subi %k, %one : i64
    %kq = arith.select %fix, %k_less, %k : i64
    %ieps_more = arith.addi %scaled#1, %one : i64
    %ieps_q = arith.select %fix, %ieps_more, %scaled#1 : i64
    // eps = (ieps*u + 7) * 2^-(P-1): CPython lowers the exponent field by 52.
    %ieps_f = arith.sitofp %ieps_q : i64 to f64
    %eps_a = arith.mulf %ieps_f, %uq : f64
    %eps_b = arith.addf %eps_a, %seven_f : f64
    %two_m52 = arith.constant 2.220446049250313e-16 : f64
    %eps = arith.mulf %eps_b, %two_m52 : f64
    %no_digit_place = arith.cmpi eq, %ilim_q, %zero : i64
    cf.cond_br %no_digit_place, ^half_place, ^generate

  ^half_place:
    // The first digit is the cutoff place: one unit there, or nothing.
    %over_five = arith.subf %uq, %five_f : f64
    %neg_eps = arith.negf %eps : f64
    %clearly_up = arith.cmpf ogt, %over_five, %eps : f64
    %clearly_down = arith.cmpf olt, %over_five, %neg_eps : f64
    cf.cond_br %clearly_up, ^one_digit(%kq : i64), ^half_place_down

  ^half_place_down:
    cf.cond_br %clearly_down, ^no_digits, ^small_integer

  ^generate:
    %ilim_q_m1 = arith.subi %ilim_q, %one : i64
    %ten_place = func.call @__ly_dtoa_ten_to(%ilim_q_m1) : (i64) -> f64
    %eps_g = arith.mulf %eps, %ten_place : f64
    %gen:3 = scf.while (%i = %one, %uu = %uq, %lim = %ilim_q) : (i64, f64, i64) -> (i64, f64, i64) {
      %digit = arith.fptosi %uu : f64 to i64
      %digit_f = arith.sitofp %digit : i64 to f64
      %rest = arith.subf %uu, %digit_f : f64
      %exhausted = arith.cmpf oeq, %rest, %zero_f : f64
      %lim_now = arith.select %exhausted, %i, %lim : i64
      %ch_i64 = arith.addi %digit, %ascii_zero : i64
      %ch = arith.trunci %ch_i64 : i64 to i8
      %pos_i64 = arith.subi %i, %one : i64
      %pos = arith.index_cast %pos_i64 : i64 to index
      memref.store %ch, %digits_out[%pos] : memref<?xi8>
      %last = arith.cmpi eq, %i, %lim_now : i64
      %go = arith.xori %last, %true : i1
      scf.condition(%go) %i, %rest, %lim_now : i64, f64, i64
    } do {
    ^bb0(%i: i64, %rest: f64, %lim: i64):
      %i_next = arith.addi %i, %one : i64
      %u_next = arith.mulf %rest, %ten_f : f64
      scf.yield %i_next, %u_next, %lim : i64, f64, i64
    }
    %up_edge = arith.addf %half_f, %eps_g : f64
    %down_edge = arith.subf %half_f, %eps_g : f64
    %round_up = arith.cmpf ogt, %gen#1, %up_edge : f64
    %round_down_q = arith.cmpf olt, %gen#1, %down_edge : f64
    cf.cond_br %round_up, ^bump(%gen#0, %kq : i64, i64), ^quick_down

  ^quick_down:
    cf.cond_br %round_down_q, ^done(%gen#0, %kq : i64, i64), ^small_integer

  ^small_integer:
    // An integer below 10^15 divides exactly by tens[k].
    %floor_u = math.floor %u : f64
    %integral = arith.cmpf oeq, %floor_u, %u : f64
    %int_max = arith.constant 14 : i64
    %k_fits = arith.cmpi sle, %k, %int_max : i64
    %k_nonneg2 = arith.cmpi sge, %k, %zero : i64
    %k_ok = arith.andi %k_fits, %k_nonneg2 : i1
    %small = arith.andi %integral, %k_ok : i1
    cf.cond_br %small, ^small_digits, ^fail

  ^small_digits:
    %ds_int = func.call @__ly_dtoa_ten_to(%k) : (i64) -> f64
    // A cutoff at or above the integer's first digit (round(x, -n)): one unit
    // there or nothing, an exact half going down.
    %nd_negative = arith.cmpi slt, %ndigits, %zero : i64
    %no_place = arith.cmpi sle, %ilim, %zero : i64
    %cut_above0 = arith.andi %fixed_mode, %nd_negative : i1
    %cut_above = arith.andi %cut_above0, %no_place : i1
    cf.cond_br %cut_above, ^cut_above, ^small_loop

  ^cut_above:
    %ilim_negative = arith.cmpi slt, %ilim, %zero : i64
    %five_ds = arith.mulf %five_f, %ds_int : f64
    %at_most_half = arith.cmpf ole, %u, %five_ds : f64
    %nothing = arith.ori %ilim_negative, %at_most_half : i1
    cf.cond_br %nothing, ^no_digits, ^one_digit(%k : i64)

  ^small_loop:
    %sgen:4 = scf.while (%i = %one, %uu = %u) : (i64, f64) -> (i64, f64, i64, i1) {
      %q = arith.divf %uu, %ds_int : f64
      %digit = arith.fptosi %q : f64 to i64
      %digit_f = arith.sitofp %digit : i64 to f64
      %taken = arith.mulf %digit_f, %ds_int : f64
      %rest = arith.subf %uu, %taken : f64
      %ch_i64 = arith.addi %digit, %ascii_zero : i64
      %ch = arith.trunci %ch_i64 : i64 to i8
      %pos_i64 = arith.subi %i, %one : i64
      %pos = arith.index_cast %pos_i64 : i64 to index
      memref.store %ch, %digits_out[%pos] : memref<?xi8>
      %exhausted = arith.cmpf oeq, %rest, %zero_f : f64
      %at_limit = arith.cmpi eq, %i, %ilim : i64
      %stop = arith.ori %exhausted, %at_limit : i1
      // Round at the limit when something is left: twice the rest against
      // tens[k], a tie going to the even digit.
      %twice = arith.addf %rest, %rest : f64
      %over = arith.cmpf ogt, %twice, %ds_int : f64
      %tie = arith.cmpf oeq, %twice, %ds_int : f64
      %odd_bit = arith.andi %digit, %one : i64
      %odd = arith.cmpi ne, %odd_bit, %zero : i64
      %tie_up = arith.andi %tie, %odd : i1
      %up0 = arith.ori %over, %tie_up : i1
      %not_exhausted = arith.xori %exhausted, %true : i1
      %up1 = arith.andi %up0, %not_exhausted : i1
      %up = arith.andi %up1, %at_limit : i1
      %go = arith.xori %stop, %true : i1
      scf.condition(%go) %i, %rest, %digit, %up : i64, f64, i64, i1
    } do {
    ^bb0(%i: i64, %rest: f64, %digit: i64, %up: i1):
      %i_next = arith.addi %i, %one : i64
      %u_next = arith.mulf %rest, %ten_f : f64
      scf.yield %i_next, %u_next : i64, f64
    }
    cf.cond_br %sgen#3, ^bump(%sgen#0, %k : i64, i64), ^done(%sgen#0, %k : i64, i64)

  ^bump(%count_b: i64, %k_b: i64):
    %bumped:2 = func.call @__ly_d4_round_up(%digits_out, %count_b) : (memref<?xi8>, i64) -> (i64, i64)
    %k_after = arith.addi %k_b, %bumped#1 : i64
    cf.br ^done(%bumped#0, %k_after : i64, i64)

  ^done(%count_d: i64, %k_d: i64):
    %trimmed = func.call @__ly_dtoa_trim(%digits_out, %count_d) : (memref<?xi8>, i64) -> i64
    %decpt = arith.addi %k_d, %one : i64
    func.return %true, %trimmed, %decpt : i1, i64, i64

  ^one_digit(%k_o: i64):
    %one_ch = arith.constant 49 : i8
    %c0 = arith.constant 0 : index
    memref.store %one_ch, %digits_out[%c0] : memref<?xi8>
    %decpt_o = arith.addi %k_o, %two : i64
    func.return %true, %one, %decpt_o : i1, i64, i64

  ^no_digits:
    %decpt_n = arith.subi %zero, %ndigits : i64
    func.return %true, %zero, %decpt_n : i1, i64, i64

  ^fail:
    func.return %false, %zero, %zero : i1, i64, i64
  }

  // Fixed-count digits for a positive finite double: `fixed_mode` counts
  // digits after the decimal point ('f'), otherwise `req` is a significant
  // digit count ('e'/'g'). Correctly rounded, half to even, like CPython's
  // dtoa modes 3 and 2. Returns (digit count, decimal exponent k); trailing
  // zeros are trimmed (presentation pads), zero digits means the value
  // rounded to zero at the cutoff.
  func.func private @__ly_dtoa_counted(%abs: f64, %fixed_mode: i1, %req: i64, %digits_out: memref<?xi8>) -> (i64, i64) {
    %quick:3 = func.call @__ly_dtoa_quick(%abs, %fixed_mode, %req, %digits_out) : (f64, i1, i64, memref<?xi8>) -> (i1, i64, i64)
    cf.cond_br %quick#0, ^quick_answer, ^exact

  ^quick_answer:
    func.return %quick#1, %quick#2 : i64, i64

  ^exact:
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %ten = arith.constant 10 : i64
    %ascii_zero = arith.constant 48 : i64
    %zero_ch = arith.constant 48 : i8
    %true = arith.constant true

    %bits = arith.bitcast %abs : f64 to i64
    %mant_mask = arith.constant 4503599627370495 : i64
    %mant = arith.andi %bits, %mant_mask : i64
    %c52 = arith.constant 52 : i64
    %shifted = arith.shrui %bits, %c52 : i64
    %exp_mask = arith.constant 2047 : i64
    %biased = arith.andi %shifted, %exp_mask : i64
    %is_denormal = arith.cmpi eq, %biased, %zero : i64
    %implicit = arith.constant 4503599627370496 : i64
    %fnorm = arith.addi %mant, %implicit : i64
    %f = arith.select %is_denormal, %mant, %fnorm : i64
    %c1075 = arith.constant 1075 : i64
    %enorm = arith.subi %biased, %c1075 : i64
    %c_min_e = arith.constant -1074 : i64
    %e = arith.select %is_denormal, %c_min_e, %enorm : i64

    %r_store = memref.alloca() : memref<48xi32>
    %s_store = memref.alloca() : memref<48xi32>
    %t_store = memref.alloca() : memref<48xi32>
    %r = memref.cast %r_store : memref<48xi32> to memref<?xi32>
    %s = memref.cast %s_store : memref<48xi32> to memref<?xi32>
    %t = memref.cast %t_store : memref<48xi32> to memref<?xi32>

    %e_neg = arith.cmpi slt, %e, %zero : i64
    %lens:2 = scf.if %e_neg -> (i64, i64) {
      %rl = func.call @__ly_d4_from_u64(%r, %f) : (memref<?xi32>, i64) -> i64
      %sl0 = func.call @__ly_d4_from_u64(%s, %one) : (memref<?xi32>, i64) -> i64
      %se = arith.subi %zero, %e : i64
      %sl = func.call @__ly_d4_mul_pow2(%s, %sl0, %se) : (memref<?xi32>, i64, i64) -> i64
      scf.yield %rl, %sl : i64, i64
    } else {
      %rl0 = func.call @__ly_d4_from_u64(%r, %f) : (memref<?xi32>, i64) -> i64
      %rl = func.call @__ly_d4_mul_pow2(%r, %rl0, %e) : (memref<?xi32>, i64, i64) -> i64
      %sl = func.call @__ly_d4_from_u64(%s, %one) : (memref<?xi32>, i64) -> i64
      scf.yield %rl, %sl : i64, i64
    }

    %l10 = math.log10 %abs : f64
    %flr = math.floor %l10 : f64
    %k0f = arith.fptosi %flr : f64 to i64
    %k0 = arith.addi %k0f, %one : i64
    %k0_neg = arith.cmpi slt, %k0, %zero : i64
    %lens2:2 = scf.if %k0_neg -> (i64, i64) {
      %n10 = arith.subi %zero, %k0 : i64
      %rl = func.call @__ly_d4_mul_pow10(%r, %lens#0, %n10) : (memref<?xi32>, i64, i64) -> i64
      scf.yield %rl, %lens#1 : i64, i64
    } else {
      %sl = func.call @__ly_d4_mul_pow10(%s, %lens#1, %k0) : (memref<?xi32>, i64, i64) -> i64
      scf.yield %lens#0, %sl : i64, i64
    }

    // Fixup: R/S must land in [0.1, 1).
    %fix1:2 = scf.while (%k = %k0, %sl = %lens2#1) : (i64, i64) -> (i64, i64) {
      %cmp = func.call @__ly_d4_cmp(%r, %lens2#0, %s, %sl) : (memref<?xi32>, i64, memref<?xi32>, i64) -> i64
      %ge = arith.cmpi sge, %cmp, %zero : i64
      scf.condition(%ge) %k, %sl : i64, i64
    } do {
    ^bb0(%k: i64, %sl: i64):
      %nsl = func.call @__ly_d4_mul_small(%s, %sl, %ten) : (memref<?xi32>, i64, i64) -> i64
      %nk = arith.addi %k, %one : i64
      scf.yield %nk, %nsl : i64, i64
    }
    %fix2:2 = scf.while (%k = %fix1#0, %rl = %lens2#0) : (i64, i64) -> (i64, i64) {
      func.call @__ly_d4_copy(%r, %rl, %t) : (memref<?xi32>, i64, memref<?xi32>) -> ()
      %tlen = func.call @__ly_d4_mul_small(%t, %rl, %ten) : (memref<?xi32>, i64, i64) -> i64
      %cmp = func.call @__ly_d4_cmp(%t, %tlen, %s, %fix1#1) : (memref<?xi32>, i64, memref<?xi32>, i64) -> i64
      %lt = arith.cmpi slt, %cmp, %zero : i64
      scf.condition(%lt) %k, %rl : i64, i64
    } do {
    ^bb0(%k: i64, %rl: i64):
      %nrl = func.call @__ly_d4_mul_small(%r, %rl, %ten) : (memref<?xi32>, i64, i64) -> i64
      %nk = arith.subi %k, %one : i64
      scf.yield %nk, %nrl : i64, i64
    }
    %count_fixed = arith.addi %fix2#0, %req : i64
    %count_wanted = arith.select %fixed_mode, %count_fixed, %req : i64
    // Exact digits of a double never exceed 767; beyond that R is zero and
    // the digits are zeros, which the presentation layer pads on demand.
    %cap = arith.constant 780 : i64
    %count_capped = arith.minsi %count_wanted, %cap : i64
    %negative_cutoff = arith.cmpi slt, %count_wanted, %zero : i64
    cf.cond_br %negative_cutoff, ^round_to_zero, ^check_zero_cutoff

  ^round_to_zero:
    // |v| is strictly below half of the cutoff place value.
    %mk = arith.subi %zero, %req : i64
    func.return %zero, %mk : i64, i64

  ^check_zero_cutoff:
    %zero_cutoff = arith.cmpi eq, %count_wanted, %zero : i64
    cf.cond_br %zero_cutoff, ^half_test, ^generate

  ^half_test:
    // Zero digits requested at the cutoff: the result is either 0 or one
    // unit in the cutoff place, decided by 2R vs S (ties to even round down).
    func.call @__ly_d4_copy(%r, %fix2#1, %t) : (memref<?xi32>, i64, memref<?xi32>) -> ()
    %tlen_h = func.call @__ly_d4_mul_small(%t, %fix2#1, %two) : (memref<?xi32>, i64, i64) -> i64
    %cmp_h = func.call @__ly_d4_cmp(%t, %tlen_h, %s, %fix1#1) : (memref<?xi32>, i64, memref<?xi32>, i64) -> i64
    %up_h = arith.cmpi sgt, %cmp_h, %zero : i64
    cf.cond_br %up_h, ^half_up, ^half_down

  ^half_up:
    %one_ch = arith.constant 49 : i8
    %c0_h = arith.constant 0 : index
    memref.store %one_ch, %digits_out[%c0_h] : memref<?xi8>
    %kp1 = arith.addi %fix2#0, %one : i64
    func.return %one, %kp1 : i64, i64

  ^half_down:
    %mk_h = arith.subi %zero, %req : i64
    func.return %zero, %mk_h : i64, i64

  ^generate:
    %gen:2 = scf.while (%i = %zero, %rl = %fix2#1) : (i64, i64) -> (i64, i64) {
      %more = arith.cmpi slt, %i, %count_capped : i64
      %rl_nonzero = arith.cmpi ne, %rl, %zero : i64
      %continue = arith.andi %more, %rl_nonzero : i1
      scf.condition(%continue) %i, %rl : i64, i64
    } do {
    ^bb0(%i: i64, %rl: i64):
      %rl_a = func.call @__ly_d4_mul_small(%r, %rl, %ten) : (memref<?xi32>, i64, i64) -> i64
      %ext:2 = scf.while (%d = %zero, %rlc = %rl_a) : (i64, i64) -> (i64, i64) {
        %cmp = func.call @__ly_d4_cmp(%r, %rlc, %s, %fix1#1) : (memref<?xi32>, i64, memref<?xi32>, i64) -> i64
        %ge = arith.cmpi sge, %cmp, %zero : i64
        scf.condition(%ge) %d, %rlc : i64, i64
      } do {
      ^bb1(%d: i64, %rlc: i64):
        %nrl = func.call @__ly_d4_sub_inplace(%r, %rlc, %s, %fix1#1) : (memref<?xi32>, i64, memref<?xi32>, i64) -> i64
        %nd = arith.addi %d, %one : i64
        scf.yield %nd, %nrl : i64, i64
      }
      %ch_i64 = arith.addi %ext#0, %ascii_zero : i64
      %ch = arith.trunci %ch_i64 : i64 to i8
      %pos = arith.index_cast %i : i64 to index
      memref.store %ch, %digits_out[%pos] : memref<?xi8>
      %ni = arith.addi %i, %one : i64
      scf.yield %ni, %ext#1 : i64, i64
    }

    // Rounding at the cutoff (only when the remainder is nonzero).
    %rl_zero = arith.cmpi eq, %gen#1, %zero : i64
    %rounded:2 = scf.if %rl_zero -> (i64, i64) {
      scf.yield %gen#0, %fix2#0 : i64, i64
    } else {
      func.call @__ly_d4_copy(%r, %gen#1, %t) : (memref<?xi32>, i64, memref<?xi32>) -> ()
      %tlen = func.call @__ly_d4_mul_small(%t, %gen#1, %two) : (memref<?xi32>, i64, i64) -> i64
      %cmp2 = func.call @__ly_d4_cmp(%t, %tlen, %s, %fix1#1) : (memref<?xi32>, i64, memref<?xi32>, i64) -> i64
      %gt = arith.cmpi sgt, %cmp2, %zero : i64
      %eq = arith.cmpi eq, %cmp2, %zero : i64
      %lastpos_i = arith.subi %gen#0, %one : i64
      %lastpos = arith.index_cast %lastpos_i : i64 to index
      %last_ch = memref.load %digits_out[%lastpos] : memref<?xi8>
      %last_i64 = arith.extui %last_ch : i8 to i64
      %last_odd_bit = arith.andi %last_i64, %one : i64
      %last_odd = arith.cmpi ne, %last_odd_bit, %zero : i64
      %tie_up = arith.andi %eq, %last_odd : i1
      %up = arith.ori %gt, %tie_up : i1
      %res:2 = scf.if %up -> (i64, i64) {
        %bumped:2 = func.call @__ly_d4_round_up(%digits_out, %gen#0) : (memref<?xi8>, i64) -> (i64, i64)
        %nk = arith.addi %fix2#0, %bumped#1 : i64
        scf.yield %bumped#0, %nk : i64, i64
      } else {
        scf.yield %gen#0, %fix2#0 : i64, i64
      }
      scf.yield %res#0, %res#1 : i64, i64
    }

    // Trim trailing zeros; the presentation layer re-pads to the format's
    // width, so the digit string stays canonical like dtoa's.
    %trimmed = scf.while (%c = %rounded#0) : (i64) -> i64 {
      %has = arith.cmpi sgt, %c, %zero : i64
      %is_zero_ch = scf.if %has -> (i1) {
        %cm1 = arith.subi %c, %one : i64
        %idx = arith.index_cast %cm1 : i64 to index
        %chv = memref.load %digits_out[%idx] : memref<?xi8>
        %z = arith.cmpi eq, %chv, %zero_ch : i8
        scf.yield %z : i1
      } else {
        %f0 = arith.constant false
        scf.yield %f0 : i1
      }
      scf.condition(%is_zero_ch) %c : i64
    } do {
    ^bb0(%c: i64):
      %cm1 = arith.subi %c, %one : i64
      scf.yield %cm1 : i64
    }
    func.return %trimmed, %rounded#1 : i64, i64
  }
}
