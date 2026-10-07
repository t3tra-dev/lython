// Hashing raw bytes -- CPython's Python/pyhash.c: SipHash-1-3 keyed with
// per-process random keys (getentropy), and the -1 -> -2 fixup every hash
// passes.

module {
  // ===== declared here, defined in another runtime file or built by the lowering =====
  func.func private @__ly_box_hash(%box: !llvm.ptr) -> i64
  func.func private @__ly_box_word_count() -> i64
  // ===== impls: hash =====
  // Runtime hash state: [k0, k1, initialized]. Filled once, lazily, from the
  // OS entropy pool (CPython randomizes str/bytes hashes per process; int and
  // float hashes stay unrandomized, matching CPython).
  memref.global "private" @__ly_hash_secret : memref<3xi64> = dense<0>

  // OS entropy (macOS libSystem / glibc >= 2.25).
  func.func private @getentropy(%buffer: !llvm.ptr, %length: i64) -> i32

  func.func private @__ly_hash_secret_keys() -> (i64, i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %secret = memref.get_global @__ly_hash_secret : memref<3xi64>
    %flag = memref.load %secret[%c2] : memref<3xi64>
    %needs_init = arith.cmpi eq, %flag, %zero : i64
    scf.if %needs_init {
      %ptr_index = memref.extract_aligned_pointer_as_index %secret : memref<3xi64> -> index
      %ptr_i64 = arith.index_cast %ptr_index : index to i64
      %ptr = llvm.inttoptr %ptr_i64 : i64 to !llvm.ptr
      %sixteen = arith.constant 16 : i64
      %rc = func.call @getentropy(%ptr, %sixteen) : (!llvm.ptr, i64) -> i32
      // On the (never-expected) entropy failure keep zero keys rather than
      // aborting: hashes stay internally consistent either way.
      %k0 = memref.load %secret[%c0] : memref<3xi64>
      %k1 = memref.load %secret[%c1] : memref<3xi64>
      %both_zero0 = arith.cmpi eq, %k0, %zero : i64
      %both_zero1 = arith.cmpi eq, %k1, %zero : i64
      %both_zero = arith.andi %both_zero0, %both_zero1 : i1
      scf.if %both_zero {
        // Degenerate zero keys weaken the mix; substitute fixed odd words.
        %fb0 = arith.constant 7266447313870364031 : i64
        %fb1 = arith.constant 4946485549665804864 : i64
        memref.store %fb0, %secret[%c0] : memref<3xi64>
        memref.store %fb1, %secret[%c1] : memref<3xi64>
      }
      memref.store %one, %secret[%c2] : memref<3xi64>
    }
    %out0 = memref.load %secret[%c0] : memref<3xi64>
    %out1 = memref.load %secret[%c1] : memref<3xi64>
    func.return %out0, %out1 : i64, i64
  }

  // One full SipHash round on the 4-lane state.
  func.func private @__ly_siphash_round(%v0_in: i64, %v1_in: i64, %v2_in: i64, %v3_in: i64) -> (i64, i64, i64, i64) {
    %c13 = arith.constant 13 : i64
    %c51 = arith.constant 51 : i64
    %c16 = arith.constant 16 : i64
    %c48 = arith.constant 48 : i64
    %c32 = arith.constant 32 : i64
    %c21 = arith.constant 21 : i64
    %c43 = arith.constant 43 : i64
    %c17 = arith.constant 17 : i64
    %c47 = arith.constant 47 : i64
    %a0 = arith.addi %v0_in, %v1_in : i64
    %r1a = arith.shli %v1_in, %c13 : i64
    %r1b = arith.shrui %v1_in, %c51 : i64
    %r1 = arith.ori %r1a, %r1b : i64
    %x1 = arith.xori %r1, %a0 : i64
    %r0a = arith.shli %a0, %c32 : i64
    %r0b = arith.shrui %a0, %c32 : i64
    %r0 = arith.ori %r0a, %r0b : i64
    %a2 = arith.addi %v2_in, %v3_in : i64
    %r3a = arith.shli %v3_in, %c16 : i64
    %r3b = arith.shrui %v3_in, %c48 : i64
    %r3 = arith.ori %r3a, %r3b : i64
    %x3 = arith.xori %r3, %a2 : i64
    %a0b = arith.addi %r0, %x3 : i64
    %r3c = arith.shli %x3, %c21 : i64
    %r3d = arith.shrui %x3, %c43 : i64
    %r3e = arith.ori %r3c, %r3d : i64
    %x3b = arith.xori %r3e, %a0b : i64
    %a2b = arith.addi %a2, %x1 : i64
    %r1c = arith.shli %x1, %c17 : i64
    %r1d = arith.shrui %x1, %c47 : i64
    %r1e = arith.ori %r1c, %r1d : i64
    %x1b = arith.xori %r1e, %a2b : i64
    %r2a = arith.shli %a2b, %c32 : i64
    %r2b = arith.shrui %a2b, %c32 : i64
    %r2 = arith.ori %r2a, %r2b : i64
    func.return %a0b, %x1b, %r2, %x3b : i64, i64, i64, i64
  }

  // SipHash-1-3 over raw bytes with the process-random keys (the CPython
  // default str/bytes hash since 3.11). -1 is remapped to -2 by callers.
  func.func private @__ly_hash_bytes(%ptr: i64, %len: i64) -> i64 {
    %zero = arith.constant 0 : i64
    %c8 = arith.constant 8 : i64
    %c56 = arith.constant 56 : i64
    %k0, %k1 = func.call @__ly_hash_secret_keys() : () -> (i64, i64)
    %iv0 = arith.constant 8317987319222330741 : i64
    %iv1 = arith.constant 7237128888997146477 : i64
    %iv2 = arith.constant 7816392313619706465 : i64
    %iv3 = arith.constant 8387220255154660723 : i64
    %v0_init = arith.xori %iv0, %k0 : i64
    %v1_init = arith.xori %iv1, %k1 : i64
    %v2_init = arith.xori %iv2, %k0 : i64
    %v3_init = arith.xori %iv3, %k1 : i64
    %base = llvm.inttoptr %ptr : i64 to !llvm.ptr
    %full_blocks = arith.divui %len, %c8 : i64
    %block_bytes = arith.muli %full_blocks, %c8 : i64
    %loop:5 = scf.while (%i = %zero, %v0 = %v0_init, %v1 = %v1_init, %v2 = %v2_init, %v3 = %v3_init) : (i64, i64, i64, i64, i64) -> (i64, i64, i64, i64, i64) {
      %more = arith.cmpi slt, %i, %block_bytes : i64
      scf.condition(%more) %i, %v0, %v1, %v2, %v3 : i64, i64, i64, i64, i64
    } do {
    ^bb0(%i: i64, %v0: i64, %v1: i64, %v2: i64, %v3: i64):
      %chunk_ptr = llvm.getelementptr %base[%i] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %m = llvm.load %chunk_ptr {alignment = 1 : i64} : !llvm.ptr -> i64
      %v3x = arith.xori %v3, %m : i64
      %s:4 = func.call @__ly_siphash_round(%v0, %v1, %v2, %v3x) : (i64, i64, i64, i64) -> (i64, i64, i64, i64)
      %v0x = arith.xori %s#0, %m : i64
      %next = arith.addi %i, %c8 : i64
      scf.yield %next, %v0x, %s#1, %s#2, %s#3 : i64, i64, i64, i64, i64
    }
    // Tail block: remaining bytes little-endian, length byte on top.
    %len_byte = arith.shli %len, %c56 : i64
    %tail_start = arith.index_cast %block_bytes : i64 to index
    %len_index = arith.index_cast %len : i64 to index
    %c1_index = arith.constant 1 : index
    %assembled = scf.for %bi = %tail_start to %len_index step %c1_index iter_args(%acc = %len_byte) -> (i64) {
      %bi_i64 = arith.index_cast %bi : index to i64
      %byte_ptr = llvm.getelementptr %base[%bi_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %byte = llvm.load %byte_ptr : !llvm.ptr -> i8
      %byte_i64 = arith.extui %byte : i8 to i64
      %rel = arith.subi %bi_i64, %block_bytes : i64
      %shift = arith.muli %rel, %c8 : i64
      %shifted = arith.shli %byte_i64, %shift : i64
      %next = arith.ori %acc, %shifted : i64
      scf.yield %next : i64
    }
    %v3t = arith.xori %loop#4, %assembled : i64
    %t:4 = func.call @__ly_siphash_round(%loop#1, %loop#2, %loop#3, %v3t) : (i64, i64, i64, i64) -> (i64, i64, i64, i64)
    %v0t = arith.xori %t#0, %assembled : i64
    %ff = arith.constant 255 : i64
    %v2f = arith.xori %t#2, %ff : i64
    %f1:4 = func.call @__ly_siphash_round(%v0t, %t#1, %v2f, %t#3) : (i64, i64, i64, i64) -> (i64, i64, i64, i64)
    %f2:4 = func.call @__ly_siphash_round(%f1#0, %f1#1, %f1#2, %f1#3) : (i64, i64, i64, i64) -> (i64, i64, i64, i64)
    %f3:4 = func.call @__ly_siphash_round(%f2#0, %f2#1, %f2#2, %f2#3) : (i64, i64, i64, i64) -> (i64, i64, i64, i64)
    %h01 = arith.xori %f3#0, %f3#1 : i64
    %h23 = arith.xori %f3#2, %f3#3 : i64
    %h = arith.xori %h01, %h23 : i64
    func.return %h : i64
  }

  // Shared -1 -> -2 remap (CPython reserves -1 as the C-level error return).
  func.func private @__ly_hash_fixup(%h: i64) -> i64 {
    %neg_one = arith.constant -1 : i64
    %neg_two = arith.constant -2 : i64
    %is_neg_one = arith.cmpi eq, %h, %neg_one : i64
    %fixed = arith.select %is_neg_one, %neg_two, %h : i1, i64
    func.return %fixed : i64
  }
  // The xxHash lanes tuplehash and slice_hash both run over their items
  // (Objects/tupleobject.c, Objects/sliceobject.c): each item's hash mixed
  // in by XXPRIME_2, rotated 31, multiplied by XXPRIME_1. The two differ
  // after the loop -- a tuple adds its length, a slice does not -- so each
  // finishes the accumulator itself.
  func.func private @__ly_xxhash_slot_lanes(%items: !llvm.ptr, %count: i64) -> i64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %words = func.call @__ly_box_word_count() : () -> i64
    %count_index = arith.index_cast %count : i64 to index
    %prime1 = arith.constant -7046029288634856825 : i64
    %prime2 = arith.constant -4417276706812531889 : i64
    %acc_init = arith.constant 2870177450012600261 : i64
    %c31 = arith.constant 31 : i64
    %c33 = arith.constant 33 : i64
    %acc = scf.for %i = %c0 to %count_index step %c1 iter_args(%a = %acc_init) -> (i64) {
      %i_i64 = arith.index_cast %i : index to i64
      %off = arith.muli %i_i64, %words : i64
      %box_ptr = llvm.getelementptr %items[%off] : (!llvm.ptr, i64) -> !llvm.ptr, i64
      %lane = func.call @__ly_box_hash(%box_ptr) : (!llvm.ptr) -> i64
      %scaled = arith.muli %lane, %prime2 : i64
      %added = arith.addi %a, %scaled : i64
      %rot_hi = arith.shli %added, %c31 : i64
      %rot_lo = arith.shrui %added, %c33 : i64
      %rotated = arith.ori %rot_hi, %rot_lo : i64
      %next = arith.muli %rotated, %prime1 : i64
      scf.yield %next : i64
    }
    func.return %acc : i64
  }
}
