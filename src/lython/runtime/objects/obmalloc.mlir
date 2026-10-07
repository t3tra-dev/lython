// The size checks in front of the allocator -- CPython's Objects/obmalloc.c
// side of an allocation.
//
// The allocator itself is built by the lowering (lowering/Common/
// RuntimeSupportBuilder.cpp, LyMem_*). Every count a manifest allocator hands
// it passes `__ly_alloc_count` first, which ends the program if the size is
// past the target's reach -- a runtime defect, because an entry point that
// takes its size from the program checks it first with
// `__ly_check_alloc_count`, which raises MemoryError.

module {
  // ===== declared here, defined in another runtime file or built by the lowering =====
  func.func private @LyErr_NoMemory() attributes {ly.runtime.contract = "builtins.MemoryError"}
  func.func private @ly_mem_max_request() -> i64
  func.func private @ly_mem_refuse(i64)

  // ⭐ THE COUNT OF A BUFFER TO ALLOCATE, guarded: past the largest request
  // the allocator makes on this target (`ly_mem_max_request`: the user
  // address space) -- `count` elements of `element_bytes` past `extra` bytes
  // -- the program ends (`ly_mem_refuse`). Every allocator below takes its
  // count through it, so a size that would wrap -- the product, or the cast to
  // `index`, at 2^32 on a 32-bit target -- is never a small block written
  // past. A size the PROGRAM computes is checked before it gets here, by
  // `__ly_check_alloc_count`, and is MemoryError.
  // ⛔ Not the MemoryError here: these allocators are behind every int, str
  // and list a program makes, and a call that may raise is one every caller
  // inherits as a landing pad -- genexpr.wasm's `__main__` grew 5 KB.
  func.func private @__ly_alloc_count(%count: i64, %element_bytes: i64, %extra: i64) -> index {
    %limit = func.call @ly_mem_max_request() : () -> i64
    %room = arith.subi %limit, %extra : i64
    %most = arith.divui %room, %element_bytes : i64
    %too_many = arith.cmpi ugt, %count, %most : i64
    scf.if %too_many {
      func.call @ly_mem_refuse(%count) : (i64) -> ()
    }
    %n = arith.index_cast %count : i64 to index
    func.return %n : index
  }

  // The same reach as `__ly_alloc_count`, refused with MemoryError: where a
  // size comes from the program (a repeat count, a pad width, `bytes(n)`, a
  // shift), as CPython's allocator refuses what no system can give.
  func.func private @__ly_check_alloc_count(%count: i64, %element_bytes: i64, %extra: i64) {
    %limit = func.call @ly_mem_max_request() : () -> i64
    %room = arith.subi %limit, %extra : i64
    %most = arith.divui %room, %element_bytes : i64
    %too_many = arith.cmpi ugt, %count, %most : i64
    scf.if %too_many {
      func.call @LyErr_NoMemory() : () -> ()
    }
    func.return
  }
}
