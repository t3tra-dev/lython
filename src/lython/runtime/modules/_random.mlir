// Contract manifest AND runtime implementation for `_random` (CPython's
// Modules/_randommodule.c counterpart): the Mersenne Twister MT19937 itself.
//
// Signature sources (1:1 correspondence target):
//   https://github.com/python/typeshed/blob/main/stdlib/_random.pyi
//   https://github.com/python/cpython/blob/main/Modules/_randommodule.c
//
// The generator is written here in MLIR rather than behind a LyHost_* wrapper
// because it needs NO libc and NO target-dependent fact: it is 32-bit integer
// arithmetic over a fixed state array, so it stays in the target-independent
// manifest where the rest of the module's semantics live.
//
// The state is bit-exact with CPython's: init_by_array over the seed's 32-bit
// words, the standard tempering, and random() built from two 32-bit draws the
// same way (a >> 5, b >> 6). Seeding with the same integer therefore yields
// the SAME sequence CPython yields, which is what lib/random.py's golden case
// verifies.
//
// Deviations from CPython:
// - there is ONE generator, not a `Random` class. CPython's module-level
//   functions are bound methods of a hidden instance and callers may build
//   more instances; a manifest cannot declare a class with 625 words of state
//   yet, so the hidden instance is a module global here and independent
//   generators are unavailable. `getstate`/`setstate` are absent for the same
//   reason (they would hand out the state as a 625-tuple).
// - the automatic seed is the realtime clock in nanoseconds, not os.urandom:
//   there is no urandom binding yet. An explicitly seeded generator is
//   unaffected, and that is the only case whose sequence is specified.
// - `seed()` takes an int whose magnitude fits in 64 bits (two key words).
//   CPython accepts an arbitrarily large int, plus None/float/str/bytes; the
//   int path is bit-identical within that range.
// - `getrandbits(k)` accepts 0 <= k <= 63. CPython has no upper bound; 63 is
//   where the result stops fitting the signed 64-bit lane a manifest can
//   return, so `randbelow` covers ranges below 2**63.
// - `randbelow` and `randint` are CPython's PYTHON-level Lib/random.py code
//   (`_randbelow_with_getrandbits` and the `a + _randbelow(b - a + 1)` that
//   calls it), moved here because a Python function containing a loop is
//   called TWICE when another Python function calls it -- reported to the
//   Wave 3 foundation track. The duplicate consumed a second draw and returned
//   it, so `randint(1, 100)` after `seed(42)` silently gave 15 instead of 82.
//   Both belong back in random.py once that is fixed; the draw sequence here is
//   CPython's either way.

module attributes {
  ly.typing.module = "_random",
  ly.typing.callable_exports = [
    "_random.seed",
    "_random.random",
    "_random.getrandbits",
    "_random.randbelow",
    "_random.randint",
    "_random.gauss_take",
    "_random.gauss_put"
  ],
  ly.typing.function_names = [
    "_random.seed",
    "_random.random",
    "_random.getrandbits",
    "_random.randbelow",
    "_random.randint",
    "_random.gauss_take",
    "_random.gauss_put"
  ],
  ly.typing.function_contracts = [
    !py.callable<[!py.contract<"builtins.int">], arg_names = ["a"], arg_defaults = [false], returns = [!py.literal<None>]>,
    !py.callable<[], arg_names = [], arg_defaults = [], returns = [!py.contract<"builtins.float">]>,
    !py.callable<[!py.contract<"builtins.int">], arg_names = ["k"], arg_defaults = [false], returns = [!py.contract<"builtins.int">]>,
    !py.callable<[!py.contract<"builtins.int">], arg_names = ["n"], arg_defaults = [false], returns = [!py.contract<"builtins.int">]>,
    !py.callable<[!py.contract<"builtins.int">, !py.contract<"builtins.int">], arg_names = ["a", "b"], arg_defaults = [false, false], returns = [!py.contract<"builtins.int">]>,
    !py.callable<[], arg_names = [], arg_defaults = [], returns = [!py.contract<"builtins.float">]>,
    !py.callable<[!py.contract<"builtins.float">], arg_names = ["value"], arg_defaults = [false], returns = [!py.literal<None>]>
  ]
} {
  // --- shared runtime entry points -----------------------------------------
  func.func private @LyLong_FromI64(%value: i64 {ly.runtime.default_i64 = 0 : i64}) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 1 : i64, ly.runtime.contract = "builtins.int", ly.runtime.initializer = "__new__"}
  func.func private @LyLong_AsI64(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %digits: memref<?xi32>) -> i64 attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__int__", ly.runtime.primitive = "unbox.i64"}
  func.func private @LyFloat_FromF64(%value: f64 {ly.runtime.default_f64 = 0.0 : f64}) -> memref<3xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 2 : i64, ly.runtime.contract = "builtins.float", ly.runtime.initializer = "__new__"}
  func.func private @LyFloat_AsF64(%header: memref<3xi64> {ly.ownership.object_header}) -> f64 attributes {ly.runtime.contract = "builtins.float", ly.runtime.method = "__float__", ly.runtime.primitive = "unbox.f64"}
  func.func private @LyUnicode_FromBytes(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 4 : i64, ly.runtime.contract = "builtins.str", ly.runtime.initializer = "__new__"}
  func.func private @LyBaseException_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 5 : i64, ly.runtime.contract = "builtins.BaseException", ly.runtime.initializer = "__new__"}
  func.func private @LyBaseException_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.BaseException", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"}
  func.func private @LyEH_ThrowException(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "raise"}
  func.func private @LyHost_ClockNs(i64) -> i64

  // MT19937 state: [0..623] the 624 state words, [624] the draw index (mti),
  // [625] the seeded flag, [626] whether a gauss deviate is cached, [627] that
  // deviate's f64 bits. The gauss cache rides here because CPython's
  // gauss()/normalvariate() caches the second Box-Muller deviate on the Random
  // instance, and this module IS that instance.
  memref.global "private" @__ly_random_state : memref<628xi64> = dense<0>

  memref.global "private" constant @__ly_random_msg_bits : memref<44xi8> = dense<[110, 117, 109, 98, 101, 114, 32, 111, 102, 32, 98, 105, 116, 115, 32, 109, 117, 115, 116, 32, 98, 101, 32, 105, 110, 32, 91, 48, 44, 32, 54, 51, 93, 32, 105, 110, 32, 116, 104, 105, 115, 32, 112, 111]>
  memref.global "private" constant @__ly_random_msg_bits2 : memref<3xi8> = dense<[114, 116, 0]>

  memref.global "private" constant @__ly_random_msg_empty : memref<24xi8> = dense<[101, 109, 112, 116, 121, 32, 114, 97, 110, 103, 101, 32, 105, 110, 32, 114, 97, 110, 100, 105, 110, 116, 40, 41]>

  func.func private @__ly_raise_static_message(%class_id: i64, %message: memref<?xi8>, %length: i64)
  func.func private @__ly_raise_message_object(%class_id: i64, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>)

  func.func private @__ly_random_raise_empty() {
    %len = arith.constant 24 : i64
    %class_id = arith.constant 53 : i64
    // The message is a static global, so it is passed as a view rather than
    // copied into a heap buffer first. The copy was the leak: nothing frees it,
    // and the throw does not return. Every static-message raise is called with
    // a global this way -- this is that, not a new convention.
    %text = memref.get_global @__ly_random_msg_empty : memref<24xi8>
    %buffer = memref.cast %text : memref<24xi8> to memref<?xi8>
    func.call @__ly_raise_static_message(%class_id, %buffer, %len) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @__ly_random_raise_bits() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c44 = arith.constant 44 : index
    %c2 = arith.constant 2 : index
    %c46 = arith.constant 46 : index
    %len = arith.constant 46 : i64
    %class_id = arith.constant 53 : i64
    %buffer = memref.alloc(%c46) : memref<?xi8>
    %head = memref.get_global @__ly_random_msg_bits : memref<44xi8>
    %tail = memref.get_global @__ly_random_msg_bits2 : memref<3xi8>
    scf.for %i = %c0 to %c44 step %c1 {
      %byte = memref.load %head[%i] : memref<44xi8>
      memref.store %byte, %buffer[%i] : memref<?xi8>
    }
    scf.for %i = %c0 to %c2 step %c1 {
      %byte = memref.load %tail[%i] : memref<3xi8>
      %dest = arith.addi %c44, %i : index
      memref.store %byte, %buffer[%dest] : memref<?xi8>
    }
    %message_header, %message_bytes = func.call @LyUnicode_FromBytes(%buffer, %c0, %len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    // This one really needs a buffer -- it joins two globals -- so it is freed as
    // soon as the string object has copied it, and BEFORE the throw, which does
    // not return. That is why it raises through the message-object entry and
    // not the static-message one. The sibling above passes a global view
    // instead and has nothing to free; deallocating THAT is a free of a
    // non-heap pointer, which is how this landed in the wrong function once
    // (SIGABRT, caught by the leak gate refusing to measure a program that does
    // not exit 0 on its own).
    memref.dealloc %buffer : memref<?xi8>
    func.call @__ly_raise_message_object(%class_id, %message_header, %message_bytes) : (i64, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  // init_genrand: Knuth's multiplicative seeding of the whole state array.
  func.func private @__ly_random_init_genrand(%s: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c624 = arith.constant 624 : index
    %c624_i64 = arith.constant 624 : i64
    %mask = arith.constant 4294967295 : i64
    %mult = arith.constant 1812433253 : i64
    %c30_i64 = arith.constant 30 : i64
    %state = memref.get_global @__ly_random_state : memref<628xi64>
    %first = arith.andi %s, %mask : i64
    memref.store %first, %state[%c0] : memref<628xi64>
    scf.for %i = %c1 to %c624 step %c1 {
      %prev_index = arith.subi %i, %c1 : index
      %prev = memref.load %state[%prev_index] : memref<628xi64>
      %shifted = arith.shrui %prev, %c30_i64 : i64
      %mixed = arith.xori %prev, %shifted : i64
      %scaled = arith.muli %mixed, %mult : i64
      %i_i64 = arith.index_cast %i : index to i64
      %summed = arith.addi %scaled, %i_i64 : i64
      %word = arith.andi %summed, %mask : i64
      memref.store %word, %state[%i] : memref<628xi64>
    }
    %index_slot = arith.constant 624 : index
    memref.store %c624_i64, %state[%index_slot] : memref<628xi64>
    func.return
  }

  // init_by_array over the seed's 32-bit words. `keylen` is 1 or 2, so the
  // key lookup is a select rather than an array read.
  func.func private @__ly_random_init_by_array(%w0: i64, %w1: i64, %keylen: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c623 = arith.constant 623 : index
    %c624 = arith.constant 624 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %mask = arith.constant 4294967295 : i64
    %thirty = arith.constant 30 : i64
    %mult1 = arith.constant 1664525 : i64
    %mult2 = arith.constant 1566083941 : i64
    %n = arith.constant 624 : i64
    %seed0 = arith.constant 19650218 : i64
    %top = arith.constant 2147483648 : i64
    %c623_i64 = arith.constant 623 : i64
    %c624_i64 = arith.constant 624 : i64

    func.call @__ly_random_init_genrand(%seed0) : (i64) -> ()
    %state = memref.get_global @__ly_random_state : memref<628xi64>

    // First pass: max(624, keylen) rounds. keylen never exceeds 2 here, so the
    // bound is the state length.
    %pass1:2 = scf.for %step = %c0 to %c624 step %c1 iter_args(%i = %one, %j = %zero) -> (i64, i64) {
      %i_index = arith.index_cast %i : i64 to index
      %prev_i = arith.subi %i, %one : i64
      %prev_index = arith.index_cast %prev_i : i64 to index
      %cur = memref.load %state[%i_index] : memref<628xi64>
      %prev = memref.load %state[%prev_index] : memref<628xi64>
      %prev_shift = arith.shrui %prev, %thirty : i64
      %prev_mix = arith.xori %prev, %prev_shift : i64
      %prev_scaled = arith.muli %prev_mix, %mult1 : i64
      %combined = arith.xori %cur, %prev_scaled : i64
      %use_w0 = arith.cmpi eq, %j, %zero : i64
      %key = arith.select %use_w0, %w0, %w1 : i64
      %with_key = arith.addi %combined, %key : i64
      %with_j = arith.addi %with_key, %j : i64
      %word = arith.andi %with_j, %mask : i64
      memref.store %word, %state[%i_index] : memref<628xi64>

      %i_next = arith.addi %i, %one : i64
      %wrapped = arith.cmpi sge, %i_next, %c624_i64 : i64
      %i_out = scf.if %wrapped -> i64 {
        %last_index = arith.index_cast %c623_i64 : i64 to index
        %last = memref.load %state[%last_index] : memref<628xi64>
        memref.store %last, %state[%c0] : memref<628xi64>
        scf.yield %one : i64
      } else {
        scf.yield %i_next : i64
      }
      %j_next = arith.addi %j, %one : i64
      %j_wrapped = arith.cmpi sge, %j_next, %keylen : i64
      %j_out = arith.select %j_wrapped, %zero, %j_next : i64
      scf.yield %i_out, %j_out : i64, i64
    }

    // Second pass: 623 rounds, subtracting the index instead of adding a key.
    %pass2 = scf.for %step = %c0 to %c623 step %c1 iter_args(%i = %pass1#0) -> (i64) {
      %i_index = arith.index_cast %i : i64 to index
      %prev_i = arith.subi %i, %one : i64
      %prev_index = arith.index_cast %prev_i : i64 to index
      %cur = memref.load %state[%i_index] : memref<628xi64>
      %prev = memref.load %state[%prev_index] : memref<628xi64>
      %prev_shift = arith.shrui %prev, %thirty : i64
      %prev_mix = arith.xori %prev, %prev_shift : i64
      %prev_scaled = arith.muli %prev_mix, %mult2 : i64
      %combined = arith.xori %cur, %prev_scaled : i64
      %reduced = arith.subi %combined, %i : i64
      %word = arith.andi %reduced, %mask : i64
      memref.store %word, %state[%i_index] : memref<628xi64>

      %i_next = arith.addi %i, %one : i64
      %wrapped = arith.cmpi sge, %i_next, %c624_i64 : i64
      %i_out = scf.if %wrapped -> i64 {
        %last_index = arith.index_cast %c623_i64 : i64 to index
        %last = memref.load %state[%last_index] : memref<628xi64>
        memref.store %last, %state[%c0] : memref<628xi64>
        scf.yield %one : i64
      } else {
        scf.yield %i_next : i64
      }
      scf.yield %i_out : i64
    }

    // MSB of state[0] set: guarantees a non-zero initial array.
    memref.store %top, %state[%c0] : memref<628xi64>
    %index_slot = arith.constant 624 : index
    memref.store %c624_i64, %state[%index_slot] : memref<628xi64>
    %seeded_slot = arith.constant 625 : index
    memref.store %one, %state[%seeded_slot] : memref<628xi64>
    func.return
  }

  // One tempered 32-bit draw. Twists the whole array when the index runs out.
  // The single modular loop is equivalent to the reference implementation's
  // three-part loop: its second part deliberately reads words this pass has
  // already rewritten, which `(kk + 397) % 624` reproduces.
  func.func private @__ly_random_genrand() -> i64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c624 = arith.constant 624 : index
    %c624_index = arith.constant 624 : index
    %c397 = arith.constant 397 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %c624_i64 = arith.constant 624 : i64
    %upper = arith.constant 2147483648 : i64
    %lower = arith.constant 2147483647 : i64
    %mag = arith.constant 2567483615 : i64
    %mask = arith.constant 4294967295 : i64
    %eleven = arith.constant 11 : i64
    %seven = arith.constant 7 : i64
    %fifteen = arith.constant 15 : i64
    %eighteen = arith.constant 18 : i64
    %temper1 = arith.constant 2636928640 : i64
    %temper2 = arith.constant 4022730752 : i64
    %index_slot = arith.constant 624 : index
    %seeded_slot = arith.constant 625 : index

    %state = memref.get_global @__ly_random_state : memref<628xi64>
    %seeded = memref.load %state[%seeded_slot] : memref<628xi64>
    %unseeded = arith.cmpi eq, %seeded, %zero : i64
    scf.if %unseeded {
      // No urandom binding yet: the realtime clock stands in, and an explicit
      // seed() overwrites it before any draw that matters.
      %now = func.call @LyHost_ClockNs(%zero) : (i64) -> i64
      %lo = arith.andi %now, %mask : i64
      %thirtytwo = arith.constant 32 : i64
      %hi_raw = arith.shrui %now, %thirtytwo : i64
      %hi = arith.andi %hi_raw, %mask : i64
      %two = arith.constant 2 : i64
      func.call @__ly_random_init_by_array(%lo, %hi, %two) : (i64, i64, i64) -> ()
    }

    %mti = memref.load %state[%index_slot] : memref<628xi64>
    %exhausted = arith.cmpi sge, %mti, %c624_i64 : i64
    scf.if %exhausted {
      scf.for %kk = %c0 to %c624 step %c1 {
        %next_raw = arith.addi %kk, %c1 : index
        %next = arith.remui %next_raw, %c624_index : index
        %cur = memref.load %state[%kk] : memref<628xi64>
        %follow = memref.load %state[%next] : memref<628xi64>
        %cur_high = arith.andi %cur, %upper : i64
        %follow_low = arith.andi %follow, %lower : i64
        %y = arith.ori %cur_high, %follow_low : i64
        %far_raw = arith.addi %kk, %c397 : index
        %far = arith.remui %far_raw, %c624_index : index
        %source = memref.load %state[%far] : memref<628xi64>
        %y_shift = arith.shrui %y, %one : i64
        %odd = arith.andi %y, %one : i64
        %is_odd = arith.cmpi eq, %odd, %one : i64
        %twist = arith.select %is_odd, %mag, %zero : i64
        %step1 = arith.xori %source, %y_shift : i64
        %word = arith.xori %step1, %twist : i64
        memref.store %word, %state[%kk] : memref<628xi64>
      }
      memref.store %zero, %state[%index_slot] : memref<628xi64>
    }

    %draw_index = memref.load %state[%index_slot] : memref<628xi64>
    %draw_slot = arith.index_cast %draw_index : i64 to index
    %raw = memref.load %state[%draw_slot] : memref<628xi64>
    %advanced = arith.addi %draw_index, %one : i64
    memref.store %advanced, %state[%index_slot] : memref<628xi64>

    %t1_shift = arith.shrui %raw, %eleven : i64
    %t1 = arith.xori %raw, %t1_shift : i64
    %t2_shift_raw = arith.shli %t1, %seven : i64
    %t2_shift = arith.andi %t2_shift_raw, %temper1 : i64
    %t2 = arith.xori %t1, %t2_shift : i64
    %t3_shift_raw = arith.shli %t2, %fifteen : i64
    %t3_shift = arith.andi %t3_shift_raw, %temper2 : i64
    %t3 = arith.xori %t2, %t3_shift : i64
    %t4_shift = arith.shrui %t3, %eighteen : i64
    %t4 = arith.xori %t3, %t4_shift : i64
    %result = arith.andi %t4, %mask : i64
    func.return %result : i64
  }

  // --- public surface ------------------------------------------------------

  func.func @LyRandom_Seed(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %digits: memref<?xi32>) attributes {ly.runtime.builtin = "_random.seed", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.int", ly.runtime.primitive = "random_seed", ly.runtime.result_contract = "types.NoneType"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %mask = arith.constant 4294967295 : i64
    %thirtytwo = arith.constant 32 : i64
    %value = func.call @LyLong_AsI64(%header, %meta, %digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    // CPython seeds from abs(a): the sign carries no entropy.
    %negative = arith.cmpi slt, %value, %zero : i64
    %negated = arith.subi %zero, %value : i64
    %magnitude = arith.select %negative, %negated, %value : i64
    %w0 = arith.andi %magnitude, %mask : i64
    %w1_raw = arith.shrui %magnitude, %thirtytwo : i64
    %w1 = arith.andi %w1_raw, %mask : i64
    // keyused is the number of significant 32-bit words, at least one.
    %needs_two = arith.cmpi ne, %w1, %zero : i64
    %keylen = arith.select %needs_two, %two, %one : i64
    func.call @__ly_random_init_by_array(%w0, %w1, %keylen) : (i64, i64, i64) -> ()
    func.return
  }

  // CPython's random_random: two draws, 53 bits of mantissa.
  func.func @LyRandom_Random() -> memref<3xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "_random.random", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.float", ly.runtime.primitive = "random_random", ly.runtime.result_contract = "builtins.float"} {
    %five = arith.constant 5 : i64
    %six = arith.constant 6 : i64
    %scale_a = arith.constant 67108864.0 : f64
    %scale_all = arith.constant 9007199254740992.0 : f64
    %first = func.call @__ly_random_genrand() : () -> i64
    %second = func.call @__ly_random_genrand() : () -> i64
    %a = arith.shrui %first, %five : i64
    %b = arith.shrui %second, %six : i64
    %a_f = arith.uitofp %a : i64 to f64
    %b_f = arith.uitofp %b : i64 to f64
    %scaled = arith.mulf %a_f, %scale_a : f64
    %summed = arith.addf %scaled, %b_f : f64
    %value = arith.divf %summed, %scale_all : f64
    %h = func.call @LyFloat_FromF64(%value) : (f64) -> memref<3xi64>
    func.return %h : memref<3xi64>
  }

  // CPython's random_getrandbits, bounded to the signed 64-bit lane: the low
  // word is the FIRST draw and the high word the second, right-shifted to keep
  // exactly the requested bits.
  func.func private @__ly_random_bits(%k: i64) -> i64 {
    %zero = arith.constant 0 : i64
    %thirtytwo = arith.constant 32 : i64
    %sixtyfour = arith.constant 64 : i64
    %is_zero = arith.cmpi sle, %k, %zero : i64
    %value = scf.if %is_zero -> i64 {
      scf.yield %zero : i64
    } else {
      %narrow = arith.cmpi sle, %k, %thirtytwo : i64
      %answer = scf.if %narrow -> i64 {
        %draw = func.call @__ly_random_genrand() : () -> i64
        %shift = arith.subi %thirtytwo, %k : i64
        %kept = arith.shrui %draw, %shift : i64
        scf.yield %kept : i64
      } else {
        %low = func.call @__ly_random_genrand() : () -> i64
        %high_raw = func.call @__ly_random_genrand() : () -> i64
        %shift = arith.subi %sixtyfour, %k : i64
        %high = arith.shrui %high_raw, %shift : i64
        %placed = arith.shli %high, %thirtytwo : i64
        %combined = arith.ori %placed, %low : i64
        scf.yield %combined : i64
      }
      scf.yield %answer : i64
    }
    func.return %value : i64
  }

  func.func @LyRandom_GetRandBits(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %digits: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "_random.getrandbits", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.int", ly.runtime.primitive = "random_getrandbits", ly.runtime.result_contract = "builtins.int"} {
    %zero = arith.constant 0 : i64
    %sixtythree = arith.constant 63 : i64
    %k = func.call @LyLong_AsI64(%header, %meta, %digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %too_small = arith.cmpi slt, %k, %zero : i64
    %too_big = arith.cmpi sgt, %k, %sixtythree : i64
    %invalid = arith.ori %too_small, %too_big : i1
    scf.if %invalid {
      func.call @__ly_random_raise_bits() : () -> ()
    }
    %value = func.call @__ly_random_bits(%k) : (i64) -> i64
    %h, %m, %d = func.call @LyLong_FromI64(%value) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  // CPython's _randbelow_with_getrandbits, which is Python in Lib/random.py.
  // It is native HERE because a Python function containing a loop is called
  // twice when another Python function calls it (reported to the Wave 3
  // foundation track), and this one consumes generator draws -- the duplicate
  // call silently returned the second draw. The draw sequence is CPython's:
  // n.bit_length() bits per attempt, rejecting anything >= n.
  func.func private @__ly_random_below(%n: i64) -> i64 {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %empty = arith.cmpi sle, %n, %zero : i64
    %value = scf.if %empty -> i64 {
      // CPython's is defined for n > 0 and returns 0 for a falsy n.
      scf.yield %zero : i64
    } else {
      // n.bit_length(): the number of halvings before n reaches zero.
      %halved:2 = scf.while (%rest = %n, %count = %zero) : (i64, i64) -> (i64, i64) {
        %more = arith.cmpi sgt, %rest, %zero : i64
        %next_rest = arith.divui %rest, %two : i64
        %next_count = arith.addi %count, %one : i64
        %rest_out = arith.select %more, %next_rest, %rest : i64
        %count_out = arith.select %more, %next_count, %count : i64
        scf.condition(%more) %rest_out, %count_out : i64, i64
      } do {
      ^body(%rest_in: i64, %count_in: i64):
        scf.yield %rest_in, %count_in : i64, i64
      }
      %bits = arith.select %empty, %zero, %halved#1 : i64
      // Rejection sampling: redraw until the value lands inside [0, n).
      %drawn = scf.while (%candidate = %n) : (i64) -> i64 {
        %rejected = arith.cmpi sge, %candidate, %n : i64
        %next = scf.if %rejected -> i64 {
          %fresh = func.call @__ly_random_bits(%bits) : (i64) -> i64
          scf.yield %fresh : i64
        } else {
          scf.yield %candidate : i64
        }
        scf.condition(%rejected) %next : i64
      } do {
      ^body(%carried: i64):
        scf.yield %carried : i64
      }
      scf.yield %drawn : i64
    }
    func.return %value : i64
  }

  func.func @LyRandom_RandBelow(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %digits: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "_random.randbelow", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.int", ly.runtime.primitive = "random_randbelow", ly.runtime.result_contract = "builtins.int"} {
    %n = func.call @LyLong_AsI64(%header, %meta, %digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %value = func.call @__ly_random_below(%n) : (i64) -> i64
    %h, %m, %d = func.call @LyLong_FromI64(%value) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  // CPython 3.14's Random.randint: `a + self._randbelow(b - a + 1)`, drawn
  // WITHOUT going through randrange. It is native for the same reason
  // randbelow is: the Python spelling in an imported module consumed a second
  // draw, so the answer was silently the wrong one.
  func.func @LyRandom_RandInt(%a_header: memref<2xi64> {ly.ownership.object_header}, %a_meta: memref<2xi64>, %a_digits: memref<?xi32>, %b_header: memref<2xi64> {ly.ownership.object_header}, %b_meta: memref<2xi64>, %b_digits: memref<?xi32>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "_random.randint", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.int", ly.runtime.primitive = "random_randint", ly.runtime.result_contract = "builtins.int"} {
    %one = arith.constant 1 : i64
    %a = func.call @LyLong_AsI64(%a_header, %a_meta, %a_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %b = func.call @LyLong_AsI64(%b_header, %b_meta, %b_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %span = arith.subi %b, %a : i64
    %width = arith.addi %span, %one : i64
    %empty = arith.cmpi slt, %b, %a : i64
    scf.if %empty {
      func.call @__ly_random_raise_empty() : () -> ()
    }
    %drawn = func.call @__ly_random_below(%width) : (i64) -> i64
    %value = arith.addi %a, %drawn : i64
    %h, %m, %d = func.call @LyLong_FromI64(%value) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  // The cached second Box-Muller deviate. NaN means "nothing cached", which is
  // how random.py avoids needing an Optional[float] across the boundary; a
  // real deviate is never NaN.
  func.func @LyRandom_GaussTake() -> memref<3xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "_random.gauss_take", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.float", ly.runtime.primitive = "random_gauss_take", ly.runtime.result_contract = "builtins.float"} {
    %zero = arith.constant 0 : i64
    %nan = arith.constant 0x7FF8000000000000 : f64
    %has_slot = arith.constant 626 : index
    %bits_slot = arith.constant 627 : index
    %state = memref.get_global @__ly_random_state : memref<628xi64>
    %has = memref.load %state[%has_slot] : memref<628xi64>
    %cached = arith.cmpi ne, %has, %zero : i64
    %value = scf.if %cached -> f64 {
      %bits = memref.load %state[%bits_slot] : memref<628xi64>
      %as_float = arith.bitcast %bits : i64 to f64
      memref.store %zero, %state[%has_slot] : memref<628xi64>
      scf.yield %as_float : f64
    } else {
      scf.yield %nan : f64
    }
    %h = func.call @LyFloat_FromF64(%value) : (f64) -> memref<3xi64>
    func.return %h : memref<3xi64>
  }

  func.func @LyRandom_GaussPut(%header: memref<3xi64> {ly.ownership.object_header}) attributes {ly.runtime.builtin = "_random.gauss_put", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.float", ly.runtime.primitive = "random_gauss_put", ly.runtime.result_contract = "types.NoneType"} {
    %one = arith.constant 1 : i64
    %has_slot = arith.constant 626 : index
    %bits_slot = arith.constant 627 : index
    %value = func.call @LyFloat_AsF64(%header) : (memref<3xi64>) -> f64
    %bits = arith.bitcast %value : f64 to i64
    %state = memref.get_global @__ly_random_state : memref<628xi64>
    memref.store %bits, %state[%bits_slot] : memref<628xi64>
    memref.store %one, %state[%has_slot] : memref<628xi64>
    func.return
  }
}
