// Contract manifest AND runtime implementation for the statically supported
// clock and calendar natives (CPython's Modules/timemodule.c counterpart).
//
// Signature sources (1:1 correspondence target):
//   https://github.com/python/typeshed/blob/main/stdlib/time.pyi
//   https://github.com/python/cpython/blob/main/Modules/timemodule.c
//
// WHY THIS IS `_time` AND NOT `time`, when CPython implements all of `time`
// in C: `time.struct_time` is a structseq -- a tuple subclass with named
// fields -- and a manifest cannot declare a class with named int fields yet.
// So the module is split the way CPython splits io: this file is the native
// half and lib/time.py is the thin public layer that owns `struct_time` and
// the calls that return one. CPython has no `_time` module; the split is
// Lython's, and lib/time.py's docstring records it.
//
// The libc boundary is the OS support cluster
// (lowering/Common/OsSupportBuilder.cpp): CLOCK_MONOTONIC's value differs per
// platform and this manifest is embedded as target-INDEPENDENT bytecode, so
// the clock reads go through LyHost_ClockNs. `struct tm`'s first nine members
// are `int tm_sec` through `int tm_isdst` on every POSIX libc, which is why
// the calendar calls can pass them as nine plain ints.
//
// Deviations from CPython:
// - the calendar fields cross as SCALARS, one call per field
//   (`_field(seconds, utc, index)`), rather than as a struct. lib/time.py
//   assembles struct_time from ten such reads, so localtime() costs ten
//   localtime_r calls. A list[int] would need the collection payload built
//   and re-read by hand on both sides for one caller.
// - `time()` / `monotonic()` / `perf_counter()` are f64 seconds derived from
//   the same nanosecond reads their `*_ns` forms return, as CPython's are.
// - `perf_counter` is CLOCK_MONOTONIC. CPython picks the platform's
//   highest-resolution counter, which on Linux is the same clock; the
//   documented guarantee (monotonic, unspecified reference point) holds.
// - `sleep()` does not retry on EINTR (PEP 475): an interrupted sleep raises
//   InterruptedError through the errno taxonomy, which is CPython's pre-3.5
//   behaviour, so `except InterruptedError` can observe it.
// - `_strftime` formats through libc strftime in the process locale and
//   reports a result longer than 1023 bytes as ''. CPython grows the buffer
//   and also rejects some formats up front.
// - `clock_gettime`, `process_time`, `thread_time`, `get_clock_info` and
//   `tzset` are not exported.

module attributes {
  ly.typing.module = "_time",
  ly.typing.callable_exports = [
    "_time.time",
    "_time.time_ns",
    "_time.monotonic",
    "_time.monotonic_ns",
    "_time.perf_counter",
    "_time.perf_counter_ns",
    "_time.sleep",
    "_time.field",
    "_time.strftime",
    "_time.mktime"
  ],
  ly.typing.function_names = [
    "_time.time",
    "_time.time_ns",
    "_time.monotonic",
    "_time.monotonic_ns",
    "_time.perf_counter",
    "_time.perf_counter_ns",
    "_time.sleep",
    "_time.field",
    "_time.strftime",
    "_time.mktime"
  ],
  ly.typing.function_contracts = [
    !py.callable<[], arg_names = [], arg_defaults = [], returns = [!py.contract<"builtins.float">]>,
    !py.callable<[], arg_names = [], arg_defaults = [], returns = [!py.contract<"builtins.int">]>,
    !py.callable<[], arg_names = [], arg_defaults = [], returns = [!py.contract<"builtins.float">]>,
    !py.callable<[], arg_names = [], arg_defaults = [], returns = [!py.contract<"builtins.int">]>,
    !py.callable<[], arg_names = [], arg_defaults = [], returns = [!py.contract<"builtins.float">]>,
    !py.callable<[], arg_names = [], arg_defaults = [], returns = [!py.contract<"builtins.int">]>,
    !py.callable<[!py.contract<"builtins.float">], arg_names = ["seconds"], arg_defaults = [false], returns = [!py.literal<None>]>,
    !py.callable<[!py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">], arg_names = ["seconds", "utc", "index"], arg_defaults = [false, false, false], returns = [!py.contract<"builtins.int">]>,
    !py.callable<[!py.contract<"builtins.str">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">], arg_names = ["format", "sec", "min", "hour", "mday", "mon", "year", "wday", "yday", "isdst"], arg_defaults = [false, false, false, false, false, false, false, false, false, false], returns = [!py.contract<"builtins.str">]>,
    !py.callable<[!py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">], arg_names = ["sec", "min", "hour", "mday", "mon", "year", "wday", "yday", "isdst"], arg_defaults = [false, false, false, false, false, false, false, false, false], returns = [!py.contract<"builtins.int">]>
  ]
} {
  // --- shared runtime entry points -----------------------------------------
  func.func private @LyLong_FromI64(%value: i64 {ly.runtime.default_i64 = 0 : i64}) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 1 : i64, ly.runtime.contract = "builtins.int", ly.runtime.initializer = "__new__"}
  func.func private @LyFloat_FromF64(%value: f64 {ly.runtime.default_f64 = 0.0 : f64}) -> (memref<2xi64>, memref<1xf64>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 2 : i64, ly.runtime.contract = "builtins.float", ly.runtime.initializer = "__new__"}
  func.func private @LyFloat_AsF64(%header: memref<2xi64> {ly.ownership.object_header}, %payload: memref<1xf64>) -> f64 attributes {ly.runtime.contract = "builtins.float", ly.runtime.method = "__float__", ly.runtime.primitive = "unbox.f64"}
  func.func private @LyUnicode_FromBytes(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 4 : i64, ly.runtime.contract = "builtins.str", ly.runtime.initializer = "__new__"}
  func.func private @LyUnicode_Encode(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> memref<6xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "encode", ly.runtime.result_contract = "builtins.bytes"}
  func.func private @LyBytes_DecRef(%header: memref<6xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.deallocator}
  func.func private @__ly_bytes_payload(%self: memref<6xi64>) -> memref<?xi8> attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.interior_word, ly.runtime.primitive = "payload_view"}
  func.func private @LyBaseException_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 5 : i64, ly.runtime.contract = "builtins.BaseException", ly.runtime.initializer = "__new__"}
  func.func private @LyBaseException_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.BaseException", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"}
  func.func private @LyEH_ThrowException(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "raise"}

  // --- host boundary -------------------------------------------------------
  func.func private @LyHost_ClockNs(i64) -> i64
  func.func private @LyHost_SleepNs(i64) -> i32
  func.func private @LyHost_TimeFields(i64, i64, memref<?xi64>) -> i32
  func.func private @LyHost_Strftime(memref<?xi8>, i64, memref<?xi64>, memref<?xi8>, i64) -> i64
  func.func private @LyHost_Mktime(memref<?xi64>) -> i64
  func.func private @LyHost_Errno() -> i32
  func.func private @LyHost_OSErrorClassId(i32) -> i64
  func.func private @LyHost_OSErrorMessage(i32, memref<?xi8>, i64) -> i64

  // --- diagnostics ---------------------------------------------------------
  // "sleep length must be non-negative" and "year out of range", CPython's own
  // ValueError texts.
  memref.global "private" constant @__ly_time_msg_neg_sleep : memref<33xi8> = dense<[115, 108, 101, 101, 112, 32, 108, 101, 110, 103, 116, 104, 32, 109, 117, 115, 116, 32, 98, 101, 32, 110, 111, 110, 45, 110, 101, 103, 97, 116, 105, 118, 101]>
  memref.global "private" constant @__ly_time_msg_bad_time : memref<17xi8> = dense<[121, 101, 97, 114, 32, 111, 117, 116, 32, 111, 102, 32, 114, 97, 110, 103, 101]>

  func.func private @__ly_time_throw(%class_id: i64, %message: memref<?xi8>, %length: i64) {
    %c0 = arith.constant 0 : index
    %message_header, %message_bytes = func.call @LyUnicode_FromBytes(%message, %c0, %length) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %exception:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    %initialized:3 = func.call @LyBaseException_Init(%exception#0, %exception#1, %exception#2, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.call @LyEH_ThrowException(%initialized#0, %initialized#1, %initialized#2) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @__ly_time_raise_neg_sleep() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c33 = arith.constant 33 : index
    %len = arith.constant 33 : i64
    %class_id = arith.constant 53 : i64
    %buffer = memref.alloc(%c33) : memref<?xi8>
    %text = memref.get_global @__ly_time_msg_neg_sleep : memref<33xi8>
    scf.for %i = %c0 to %c33 step %c1 {
      %byte = memref.load %text[%i] : memref<33xi8>
      memref.store %byte, %buffer[%i] : memref<?xi8>
    }
    func.call @__ly_time_throw(%class_id, %buffer, %len) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @__ly_time_raise_bad_time() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c17 = arith.constant 17 : index
    %len = arith.constant 17 : i64
    %class_id = arith.constant 53 : i64
    %buffer = memref.alloc(%c17) : memref<?xi8>
    %text = memref.get_global @__ly_time_msg_bad_time : memref<17xi8>
    scf.for %i = %c0 to %c17 step %c1 {
      %byte = memref.load %text[%i] : memref<17xi8>
      memref.store %byte, %buffer[%i] : memref<?xi8>
    }
    func.call @__ly_time_throw(%class_id, %buffer, %len) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  // The errno-mapped OSError, with no filename: no path is involved in a
  // clock or a sleep failure.
  func.func private @__ly_time_raise_errno(%err: i32) {
    %cap_index = arith.constant 256 : index
    %cap = arith.constant 256 : i64
    %class_id = func.call @LyHost_OSErrorClassId(%err) : (i32) -> i64
    %buffer = memref.alloc(%cap_index) : memref<?xi8>
    %len = func.call @LyHost_OSErrorMessage(%err, %buffer, %cap) : (i32, memref<?xi8>, i64) -> i64
    func.call @__ly_time_throw(%class_id, %buffer, %len) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  // --- clocks --------------------------------------------------------------

  func.func @LyTime_TimeNs() -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "_time.time_ns", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.int", ly.runtime.primitive = "time_time_ns", ly.runtime.result_contract = "builtins.int"} {
    %realtime = arith.constant 0 : i64
    %value = func.call @LyHost_ClockNs(%realtime) : (i64) -> i64
    %h, %m, %d = func.call @LyLong_FromI64(%value) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyTime_MonotonicNs() -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "_time.monotonic_ns", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.int", ly.runtime.primitive = "time_monotonic_ns", ly.runtime.result_contract = "builtins.int"} {
    %monotonic = arith.constant 1 : i64
    %value = func.call @LyHost_ClockNs(%monotonic) : (i64) -> i64
    %h, %m, %d = func.call @LyLong_FromI64(%value) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyTime_PerfCounterNs() -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "_time.perf_counter_ns", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.int", ly.runtime.primitive = "time_perf_counter_ns", ly.runtime.result_contract = "builtins.int"} {
    %monotonic = arith.constant 1 : i64
    %value = func.call @LyHost_ClockNs(%monotonic) : (i64) -> i64
    %h, %m, %d = func.call @LyLong_FromI64(%value) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyTime_Time() -> (memref<2xi64>, memref<1xf64>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "_time.time", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.float", ly.runtime.primitive = "time_time", ly.runtime.result_contract = "builtins.float"} {
    %realtime = arith.constant 0 : i64
    %billion = arith.constant 1000000000.0 : f64
    %nanos = func.call @LyHost_ClockNs(%realtime) : (i64) -> i64
    %as_float = arith.sitofp %nanos : i64 to f64
    %seconds = arith.divf %as_float, %billion : f64
    %h, %p = func.call @LyFloat_FromF64(%seconds) : (f64) -> (memref<2xi64>, memref<1xf64>)
    func.return %h, %p : memref<2xi64>, memref<1xf64>
  }

  func.func @LyTime_Monotonic() -> (memref<2xi64>, memref<1xf64>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "_time.monotonic", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.float", ly.runtime.primitive = "time_monotonic", ly.runtime.result_contract = "builtins.float"} {
    %monotonic = arith.constant 1 : i64
    %billion = arith.constant 1000000000.0 : f64
    %nanos = func.call @LyHost_ClockNs(%monotonic) : (i64) -> i64
    %as_float = arith.sitofp %nanos : i64 to f64
    %seconds = arith.divf %as_float, %billion : f64
    %h, %p = func.call @LyFloat_FromF64(%seconds) : (f64) -> (memref<2xi64>, memref<1xf64>)
    func.return %h, %p : memref<2xi64>, memref<1xf64>
  }

  func.func @LyTime_PerfCounter() -> (memref<2xi64>, memref<1xf64>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "_time.perf_counter", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.float", ly.runtime.primitive = "time_perf_counter", ly.runtime.result_contract = "builtins.float"} {
    %monotonic = arith.constant 1 : i64
    %billion = arith.constant 1000000000.0 : f64
    %nanos = func.call @LyHost_ClockNs(%monotonic) : (i64) -> i64
    %as_float = arith.sitofp %nanos : i64 to f64
    %seconds = arith.divf %as_float, %billion : f64
    %h, %p = func.call @LyFloat_FromF64(%seconds) : (f64) -> (memref<2xi64>, memref<1xf64>)
    func.return %h, %p : memref<2xi64>, memref<1xf64>
  }

  func.func @LyTime_Sleep(%header: memref<2xi64> {ly.ownership.object_header}, %payload: memref<1xf64>) attributes {ly.runtime.builtin = "_time.sleep", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.float", ly.runtime.primitive = "time_sleep", ly.runtime.result_contract = "types.NoneType"} {
    %zero = arith.constant 0.0 : f64
    %zero32 = arith.constant 0 : i32
    %billion = arith.constant 1000000000.0 : f64
    %seconds = func.call @LyFloat_AsF64(%header, %payload) : (memref<2xi64>, memref<1xf64>) -> f64
    %negative = arith.cmpf olt, %seconds, %zero : f64
    scf.if %negative {
      func.call @__ly_time_raise_neg_sleep() : () -> ()
    }
    %nanos_f = arith.mulf %seconds, %billion : f64
    %nanos = arith.fptosi %nanos_f : f64 to i64
    %status = func.call @LyHost_SleepNs(%nanos) : (i64) -> i32
    %failed = arith.cmpi ne, %status, %zero32 : i32
    scf.if %failed {
      %err = func.call @LyHost_Errno() : () -> i32
      func.call @__ly_time_raise_errno(%err) : (i32) -> ()
    }
    func.return
  }

  // --- calendar ------------------------------------------------------------

  // One `struct tm` field of the broken-down time: index 0..8 are tm_sec,
  // tm_min, tm_hour, tm_mday, tm_mon, tm_year, tm_wday, tm_yday, tm_isdst
  // (raw libc values -- tm_mon 0-based, tm_year since 1900) and index 9 is
  // tm_gmtoff. `utc` picks gmtime_r over localtime_r.
  func.func @LyTime_Field(%seconds: i64, %utc: i64, %index: i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "_time.field", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.int", ly.runtime.primitive = "time_field", ly.runtime.result_contract = "builtins.int"} {
    %c10 = arith.constant 10 : index
    %zero32 = arith.constant 0 : i32
    %zero = arith.constant 0 : i64
    %nine = arith.constant 9 : i64
    %fields = memref.alloc(%c10) : memref<?xi64>
    %status = func.call @LyHost_TimeFields(%seconds, %utc, %fields) : (i64, i64, memref<?xi64>) -> i32
    %failed = arith.cmpi ne, %status, %zero32 : i32
    scf.if %failed {
      func.call @__ly_time_raise_bad_time() : () -> ()
    }
    %low = arith.maxsi %index, %zero : i64
    %clamped = arith.minsi %low, %nine : i64
    %slot = arith.index_cast %clamped : i64 to index
    %value = memref.load %fields[%slot] : memref<?xi64>
    memref.dealloc %fields : memref<?xi64>
    %h, %m, %d = func.call @LyLong_FromI64(%value) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  // The inverse: local-time epoch seconds from the nine raw fields. isdst
  // passes through, so -1 asks the platform to guess as CPython's mktime does.
  func.func @LyTime_Mktime(%sec: i64, %min: i64, %hour: i64, %mday: i64, %mon: i64, %year: i64, %wday: i64, %yday: i64, %isdst: i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "_time.mktime", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.int", ly.runtime.primitive = "time_mktime", ly.runtime.result_contract = "builtins.int"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index
    %c8 = arith.constant 8 : index
    %c9 = arith.constant 9 : index
    %minus_one = arith.constant -1 : i64
    %fields = memref.alloc(%c9) : memref<?xi64>
    memref.store %sec, %fields[%c0] : memref<?xi64>
    memref.store %min, %fields[%c1] : memref<?xi64>
    memref.store %hour, %fields[%c2] : memref<?xi64>
    memref.store %mday, %fields[%c3] : memref<?xi64>
    memref.store %mon, %fields[%c4] : memref<?xi64>
    memref.store %year, %fields[%c5] : memref<?xi64>
    memref.store %wday, %fields[%c6] : memref<?xi64>
    memref.store %yday, %fields[%c7] : memref<?xi64>
    memref.store %isdst, %fields[%c8] : memref<?xi64>
    %seconds = func.call @LyHost_Mktime(%fields) : (memref<?xi64>) -> i64
    memref.dealloc %fields : memref<?xi64>
    %failed = arith.cmpi eq, %seconds, %minus_one : i64
    scf.if %failed {
      func.call @__ly_time_raise_bad_time() : () -> ()
    }
    %h, %m, %d = func.call @LyLong_FromI64(%seconds) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyTime_Strftime(%fmt_header: memref<2xi64> {ly.ownership.object_header}, %fmt_bytes: memref<?xi8>, %sec: i64, %min: i64, %hour: i64, %mday: i64, %mon: i64, %year: i64, %wday: i64, %yday: i64, %isdst: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "_time.strftime", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.str", ly.runtime.primitive = "time_strftime", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index
    %c8 = arith.constant 8 : index
    %c9 = arith.constant 9 : index
    %cap_index = arith.constant 1024 : index
    %cap = arith.constant 1024 : i64
    %zero = arith.constant 0 : i64
    %enc_header = func.call @LyUnicode_Encode(%fmt_header, %fmt_bytes) : (memref<2xi64>, memref<?xi8>) -> memref<6xi64>
    %enc_bytes = func.call @__ly_bytes_payload(%enc_header) : (memref<6xi64>) -> memref<?xi8>
    %enc_dim = memref.dim %enc_bytes, %c0 : memref<?xi8>
    %enc_len = arith.index_cast %enc_dim : index to i64
    %fields = memref.alloc(%c9) : memref<?xi64>
    memref.store %sec, %fields[%c0] : memref<?xi64>
    memref.store %min, %fields[%c1] : memref<?xi64>
    memref.store %hour, %fields[%c2] : memref<?xi64>
    memref.store %mday, %fields[%c3] : memref<?xi64>
    memref.store %mon, %fields[%c4] : memref<?xi64>
    memref.store %year, %fields[%c5] : memref<?xi64>
    memref.store %wday, %fields[%c6] : memref<?xi64>
    memref.store %yday, %fields[%c7] : memref<?xi64>
    memref.store %isdst, %fields[%c8] : memref<?xi64>
    %buffer = memref.alloc(%cap_index) : memref<?xi8>
    %len = func.call @LyHost_Strftime(%enc_bytes, %enc_len, %fields, %buffer, %cap) : (memref<?xi8>, i64, memref<?xi64>, memref<?xi8>, i64) -> i64
    memref.dealloc %fields : memref<?xi64>
    func.call @LyBytes_DecRef(%enc_header) : (memref<6xi64>) -> ()
    %clamped_high = arith.minsi %len, %cap : i64
    %clamped = arith.maxsi %clamped_high, %zero : i64
    %out_header, %out_bytes = func.call @LyUnicode_FromBytes(%buffer, %c0, %clamped) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    memref.dealloc %buffer : memref<?xi8>
    func.return %out_header, %out_bytes : memref<2xi64>, memref<?xi8>
  }
}
