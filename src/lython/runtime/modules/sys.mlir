// Contract manifest for the statically supported `sys` surface (CPython's
// Python/sysmodule.c counterpart).
//
// Signature source (1:1 correspondence target):
//   https://github.com/python/typeshed/blob/main/stdlib/sys/__init__.pyi
//
// Only the target-independent constant surface lives here. Target-dependent
// members (sys.platform, sys.byteorder, sys.maxsize) and the zero-arg
// encoding callables (sys.getdefaultencoding, sys.getfilesystemencoding)
// are NOT declared in this manifest: their values depend on the target
// triple (or fold through the same machinery), which a build-time manifest
// cannot know, so they resolve through py::platform_constants
// (src/lython/common/PlatformConstants.h) against the compile-time triple.
//
// Deviations from CPython:
// - `version` reports the implemented language version with a "(lython)"
//   suffix instead of a build-host banner.
// - `copyright` is a single line (CPython's is multi-line).
// - `abiflags` is present on every target (CPython omits it on Windows).
// - `exit` requires an explicit int status (CPython accepts
//   `None | int | str` with a `None` default); it raises SystemExit through
//   the normal unwind path, so `finally` blocks run and `except SystemExit`
//   / `except BaseException` can observe it, and the top-level runner
//   converts an unhandled SystemExit into the process exit status.
// - `SystemExit(...)` constructed directly accepts only a str message (the
//   shared exception contract); an unhandled one prints the message to
//   stderr and exits 1, matching CPython's non-int code path. An int code
//   must go through sys.exit (the exception object has no code slot).
// - `argv` materializes a fresh list[str] on every read (there is no cached
//   module-attribute object); reads are value-identical to CPython's, and
//   mutating the temporary (`sys.argv.append(...)`) is rejected statically
//   by the structural-mutation receiver check instead of silently dropping
//   the write.
// - `stdout` / `stderr` are immortal _io.TextIOWrapper singletons exposing
//   only write(str) -> int and flush() (CPython's wrapper adds buffering
//   and reconfiguration, which the unbuffered fd write path does not
//   need). `stdin` is not supported.
// - `version_info`, `implementation`, `float_info`, `int_info` (structseq
//   attribute access), `stdin`, `path`, `modules`, and the
//   frame/trace/audit introspection surface are not yet supported;
//   referencing them is rejected at the static import boundary.

module attributes {
  ly.typing.module = "sys",
  ly.typing.callable_exports = ["sys.exit"],
  ly.typing.function_names = ["sys.exit"],
  ly.typing.function_contracts = [
    !py.callable<[!py.contract<"builtins.int">], arg_names = ["status"], arg_defaults = [false], returns = [!py.literal<None>]>
  ],
  ly.typing.int_constant_names = [
    "sys.hexversion",
    "sys.api_version",
    "sys.maxunicode"
  ],
  ly.typing.int_constant_values = [
    51249392 : i64,
    1013 : i64,
    1114111 : i64
  ],
  ly.typing.str_constant_names = [
    "sys.version",
    "sys.copyright",
    "sys.abiflags",
    "sys.float_repr_style"
  ],
  ly.typing.str_constant_values = [
    "3.14.0 (lython)",
    "Copyright (c) 2001-2026 Python Software Foundation. All Rights Reserved.",
    "",
    "short"
  ]
} {
  func.func private @LyLong_AsI64(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %digits: memref<?xi32>) -> i64 attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__int__", ly.runtime.primitive = "unbox.i64"}
  func.func private @LyList_FromLength(%length: i64 {ly.runtime.default_i64 = 0 : i64}) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 10 : i64, ly.runtime.contract = "builtins.list", ly.runtime.initializer = "__new__"}
  func.func private @LyUnicode_FromBytes(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 4 : i64, ly.runtime.contract = "builtins.str", ly.runtime.initializer = "__new__"}
  func.func private @LyHost_ArgvCount() -> i64
  func.func private @LyHost_ArgvLen(i64) -> i64
  func.func private @LyHost_ArgvCopy(i64, memref<?xi8>, i64)

  // stdout/stderr singleton instances of _io.TextIOWrapper (the class and
  // its methods live in modules/_io.mlir; CPython likewise creates the
  // wrapper instances during init_sys_streams and stores them on sys).
  // Layout mirrors _io.mlir: [refcount (immortal marker), class id, handle
  // (raw fd), kind (0 = fd), readable, writable, closed, reserved]. Generic
  // retains/releases may drift the marker but it never reaches zero, so the
  // deallocator's close-and-free path never runs for them.
  memref.global "private" @__ly_sys_stdout : memref<8xi64> = dense<[9223372036854775807, 65, 1, 0, 0, 1, 0, 0]>
  memref.global "private" @__ly_sys_stderr : memref<8xi64> = dense<[9223372036854775807, 65, 2, 0, 0, 1, 0, 0]>

  func.func @LySys_GetStdout() -> memref<8xi64> attributes {ly.runtime.contract = "_io.TextIOWrapper", ly.runtime.primitive = "sys_stdout"} {
    %singleton = memref.get_global @__ly_sys_stdout : memref<8xi64>
    func.return %singleton : memref<8xi64>
  }

  func.func @LySys_GetStderr() -> memref<8xi64> attributes {ly.runtime.contract = "_io.TextIOWrapper", ly.runtime.primitive = "sys_stderr"} {
    %singleton = memref.get_global @__ly_sys_stderr : memref<8xi64>
    func.return %singleton : memref<8xi64>
  }
  func.func private @LySystemExit_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 64 : i64, ly.runtime.contract = "builtins.SystemExit", ly.runtime.initializer = "__new__"}
  func.func private @LyEH_ThrowException(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "raise"}
  func.func private @LyHost_SetExitStatus(i64)

  // sys.argv builds a fresh list[str] from the process argument vector that
  // LyHost_InitArgs recorded before the program body ran. Each element is an
  // owned str stored as the standard 16-word collection payload handle
  // (CollectionPayload.cpp layout: [0] refcount, [1] class id, [2] header
  // pointer, [3] value count, [4..8] value pointers, [9..13] value sizes,
  // [14] owned flag).
  func.func @LySys_GetArgv() -> (memref<2xi64>, memref<2xi64>, memref<?xi64>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.list", ly.runtime.primitive = "sys_argv"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %header_class_slot = arith.constant 1 : index

    %count = func.call @LyHost_ArgvCount() : () -> i64
    %header, %meta, %items = func.call @LyList_FromLength(%count) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>)
    %count_index = arith.index_cast %count : i64 to index
    scf.for %i = %c0 to %count_index step %c1 {
      %i_i64 = arith.index_cast %i : index to i64
      %len = func.call @LyHost_ArgvLen(%i_i64) : (i64) -> i64
      %len_index = arith.index_cast %len : i64 to index
      %buffer = memref.alloc(%len_index) : memref<?xi8>
      func.call @LyHost_ArgvCopy(%i_i64, %buffer, %len) : (i64, memref<?xi8>, i64) -> ()
      %str_header, %str_bytes = func.call @LyUnicode_FromBytes(%buffer, %c0, %len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
      memref.dealloc %buffer : memref<?xi8>

      %base = arith.muli %i, %c16 : index
      scf.for %w = %c0 to %c16 step %c1 {
        %slot = arith.addi %base, %w : index
        memref.store %zero, %items[%slot] : memref<?xi64>
      }
      %class = memref.load %str_header[%header_class_slot] : memref<2xi64>
      %header_ptr_index = memref.extract_aligned_pointer_as_index %str_header : memref<2xi64> -> index
      %header_ptr = arith.index_cast %header_ptr_index : index to i64
      %bytes_ptr_index = memref.extract_aligned_pointer_as_index %str_bytes : memref<?xi8> -> index
      %bytes_ptr = arith.index_cast %bytes_ptr_index : index to i64

      %slot0 = arith.addi %base, %c0 : index
      memref.store %one, %items[%slot0] : memref<?xi64>
      %slot1 = arith.addi %base, %c1 : index
      memref.store %class, %items[%slot1] : memref<?xi64>
      %c2 = arith.constant 2 : index
      %slot2 = arith.addi %base, %c2 : index
      memref.store %header_ptr, %items[%slot2] : memref<?xi64>
      %c3 = arith.constant 3 : index
      %slot3 = arith.addi %base, %c3 : index
      memref.store %two, %items[%slot3] : memref<?xi64>
      %c4 = arith.constant 4 : index
      %slot4 = arith.addi %base, %c4 : index
      memref.store %header_ptr, %items[%slot4] : memref<?xi64>
      %c5 = arith.constant 5 : index
      %slot5 = arith.addi %base, %c5 : index
      memref.store %bytes_ptr, %items[%slot5] : memref<?xi64>
      %c9 = arith.constant 9 : index
      %slot9 = arith.addi %base, %c9 : index
      memref.store %two, %items[%slot9] : memref<?xi64>
      %c10 = arith.constant 10 : index
      %slot10 = arith.addi %base, %c10 : index
      memref.store %len, %items[%slot10] : memref<?xi64>
      %c14 = arith.constant 14 : index
      %slot14 = arith.addi %base, %c14 : index
      memref.store %one, %items[%slot14] : memref<?xi64>
    }
    func.return %header, %meta, %items : memref<2xi64>, memref<2xi64>, memref<?xi64>
  }

  // sys.exit records the status out-of-band (the exception object layout has
  // no payload slot beyond the message string) and raises SystemExit with an
  // empty message; LyRunPythonMain maps that back to the process exit status.
  func.func @LySys_Exit(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %digits: memref<?xi32>) attributes {ly.runtime.builtin = "sys.exit", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.int", ly.runtime.primitive = "sys_exit", ly.runtime.result_contract = "types.NoneType"} {
    %status = func.call @LyLong_AsI64(%header, %meta, %digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    func.call @LyHost_SetExitStatus(%status) : (i64) -> ()
    %class_id = arith.constant 64 : i64
    %exception:3 = func.call @LySystemExit_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.call @LyEH_ThrowException(%exception#0, %exception#1, %exception#2) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }
}
