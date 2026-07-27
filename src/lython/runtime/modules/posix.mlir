// Contract manifest AND runtime implementation for the statically supported
// `posix` surface (CPython's Modules/posixmodule.c counterpart). `os.py`
// re-exports it with `from posix import *`, exactly as CPython's does.
//
// Signature sources (1:1 correspondence target):
//   https://github.com/python/typeshed/blob/main/stdlib/posix.pyi
//   https://github.com/python/typeshed/blob/main/stdlib/os/__init__.pyi
//   https://github.com/python/cpython/blob/main/Modules/posixmodule.c
//
// The libc boundary is the OS support cluster
// (lowering/Common/OsSupportBuilder.cpp): this manifest is embedded as
// target-INDEPENDENT bytecode, so every routine that needs a `struct stat`
// offset, the errno accessor's symbol, or a $INODE64 variant calls a
// LyHost_* wrapper instead of libc directly.
//
// Deviations from CPython:
// - `environ` is absent. It is a computed dict, and a container-typed module
//   global is not visible across an import boundary yet (reported to the
//   Wave 3 foundation track); `_environ_entries()` hands os.py the raw
//   "KEY=VALUE" vector for when it becomes expressible, and
//   getenv/putenv/unsetenv go straight to the process environment meanwhile.
// - `stat()` is not a structseq here. `_stat_field(path, follow, index)`
//   returns ONE os.stat_result field (or `-errno`, never raising), and os.py
//   assembles the `stat_result` class from it. os.path's predicates need only
//   index 0 (st_mode), so the common case is one syscall; `os.stat()` costs
//   one syscall per field read.
// - error reporting is split: the calls that must fail loudly raise from
//   here through `_raise_errno`, which maps errno to the OSError subclass
//   with the compiler's own kOSErrorErrnoMap table (LyHost_OSErrorClassId)
//   and formats CPython's "[Errno %d] %s: '%s'" message; the predicate-shaped
//   calls return `-errno` and let os.py decide.
// - `mkdir` takes no `dir_fd`, `unlink` no `dir_fd`, `access` no
//   `effective_ids`/`follow_symlinks`: the *at() family and the fd-relative
//   keyword arguments are not supported.
// - path arguments are `str` only (no bytes paths, no PathLike protocol:
//   pathlib passes `str(path)`), and are encoded as UTF-8 -- the filesystem
//   encoding is not configurable.
// - on Windows targets every entry point below still compiles (the manifest
//   is target-independent) but the underlying libc names do not exist there;
//   `os.py` imports `nt` instead, as CPython does.

module attributes {
  ly.typing.module = "posix",
  ly.typing.callable_exports = [
    "posix.getpid",
    "posix.getppid",
    "posix.getuid",
    "posix.geteuid",
    "posix.getgid",
    "posix.getegid",
    "posix.getcwd",
    "posix.chdir",
    "posix.mkdir",
    "posix.rmdir",
    "posix.unlink",
    "posix.rename",
    "posix.access",
    "posix.listdir",
    "posix.strerror",
    "posix.putenv",
    "posix.unsetenv",
    "posix._getenv",
    "posix._has_env",
    "posix._stat_field",
    "posix._raise_errno",
    "posix._environ_entries"
  ],
  ly.typing.function_names = [
    "posix.getpid",
    "posix.getppid",
    "posix.getuid",
    "posix.geteuid",
    "posix.getgid",
    "posix.getegid",
    "posix.getcwd",
    "posix.chdir",
    "posix.mkdir",
    "posix.rmdir",
    "posix.unlink",
    "posix.rename",
    "posix.access",
    "posix.listdir",
    "posix.strerror",
    "posix.putenv",
    "posix.unsetenv",
    "posix._getenv",
    "posix._has_env",
    "posix._stat_field",
    "posix._raise_errno",
    "posix._environ_entries"
  ],
  ly.typing.function_contracts = [
    !py.callable<[], arg_names = [], arg_defaults = [], returns = [!py.contract<"builtins.int">]>,
    !py.callable<[], arg_names = [], arg_defaults = [], returns = [!py.contract<"builtins.int">]>,
    !py.callable<[], arg_names = [], arg_defaults = [], returns = [!py.contract<"builtins.int">]>,
    !py.callable<[], arg_names = [], arg_defaults = [], returns = [!py.contract<"builtins.int">]>,
    !py.callable<[], arg_names = [], arg_defaults = [], returns = [!py.contract<"builtins.int">]>,
    !py.callable<[], arg_names = [], arg_defaults = [], returns = [!py.contract<"builtins.int">]>,
    !py.callable<[], arg_names = [], arg_defaults = [], returns = [!py.contract<"builtins.str">]>,
    !py.callable<[!py.contract<"builtins.str">], arg_names = ["path"], arg_defaults = [false], returns = [!py.literal<None>]>,
    !py.callable<[!py.contract<"builtins.str">, !py.contract<"builtins.int">], arg_names = ["path", "mode"], arg_defaults = [false, true], returns = [!py.literal<None>]>,
    !py.callable<[!py.contract<"builtins.str">], arg_names = ["path"], arg_defaults = [false], returns = [!py.literal<None>]>,
    !py.callable<[!py.contract<"builtins.str">], arg_names = ["path"], arg_defaults = [false], returns = [!py.literal<None>]>,
    !py.callable<[!py.contract<"builtins.str">, !py.contract<"builtins.str">], arg_names = ["src", "dst"], arg_defaults = [false, false], returns = [!py.literal<None>]>,
    !py.callable<[!py.contract<"builtins.str">, !py.contract<"builtins.int">], arg_names = ["path", "mode"], arg_defaults = [false, false], returns = [!py.contract<"builtins.bool">]>,
    !py.callable<[!py.contract<"builtins.str">], arg_names = ["path"], arg_defaults = [false], returns = [!py.contract<"builtins.list", [!py.contract<"builtins.str">]>]>,
    !py.callable<[!py.contract<"builtins.int">], arg_names = ["code"], arg_defaults = [false], returns = [!py.contract<"builtins.str">]>,
    !py.callable<[!py.contract<"builtins.str">, !py.contract<"builtins.str">], arg_names = ["name", "value"], arg_defaults = [false, false], returns = [!py.literal<None>]>,
    !py.callable<[!py.contract<"builtins.str">], arg_names = ["name"], arg_defaults = [false], returns = [!py.literal<None>]>,
    !py.callable<[!py.contract<"builtins.str">], arg_names = ["name"], arg_defaults = [false], returns = [!py.contract<"builtins.str">]>,
    !py.callable<[!py.contract<"builtins.str">], arg_names = ["name"], arg_defaults = [false], returns = [!py.contract<"builtins.bool">]>,
    !py.callable<[!py.contract<"builtins.str">, !py.contract<"builtins.int">, !py.contract<"builtins.int">], arg_names = ["path", "follow", "index"], arg_defaults = [false, false, false], returns = [!py.contract<"builtins.int">]>,
    !py.callable<[!py.contract<"builtins.int">, !py.contract<"builtins.str">], arg_names = ["code", "path"], arg_defaults = [false, false], returns = [!py.literal<None>]>,
    !py.callable<[], arg_names = [], arg_defaults = [], returns = [!py.contract<"builtins.list", [!py.contract<"builtins.str">]>]>
  ],
  ly.typing.int_constant_names = [
    "posix.F_OK",
    "posix.R_OK",
    "posix.W_OK",
    "posix.X_OK"
  ],
  ly.typing.int_constant_values = [0 : i64, 4 : i64, 2 : i64, 1 : i64]
} {
  // --- shared runtime entry points -----------------------------------------
  func.func private @LyLong_FromI64(%value: i64 {ly.runtime.default_i64 = 0 : i64}) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 1 : i64, ly.runtime.contract = "builtins.int", ly.runtime.initializer = "__new__"}
  func.func private @LyUnicode_FromBytes(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 4 : i64, ly.runtime.contract = "builtins.str", ly.runtime.initializer = "__new__"}
  func.func private @LyUnicode_Encode(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> memref<6xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "encode", ly.runtime.result_contract = "builtins.bytes"}
  func.func private @LyBytes_DecRef(%header: memref<6xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.deallocator}
  func.func private @__ly_bytes_payload(%self: memref<6xi64>) -> memref<?xi8> attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.interior_word, ly.runtime.primitive = "payload_view"}
  func.func private @LyList_FromLength(%length: i64 {ly.runtime.default_i64 = 0 : i64}) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 10 : i64, ly.runtime.contract = "builtins.list", ly.runtime.initializer = "__new__"}
  func.func private @LyBaseException_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 5 : i64, ly.runtime.contract = "builtins.BaseException", ly.runtime.initializer = "__new__"}
  func.func private @LyBaseException_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.BaseException", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"}
  func.func private @LyEH_ThrowException(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "raise"}

  // --- host boundary -------------------------------------------------------
  func.func private @LyHost_GetPid() -> i64
  func.func private @LyHost_GetPPid() -> i64
  func.func private @LyHost_GetUid() -> i64
  func.func private @LyHost_GetEUid() -> i64
  func.func private @LyHost_GetGid() -> i64
  func.func private @LyHost_GetEGid() -> i64
  func.func private @LyHost_Errno() -> i32
  func.func private @LyHost_OSErrorClassId(i32) -> i64
  func.func private @LyHost_OSErrorMessagePath(i32, memref<?xi8>, i64, memref<?xi8>, i64) -> i64
  func.func private @LyHost_OSErrorMessagePath2(i32, memref<?xi8>, i64, memref<?xi8>, i64, memref<?xi8>, i64) -> i64
  func.func private @LyHost_Strerror(i32, memref<?xi8>, i64) -> i64
  func.func private @LyHost_GetCwd(memref<?xi8>, i64) -> i64
  func.func private @LyHost_Chdir(memref<?xi8>, i64) -> i32
  func.func private @LyHost_Rmdir(memref<?xi8>, i64) -> i32
  func.func private @LyHost_Unlink(memref<?xi8>, i64) -> i32
  func.func private @LyHost_Mkdir(memref<?xi8>, i64, i64) -> i32
  func.func private @LyHost_Access(memref<?xi8>, i64, i64) -> i32
  func.func private @LyHost_Rename(memref<?xi8>, i64, memref<?xi8>, i64) -> i32
  func.func private @LyHost_Stat(memref<?xi8>, i64, memref<?xi64>) -> i32
  func.func private @LyHost_LStat(memref<?xi8>, i64, memref<?xi64>) -> i32
  func.func private @LyHost_OpenDir(memref<?xi8>, i64) -> i64
  func.func private @LyHost_ReadDirName(i64, memref<?xi8>, i64) -> i64
  func.func private @LyHost_CloseDir(i64) -> i32
  func.func private @LyHost_GetEnv(memref<?xi8>, i64, memref<?xi8>, i64) -> i64
  func.func private @LyHost_SetEnv(memref<?xi8>, i64, memref<?xi8>, i64) -> i32
  func.func private @LyHost_UnsetEnv(memref<?xi8>, i64) -> i32
  func.func private @LyHost_EnvironCount() -> i64
  func.func private @LyHost_EnvironLen(i64) -> i64
  func.func private @LyHost_EnvironCopy(i64, memref<?xi8>, i64)

  // --- internal helpers ----------------------------------------------------

  // Raises the OSError subclass errno maps to, with CPython's message. The
  // class id comes from the compiler's kOSErrorErrnoMap through
  // LyHost_OSErrorClassId, so `except FileNotFoundError` matches here exactly
  // as it does for a hand-raised one.
  func.func private @__ly_posix_throw(%err: i32, %path: memref<?xi8>, %path_len: i64) {
    %c0 = arith.constant 0 : index
    %cap_index = arith.constant 1024 : index
    %cap = arith.constant 1024 : i64
    %class_id = func.call @LyHost_OSErrorClassId(%err) : (i32) -> i64
    %buffer = memref.alloc(%cap_index) : memref<?xi8>
    %len = func.call @LyHost_OSErrorMessagePath(%err, %path, %path_len, %buffer, %cap) : (i32, memref<?xi8>, i64, memref<?xi8>, i64) -> i64
    %message_header, %message_bytes = func.call @LyUnicode_FromBytes(%buffer, %c0, %len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    memref.dealloc %buffer : memref<?xi8>
    %exception:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    %initialized:3 = func.call @LyBaseException_Init(%exception#0, %exception#1, %exception#2, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.call @LyEH_ThrowException(%initialized#0, %initialized#1, %initialized#2) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  // The two-path variant, for rename's "'src' -> 'dst'" message.
  func.func private @__ly_posix_throw2(%err: i32, %src: memref<?xi8>, %src_len: i64, %dst: memref<?xi8>, %dst_len: i64) {
    %c0 = arith.constant 0 : index
    %cap_index = arith.constant 1024 : index
    %cap = arith.constant 1024 : i64
    %class_id = func.call @LyHost_OSErrorClassId(%err) : (i32) -> i64
    %buffer = memref.alloc(%cap_index) : memref<?xi8>
    %len = func.call @LyHost_OSErrorMessagePath2(%err, %src, %src_len, %dst, %dst_len, %buffer, %cap) : (i32, memref<?xi8>, i64, memref<?xi8>, i64, memref<?xi8>, i64) -> i64
    %message_header, %message_bytes = func.call @LyUnicode_FromBytes(%buffer, %c0, %len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    memref.dealloc %buffer : memref<?xi8>
    %exception:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    %initialized:3 = func.call @LyBaseException_Init(%exception#0, %exception#1, %exception#2, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.call @LyEH_ThrowException(%initialized#0, %initialized#1, %initialized#2) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  // Writes one owned str into a list payload slot. The 16-word slot layout is
  // CollectionPayload.cpp's: [0] refcount, [1] class id, [2] header pointer,
  // [3] value count, [4..8] value pointers, [9..13] value sizes, [14] owned.
  func.func private @__ly_posix_store_str(%items: memref<?xi64>, %index: index, %str_header: memref<2xi64>, %str_bytes: memref<?xi8>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c9 = arith.constant 9 : index
    %c10 = arith.constant 10 : index
    %c14 = arith.constant 14 : index
    %c16 = arith.constant 16 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %class_slot = arith.constant 1 : index

    %base = arith.muli %index, %c16 : index
    scf.for %w = %c0 to %c16 step %c1 {
      %slot = arith.addi %base, %w : index
      memref.store %zero, %items[%slot] : memref<?xi64>
    }
    %class = memref.load %str_header[%class_slot] : memref<2xi64>
    %header_ptr_index = memref.extract_aligned_pointer_as_index %str_header : memref<2xi64> -> index
    %header_ptr = arith.index_cast %header_ptr_index : index to i64
    %bytes_ptr_index = memref.extract_aligned_pointer_as_index %str_bytes : memref<?xi8> -> index
    %bytes_ptr = arith.index_cast %bytes_ptr_index : index to i64
    %bytes_dim = memref.dim %str_bytes, %c0 : memref<?xi8>
    %bytes_len = arith.index_cast %bytes_dim : index to i64

    %slot0 = arith.addi %base, %c0 : index
    memref.store %one, %items[%slot0] : memref<?xi64>
    %slot1 = arith.addi %base, %c1 : index
    memref.store %class, %items[%slot1] : memref<?xi64>
    %slot2 = arith.addi %base, %c2 : index
    memref.store %header_ptr, %items[%slot2] : memref<?xi64>
    %slot3 = arith.addi %base, %c3 : index
    memref.store %two, %items[%slot3] : memref<?xi64>
    %slot4 = arith.addi %base, %c4 : index
    memref.store %header_ptr, %items[%slot4] : memref<?xi64>
    %slot5 = arith.addi %base, %c5 : index
    memref.store %bytes_ptr, %items[%slot5] : memref<?xi64>
    %slot9 = arith.addi %base, %c9 : index
    memref.store %two, %items[%slot9] : memref<?xi64>
    %slot10 = arith.addi %base, %c10 : index
    memref.store %bytes_len, %items[%slot10] : memref<?xi64>
    %slot14 = arith.addi %base, %c14 : index
    memref.store %one, %items[%slot14] : memref<?xi64>
    func.return
  }

  // --- process identity ----------------------------------------------------
  func.func @LyPosix_GetPid() -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "posix.getpid", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.int", ly.runtime.primitive = "posix_getpid", ly.runtime.result_contract = "builtins.int"} {
    %value = func.call @LyHost_GetPid() : () -> i64
    %h, %m, %d = func.call @LyLong_FromI64(%value) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyPosix_GetPPid() -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "posix.getppid", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.int", ly.runtime.primitive = "posix_getppid", ly.runtime.result_contract = "builtins.int"} {
    %value = func.call @LyHost_GetPPid() : () -> i64
    %h, %m, %d = func.call @LyLong_FromI64(%value) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyPosix_GetUid() -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "posix.getuid", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.int", ly.runtime.primitive = "posix_getuid", ly.runtime.result_contract = "builtins.int"} {
    %value = func.call @LyHost_GetUid() : () -> i64
    %h, %m, %d = func.call @LyLong_FromI64(%value) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyPosix_GetEUid() -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "posix.geteuid", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.int", ly.runtime.primitive = "posix_geteuid", ly.runtime.result_contract = "builtins.int"} {
    %value = func.call @LyHost_GetEUid() : () -> i64
    %h, %m, %d = func.call @LyLong_FromI64(%value) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyPosix_GetGid() -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "posix.getgid", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.int", ly.runtime.primitive = "posix_getgid", ly.runtime.result_contract = "builtins.int"} {
    %value = func.call @LyHost_GetGid() : () -> i64
    %h, %m, %d = func.call @LyLong_FromI64(%value) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyPosix_GetEGid() -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "posix.getegid", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.int", ly.runtime.primitive = "posix_getegid", ly.runtime.result_contract = "builtins.int"} {
    %value = func.call @LyHost_GetEGid() : () -> i64
    %h, %m, %d = func.call @LyLong_FromI64(%value) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  // --- errno surface -------------------------------------------------------

  func.func @LyPosix_Strerror(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %digits: memref<?xi32>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "posix.strerror", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.int", ly.runtime.primitive = "posix_strerror", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %cap_index = arith.constant 256 : index
    %cap = arith.constant 256 : i64
    %code64 = func.call @LyLong_AsI64(%header, %meta, %digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %code = arith.trunci %code64 : i64 to i32
    %buffer = memref.alloc(%cap_index) : memref<?xi8>
    %len = func.call @LyHost_Strerror(%code, %buffer, %cap) : (i32, memref<?xi8>, i64) -> i64
    %clamped = arith.minsi %len, %cap : i64
    %out_header, %out_bytes = func.call @LyUnicode_FromBytes(%buffer, %c0, %clamped) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    memref.dealloc %buffer : memref<?xi8>
    func.return %out_header, %out_bytes : memref<2xi64>, memref<?xi8>
  }
  func.func private @LyLong_AsI64(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %digits: memref<?xi32>) -> i64 attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__int__", ly.runtime.primitive = "unbox.i64"}

  func.func @LyPosix_RaiseErrno(%code_header: memref<2xi64> {ly.ownership.object_header}, %code_meta: memref<2xi64>, %code_digits: memref<?xi32>, %path_header: memref<2xi64> {ly.ownership.object_header}, %path_bytes: memref<?xi8>) attributes {ly.runtime.builtin = "posix._raise_errno", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.int", ly.runtime.primitive = "posix_raise_errno", ly.runtime.result_contract = "types.NoneType"} {
    %c0 = arith.constant 0 : index
    %code64 = func.call @LyLong_AsI64(%code_header, %code_meta, %code_digits) : (memref<2xi64>, memref<2xi64>, memref<?xi32>) -> i64
    %code = arith.trunci %code64 : i64 to i32
    %enc_header = func.call @LyUnicode_Encode(%path_header, %path_bytes) : (memref<2xi64>, memref<?xi8>) -> memref<6xi64>
    %enc_bytes = func.call @__ly_bytes_payload(%enc_header) : (memref<6xi64>) -> memref<?xi8>
    %enc_dim = memref.dim %enc_bytes, %c0 : memref<?xi8>
    %enc_len = arith.index_cast %enc_dim : index to i64
    func.call @__ly_posix_throw(%code, %enc_bytes, %enc_len) : (i32, memref<?xi8>, i64) -> ()
    func.call @LyBytes_DecRef(%enc_header) : (memref<6xi64>) -> ()
    func.return
  }

  // --- working directory ---------------------------------------------------

  func.func @LyPosix_GetCwd() -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "posix.getcwd", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.str", ly.runtime.primitive = "posix_getcwd", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %cap_index = arith.constant 4096 : index
    %cap = arith.constant 4096 : i64
    %zero = arith.constant 0 : i64
    %buffer = memref.alloc(%cap_index) : memref<?xi8>
    %len = func.call @LyHost_GetCwd(%buffer, %cap) : (memref<?xi8>, i64) -> i64
    %failed = arith.cmpi slt, %len, %zero : i64
    scf.if %failed {
      %err = func.call @LyHost_Errno() : () -> i32
      func.call @__ly_posix_throw(%err, %buffer, %zero) : (i32, memref<?xi8>, i64) -> ()
    }
    %out_header, %out_bytes = func.call @LyUnicode_FromBytes(%buffer, %c0, %len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    memref.dealloc %buffer : memref<?xi8>
    func.return %out_header, %out_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyPosix_Chdir(%path_header: memref<2xi64> {ly.ownership.object_header}, %path_bytes: memref<?xi8>) attributes {ly.runtime.builtin = "posix.chdir", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.str", ly.runtime.primitive = "posix_chdir", ly.runtime.result_contract = "types.NoneType"} {
    %c0 = arith.constant 0 : index
    %zero32 = arith.constant 0 : i32
    %enc_header = func.call @LyUnicode_Encode(%path_header, %path_bytes) : (memref<2xi64>, memref<?xi8>) -> memref<6xi64>
    %enc_bytes = func.call @__ly_bytes_payload(%enc_header) : (memref<6xi64>) -> memref<?xi8>
    %enc_dim = memref.dim %enc_bytes, %c0 : memref<?xi8>
    %enc_len = arith.index_cast %enc_dim : index to i64
    %status = func.call @LyHost_Chdir(%enc_bytes, %enc_len) : (memref<?xi8>, i64) -> i32
    %failed = arith.cmpi ne, %status, %zero32 : i32
    scf.if %failed {
      %err = func.call @LyHost_Errno() : () -> i32
      func.call @__ly_posix_throw(%err, %enc_bytes, %enc_len) : (i32, memref<?xi8>, i64) -> ()
    }
    func.call @LyBytes_DecRef(%enc_header) : (memref<6xi64>) -> ()
    func.return
  }

  // --- single-path mutators ------------------------------------------------

  func.func @LyPosix_Rmdir(%path_header: memref<2xi64> {ly.ownership.object_header}, %path_bytes: memref<?xi8>) attributes {ly.runtime.builtin = "posix.rmdir", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.str", ly.runtime.primitive = "posix_rmdir", ly.runtime.result_contract = "types.NoneType"} {
    %c0 = arith.constant 0 : index
    %zero32 = arith.constant 0 : i32
    %enc_header = func.call @LyUnicode_Encode(%path_header, %path_bytes) : (memref<2xi64>, memref<?xi8>) -> memref<6xi64>
    %enc_bytes = func.call @__ly_bytes_payload(%enc_header) : (memref<6xi64>) -> memref<?xi8>
    %enc_dim = memref.dim %enc_bytes, %c0 : memref<?xi8>
    %enc_len = arith.index_cast %enc_dim : index to i64
    %status = func.call @LyHost_Rmdir(%enc_bytes, %enc_len) : (memref<?xi8>, i64) -> i32
    %failed = arith.cmpi ne, %status, %zero32 : i32
    scf.if %failed {
      %err = func.call @LyHost_Errno() : () -> i32
      func.call @__ly_posix_throw(%err, %enc_bytes, %enc_len) : (i32, memref<?xi8>, i64) -> ()
    }
    func.call @LyBytes_DecRef(%enc_header) : (memref<6xi64>) -> ()
    func.return
  }

  func.func @LyPosix_Unlink(%path_header: memref<2xi64> {ly.ownership.object_header}, %path_bytes: memref<?xi8>) attributes {ly.runtime.builtin = "posix.unlink", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.str", ly.runtime.primitive = "posix_unlink", ly.runtime.result_contract = "types.NoneType"} {
    %c0 = arith.constant 0 : index
    %zero32 = arith.constant 0 : i32
    %enc_header = func.call @LyUnicode_Encode(%path_header, %path_bytes) : (memref<2xi64>, memref<?xi8>) -> memref<6xi64>
    %enc_bytes = func.call @__ly_bytes_payload(%enc_header) : (memref<6xi64>) -> memref<?xi8>
    %enc_dim = memref.dim %enc_bytes, %c0 : memref<?xi8>
    %enc_len = arith.index_cast %enc_dim : index to i64
    %status = func.call @LyHost_Unlink(%enc_bytes, %enc_len) : (memref<?xi8>, i64) -> i32
    %failed = arith.cmpi ne, %status, %zero32 : i32
    scf.if %failed {
      %err = func.call @LyHost_Errno() : () -> i32
      func.call @__ly_posix_throw(%err, %enc_bytes, %enc_len) : (i32, memref<?xi8>, i64) -> ()
    }
    func.call @LyBytes_DecRef(%enc_header) : (memref<6xi64>) -> ()
    func.return
  }

  func.func @LyPosix_Mkdir(%path_header: memref<2xi64> {ly.ownership.object_header}, %path_bytes: memref<?xi8>, %mode: i64 {ly.runtime.default_i64 = 511 : i64}) attributes {ly.runtime.builtin = "posix.mkdir", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.str", ly.runtime.primitive = "posix_mkdir", ly.runtime.result_contract = "types.NoneType"} {
    %c0 = arith.constant 0 : index
    %zero32 = arith.constant 0 : i32
    %enc_header = func.call @LyUnicode_Encode(%path_header, %path_bytes) : (memref<2xi64>, memref<?xi8>) -> memref<6xi64>
    %enc_bytes = func.call @__ly_bytes_payload(%enc_header) : (memref<6xi64>) -> memref<?xi8>
    %enc_dim = memref.dim %enc_bytes, %c0 : memref<?xi8>
    %enc_len = arith.index_cast %enc_dim : index to i64
    %status = func.call @LyHost_Mkdir(%enc_bytes, %enc_len, %mode) : (memref<?xi8>, i64, i64) -> i32
    %failed = arith.cmpi ne, %status, %zero32 : i32
    scf.if %failed {
      %err = func.call @LyHost_Errno() : () -> i32
      func.call @__ly_posix_throw(%err, %enc_bytes, %enc_len) : (i32, memref<?xi8>, i64) -> ()
    }
    func.call @LyBytes_DecRef(%enc_header) : (memref<6xi64>) -> ()
    func.return
  }

  func.func @LyPosix_Rename(%src_header: memref<2xi64> {ly.ownership.object_header}, %src_bytes: memref<?xi8>, %dst_header: memref<2xi64> {ly.ownership.object_header}, %dst_bytes: memref<?xi8>) attributes {ly.runtime.builtin = "posix.rename", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.str", ly.runtime.primitive = "posix_rename", ly.runtime.result_contract = "types.NoneType"} {
    %c0 = arith.constant 0 : index
    %zero32 = arith.constant 0 : i32
    %src_enc_header = func.call @LyUnicode_Encode(%src_header, %src_bytes) : (memref<2xi64>, memref<?xi8>) -> memref<6xi64>
    %src_enc = func.call @__ly_bytes_payload(%src_enc_header) : (memref<6xi64>) -> memref<?xi8>
    %src_dim = memref.dim %src_enc, %c0 : memref<?xi8>
    %src_len = arith.index_cast %src_dim : index to i64
    %dst_enc_header = func.call @LyUnicode_Encode(%dst_header, %dst_bytes) : (memref<2xi64>, memref<?xi8>) -> memref<6xi64>
    %dst_enc = func.call @__ly_bytes_payload(%dst_enc_header) : (memref<6xi64>) -> memref<?xi8>
    %dst_dim = memref.dim %dst_enc, %c0 : memref<?xi8>
    %dst_len = arith.index_cast %dst_dim : index to i64
    %status = func.call @LyHost_Rename(%src_enc, %src_len, %dst_enc, %dst_len) : (memref<?xi8>, i64, memref<?xi8>, i64) -> i32
    %failed = arith.cmpi ne, %status, %zero32 : i32
    scf.if %failed {
      %err = func.call @LyHost_Errno() : () -> i32
      func.call @__ly_posix_throw2(%err, %src_enc, %src_len, %dst_enc, %dst_len) : (i32, memref<?xi8>, i64, memref<?xi8>, i64) -> ()
    }
    func.call @LyBytes_DecRef(%src_enc_header) : (memref<6xi64>) -> ()
    func.call @LyBytes_DecRef(%dst_enc_header) : (memref<6xi64>) -> ()
    func.return
  }

  func.func @LyPosix_Access(%path_header: memref<2xi64> {ly.ownership.object_header}, %path_bytes: memref<?xi8>, %mode: i64) -> i1 attributes {ly.runtime.builtin = "posix.access", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.str", ly.runtime.primitive = "posix_access", ly.runtime.result_contract = "builtins.bool"} {
    %c0 = arith.constant 0 : index
    %zero32 = arith.constant 0 : i32
    %enc_header = func.call @LyUnicode_Encode(%path_header, %path_bytes) : (memref<2xi64>, memref<?xi8>) -> memref<6xi64>
    %enc_bytes = func.call @__ly_bytes_payload(%enc_header) : (memref<6xi64>) -> memref<?xi8>
    %enc_dim = memref.dim %enc_bytes, %c0 : memref<?xi8>
    %enc_len = arith.index_cast %enc_dim : index to i64
    %status = func.call @LyHost_Access(%enc_bytes, %enc_len, %mode) : (memref<?xi8>, i64, i64) -> i32
    func.call @LyBytes_DecRef(%enc_header) : (memref<6xi64>) -> ()
    %allowed = arith.cmpi eq, %status, %zero32 : i32
    func.return %allowed : i1
  }

  // --- stat ----------------------------------------------------------------

  // One os.stat_result field, or `-errno`. Never raises: os.path's predicates
  // read st_mode and treat any negative answer as "no such thing", which is
  // CPython's own os.path.exists shape (it swallows OSError).
  func.func @LyPosix_StatField(%path_header: memref<2xi64> {ly.ownership.object_header}, %path_bytes: memref<?xi8>, %follow: i64, %index: i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "posix._stat_field", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.str", ly.runtime.primitive = "posix_stat_field", ly.runtime.result_contract = "builtins.int"} {
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %zero = arith.constant 0 : i64
    %zero32 = arith.constant 0 : i32
    %enc_header = func.call @LyUnicode_Encode(%path_header, %path_bytes) : (memref<2xi64>, memref<?xi8>) -> memref<6xi64>
    %enc_bytes = func.call @__ly_bytes_payload(%enc_header) : (memref<6xi64>) -> memref<?xi8>
    %enc_dim = memref.dim %enc_bytes, %c0 : memref<?xi8>
    %enc_len = arith.index_cast %enc_dim : index to i64
    %fields = memref.alloc(%c10) : memref<?xi64>
    %want_follow = arith.cmpi ne, %follow, %zero : i64
    %status = scf.if %want_follow -> i32 {
      %rc = func.call @LyHost_Stat(%enc_bytes, %enc_len, %fields) : (memref<?xi8>, i64, memref<?xi64>) -> i32
      scf.yield %rc : i32
    } else {
      %rc = func.call @LyHost_LStat(%enc_bytes, %enc_len, %fields) : (memref<?xi8>, i64, memref<?xi64>) -> i32
      scf.yield %rc : i32
    }
    func.call @LyBytes_DecRef(%enc_header) : (memref<6xi64>) -> ()
    %failed = arith.cmpi ne, %status, %zero32 : i32
    %answer = scf.if %failed -> i64 {
      %err = func.call @LyHost_Errno() : () -> i32
      %err64 = arith.extsi %err : i32 to i64
      %negated = arith.subi %zero, %err64 : i64
      scf.yield %negated : i64
    } else {
      %slot = arith.index_cast %index : i64 to index
      %value = memref.load %fields[%slot] : memref<?xi64>
      scf.yield %value : i64
    }
    memref.dealloc %fields : memref<?xi64>
    %h, %m, %d = func.call @LyLong_FromI64(%answer) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %h, %m, %d : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  // --- directory listing ---------------------------------------------------

  // listdir walks the directory twice: once to count the entries the list has
  // to hold (LyList_FromLength takes the final length) and once to fill it.
  // "." and ".." are skipped, as CPython's does.
  func.func @LyPosix_ListDir(%path_header: memref<2xi64> {ly.ownership.object_header}, %path_bytes: memref<?xi8>) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "posix.listdir", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.str", ly.runtime.element_contract = "builtins.str", ly.runtime.primitive = "posix_listdir", ly.runtime.result_contract = "builtins.list"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cap_index = arith.constant 1024 : index
    %cap = arith.constant 1024 : i64
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64

    %enc_header = func.call @LyUnicode_Encode(%path_header, %path_bytes) : (memref<2xi64>, memref<?xi8>) -> memref<6xi64>
    %enc_bytes = func.call @__ly_bytes_payload(%enc_header) : (memref<6xi64>) -> memref<?xi8>
    %enc_dim = memref.dim %enc_bytes, %c0 : memref<?xi8>
    %enc_len = arith.index_cast %enc_dim : index to i64

    %dir = func.call @LyHost_OpenDir(%enc_bytes, %enc_len) : (memref<?xi8>, i64) -> i64
    %opened = arith.cmpi ne, %dir, %zero : i64
    %missing = arith.cmpi eq, %dir, %zero : i64
    scf.if %missing {
      %err = func.call @LyHost_Errno() : () -> i32
      func.call @__ly_posix_throw(%err, %enc_bytes, %enc_len) : (i32, memref<?xi8>, i64) -> ()
    }
    %count = scf.if %opened -> i64 {
      %name = memref.alloc(%cap_index) : memref<?xi8>
      %seen = scf.while (%acc = %zero) : (i64) -> i64 {
        %len = func.call @LyHost_ReadDirName(%dir, %name, %cap) : (i64, memref<?xi8>, i64) -> i64
        %more = arith.cmpi sge, %len, %zero : i64
        %next = scf.if %more -> i64 {
          %skip = func.call @__ly_posix_is_dot_entry(%name, %len) : (memref<?xi8>, i64) -> i1
          %step = arith.select %skip, %zero, %one : i64
          %bumped = arith.addi %acc, %step : i64
          scf.yield %bumped : i64
        } else {
          scf.yield %acc : i64
        }
        scf.condition(%more) %next : i64
      } do {
      ^body(%carried: i64):
        scf.yield %carried : i64
      }
      memref.dealloc %name : memref<?xi8>
      %closed = func.call @LyHost_CloseDir(%dir) : (i64) -> i32
      scf.yield %seen : i64
    } else {
      scf.yield %zero : i64
    }

    %header, %meta, %items = func.call @LyList_FromLength(%count) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>)
    %dir2 = func.call @LyHost_OpenDir(%enc_bytes, %enc_len) : (memref<?xi8>, i64) -> i64
    %opened2 = arith.cmpi ne, %dir2, %zero : i64
    scf.if %opened2 {
      %name = memref.alloc(%cap_index) : memref<?xi8>
      %filled = scf.while (%slot = %zero) : (i64) -> i64 {
        %len = func.call @LyHost_ReadDirName(%dir2, %name, %cap) : (i64, memref<?xi8>, i64) -> i64
        %in_range = arith.cmpi slt, %slot, %count : i64
        %more_raw = arith.cmpi sge, %len, %zero : i64
        %more = arith.andi %more_raw, %in_range : i1
        %next = scf.if %more -> i64 {
          %skip = func.call @__ly_posix_is_dot_entry(%name, %len) : (memref<?xi8>, i64) -> i1
          %advanced = scf.if %skip -> i64 {
            scf.yield %slot : i64
          } else {
            %clamped = arith.minsi %len, %cap : i64
            %str_header, %str_bytes = func.call @LyUnicode_FromBytes(%name, %c0, %clamped) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
            %slot_index = arith.index_cast %slot : i64 to index
            func.call @__ly_posix_store_str(%items, %slot_index, %str_header, %str_bytes) : (memref<?xi64>, index, memref<2xi64>, memref<?xi8>) -> ()
            %bumped = arith.addi %slot, %one : i64
            scf.yield %bumped : i64
          }
          scf.yield %advanced : i64
        } else {
          scf.yield %slot : i64
        }
        scf.condition(%more) %next : i64
      } do {
      ^body(%carried: i64):
        scf.yield %carried : i64
      }
      memref.dealloc %name : memref<?xi8>
      %closed2 = func.call @LyHost_CloseDir(%dir2) : (i64) -> i32
    }
    func.call @LyBytes_DecRef(%enc_header) : (memref<6xi64>) -> ()
    func.return %header, %meta, %items : memref<2xi64>, memref<2xi64>, memref<?xi64>
  }

  // "." / ".." -- the two entries readdir reports that listdir must drop.
  func.func private @__ly_posix_is_dot_entry(%name: memref<?xi8>, %len: i64) -> i1 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %dot = arith.constant 46 : i8
    %is_one = arith.cmpi eq, %len, %one : i64
    %is_two = arith.cmpi eq, %len, %two : i64
    %short = arith.ori %is_one, %is_two : i1
    %result = scf.if %short -> i1 {
      %first = memref.load %name[%c0] : memref<?xi8>
      %first_dot = arith.cmpi eq, %first, %dot : i8
      %rest_dot = scf.if %is_two -> i1 {
        %second = memref.load %name[%c1] : memref<?xi8>
        %second_dot = arith.cmpi eq, %second, %dot : i8
        scf.yield %second_dot : i1
      } else {
        %true = arith.constant true
        scf.yield %true : i1
      }
      %both = arith.andi %first_dot, %rest_dot : i1
      scf.yield %both : i1
    } else {
      %false = arith.constant false
      scf.yield %false : i1
    }
    func.return %result : i1
  }

  // --- environment ---------------------------------------------------------

  func.func @LyPosix_HasEnv(%name_header: memref<2xi64> {ly.ownership.object_header}, %name_bytes: memref<?xi8>) -> i1 attributes {ly.runtime.builtin = "posix._has_env", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.str", ly.runtime.primitive = "posix_has_env", ly.runtime.result_contract = "builtins.bool"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %one = arith.constant 1 : i64
    %zero = arith.constant 0 : i64
    %enc_header = func.call @LyUnicode_Encode(%name_header, %name_bytes) : (memref<2xi64>, memref<?xi8>) -> memref<6xi64>
    %enc_bytes = func.call @__ly_bytes_payload(%enc_header) : (memref<6xi64>) -> memref<?xi8>
    %enc_dim = memref.dim %enc_bytes, %c0 : memref<?xi8>
    %enc_len = arith.index_cast %enc_dim : index to i64
    %probe = memref.alloc(%c1) : memref<?xi8>
    %len = func.call @LyHost_GetEnv(%enc_bytes, %enc_len, %probe, %zero) : (memref<?xi8>, i64, memref<?xi8>, i64) -> i64
    memref.dealloc %probe : memref<?xi8>
    func.call @LyBytes_DecRef(%enc_header) : (memref<6xi64>) -> ()
    %present = arith.cmpi sge, %len, %zero : i64
    func.return %present : i1
  }

  func.func @LyPosix_GetEnv(%name_header: memref<2xi64> {ly.ownership.object_header}, %name_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "posix._getenv", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.str", ly.runtime.primitive = "posix_getenv", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0 : i64
    %cap_index = arith.constant 4096 : index
    %cap = arith.constant 4096 : i64
    %enc_header = func.call @LyUnicode_Encode(%name_header, %name_bytes) : (memref<2xi64>, memref<?xi8>) -> memref<6xi64>
    %enc_bytes = func.call @__ly_bytes_payload(%enc_header) : (memref<6xi64>) -> memref<?xi8>
    %enc_dim = memref.dim %enc_bytes, %c0 : memref<?xi8>
    %enc_len = arith.index_cast %enc_dim : index to i64
    %buffer = memref.alloc(%cap_index) : memref<?xi8>
    %len = func.call @LyHost_GetEnv(%enc_bytes, %enc_len, %buffer, %cap) : (memref<?xi8>, i64, memref<?xi8>, i64) -> i64
    func.call @LyBytes_DecRef(%enc_header) : (memref<6xi64>) -> ()
    %clamped_high = arith.minsi %len, %cap : i64
    %clamped = arith.maxsi %clamped_high, %zero : i64
    %out_header, %out_bytes = func.call @LyUnicode_FromBytes(%buffer, %c0, %clamped) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    memref.dealloc %buffer : memref<?xi8>
    func.return %out_header, %out_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyPosix_PutEnv(%name_header: memref<2xi64> {ly.ownership.object_header}, %name_bytes: memref<?xi8>, %value_header: memref<2xi64> {ly.ownership.object_header}, %value_bytes: memref<?xi8>) attributes {ly.runtime.builtin = "posix.putenv", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.str", ly.runtime.primitive = "posix_putenv", ly.runtime.result_contract = "types.NoneType"} {
    %c0 = arith.constant 0 : index
    %zero32 = arith.constant 0 : i32
    %name_enc_header = func.call @LyUnicode_Encode(%name_header, %name_bytes) : (memref<2xi64>, memref<?xi8>) -> memref<6xi64>
    %name_enc = func.call @__ly_bytes_payload(%name_enc_header) : (memref<6xi64>) -> memref<?xi8>
    %name_dim = memref.dim %name_enc, %c0 : memref<?xi8>
    %name_len = arith.index_cast %name_dim : index to i64
    %value_enc_header = func.call @LyUnicode_Encode(%value_header, %value_bytes) : (memref<2xi64>, memref<?xi8>) -> memref<6xi64>
    %value_enc = func.call @__ly_bytes_payload(%value_enc_header) : (memref<6xi64>) -> memref<?xi8>
    %value_dim = memref.dim %value_enc, %c0 : memref<?xi8>
    %value_len = arith.index_cast %value_dim : index to i64
    %status = func.call @LyHost_SetEnv(%name_enc, %name_len, %value_enc, %value_len) : (memref<?xi8>, i64, memref<?xi8>, i64) -> i32
    %failed = arith.cmpi ne, %status, %zero32 : i32
    scf.if %failed {
      %err = func.call @LyHost_Errno() : () -> i32
      func.call @__ly_posix_throw(%err, %name_enc, %name_len) : (i32, memref<?xi8>, i64) -> ()
    }
    func.call @LyBytes_DecRef(%name_enc_header) : (memref<6xi64>) -> ()
    func.call @LyBytes_DecRef(%value_enc_header) : (memref<6xi64>) -> ()
    func.return
  }

  func.func @LyPosix_UnsetEnv(%name_header: memref<2xi64> {ly.ownership.object_header}, %name_bytes: memref<?xi8>) attributes {ly.runtime.builtin = "posix.unsetenv", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.str", ly.runtime.primitive = "posix_unsetenv", ly.runtime.result_contract = "types.NoneType"} {
    %c0 = arith.constant 0 : index
    %zero32 = arith.constant 0 : i32
    %enc_header = func.call @LyUnicode_Encode(%name_header, %name_bytes) : (memref<2xi64>, memref<?xi8>) -> memref<6xi64>
    %enc_bytes = func.call @__ly_bytes_payload(%enc_header) : (memref<6xi64>) -> memref<?xi8>
    %enc_dim = memref.dim %enc_bytes, %c0 : memref<?xi8>
    %enc_len = arith.index_cast %enc_dim : index to i64
    %status = func.call @LyHost_UnsetEnv(%enc_bytes, %enc_len) : (memref<?xi8>, i64) -> i32
    %failed = arith.cmpi ne, %status, %zero32 : i32
    scf.if %failed {
      %err = func.call @LyHost_Errno() : () -> i32
      func.call @__ly_posix_throw(%err, %enc_bytes, %enc_len) : (i32, memref<?xi8>, i64) -> ()
    }
    func.call @LyBytes_DecRef(%enc_header) : (memref<6xi64>) -> ()
    func.return
  }

  // The raw "KEY=VALUE" vector, same shape as sys.argv's list[str] build.
  func.func @LyPosix_EnvironEntries() -> (memref<2xi64>, memref<2xi64>, memref<?xi64>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "posix._environ_entries", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.list", ly.runtime.element_contract = "builtins.str", ly.runtime.primitive = "posix_environ_entries", ly.runtime.result_contract = "builtins.list"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %count = func.call @LyHost_EnvironCount() : () -> i64
    %header, %meta, %items = func.call @LyList_FromLength(%count) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>)
    %count_index = arith.index_cast %count : i64 to index
    scf.for %i = %c0 to %count_index step %c1 {
      %i_i64 = arith.index_cast %i : index to i64
      %len = func.call @LyHost_EnvironLen(%i_i64) : (i64) -> i64
      %len_index = arith.index_cast %len : i64 to index
      %buffer = memref.alloc(%len_index) : memref<?xi8>
      func.call @LyHost_EnvironCopy(%i_i64, %buffer, %len) : (i64, memref<?xi8>, i64) -> ()
      %str_header, %str_bytes = func.call @LyUnicode_FromBytes(%buffer, %c0, %len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
      memref.dealloc %buffer : memref<?xi8>
      func.call @__ly_posix_store_str(%items, %i, %str_header, %str_bytes) : (memref<?xi64>, index, memref<2xi64>, memref<?xi8>) -> ()
    }
    func.return %header, %meta, %items : memref<2xi64>, memref<2xi64>, memref<?xi64>
  }
}
