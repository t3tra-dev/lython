// Contract manifest AND runtime implementation for the statically supported
// `_io` surface (CPython's Modules/_io/ counterpart).
//
// Signature sources (1:1 correspondence target):
//   https://github.com/python/typeshed/blob/main/stdlib/_io.pyi
//   https://github.com/python/cpython/blob/main/Modules/_io/_iomodule.c
//   https://github.com/python/cpython/blob/main/Modules/_io/textio.c
//   https://github.com/python/cpython/blob/main/Modules/_io/fileio.c
//
// Deviations from CPython:
// - TextIOWrapper wraps either a raw fd (the sys.stdout/stderr singletons)
//   or a libc FILE* (files returned by open()) directly instead of a
//   FileIO + BufferedReader/Writer stack. FILE* carries the buffering, so
//   flush()/close() behave like the buffered wrapper's.
// - open(file, mode='r') only accepts text modes (r/w/a/x with optional
//   '+'; 'b' raises ValueError because binary streams go through FileIO)
//   and always returns TextIOWrapper (encoding is always UTF-8, the str
//   representation). Clinic defaults map to `arg_defaults` in the typing
//   contract plus `ly.runtime.default_*` on the runtime parameter.
// - read() takes no size argument (clinic: read(size=-1)); it always reads
//   to EOF. readline() takes no size argument either.
// - seek()/tell() (cookie-based text seeking), reconfigure(), detach(),
//   buffer/encoding/errors/newlines attributes are not yet supported.
// - FileIO is absent (open()'s text path covers files until the raw
//   fd/bytes stream is ported).
// - StringIO/BytesIO (stringio.c / bytesio.c): the malloc'd payload lives
//   as an i64 pointer INSIDE the object header words (the FILE* handle
//   trick), so buffer growth updates the header in place and the object's
//   physical values never change. write() overwrites at the stream
//   position and extends like CPython's; seek(pos, whence=0)/tell() follow
//   the per-class C semantics; truncate() is not yet supported.
// - open() errors raise FileNotFoundError with a simplified message (no
//   errno prefix); UnsupportedOperation derives from OSError only (CPython
//   also mixes in ValueError; multiple inheritance is outside the static
//   surface).
//
// The stdout/stderr singleton instances live in sys.mlir (CPython
// initializes them in pylifecycle's init_sys_streams and stores them on
// sys); this module owns the class and its methods.

module attributes {
  ly.typing.module = "_io",
  ly.runtime.contracts = ["_io.TextIOWrapper", "_io.UnsupportedOperation", "_io.StringIO", "_io.BytesIO", "_io.FileIO"],
  ly.typing.class_exports = [
    "_io.TextIOWrapper=_io.TextIOWrapper",
    "_io.UnsupportedOperation=_io.UnsupportedOperation",
    "_io.StringIO=_io.StringIO",
    "_io.BytesIO=_io.BytesIO",
    "_io.FileIO=_io.FileIO"
  ],
  ly.typing.callable_exports = ["_io.open", "_io.open_binary"],
  ly.typing.function_names = ["_io.open", "_io.open_binary"],
  ly.typing.function_contracts = [
    !py.callable<[!py.contract<"builtins.str">, !py.contract<"builtins.str">], arg_names = ["file", "mode"], arg_defaults = [false, true], returns = [!py.contract<"_io.TextIOWrapper">]>,
    !py.callable<[!py.contract<"builtins.str">, !py.contract<"builtins.str">], arg_names = ["file", "mode"], arg_defaults = [false, false], returns = [!py.contract<"_io.FileIO">]>
  ],
  ly.typing.int_constant_names = ["_io.DEFAULT_BUFFER_SIZE"],
  ly.typing.int_constant_values = [131072 : i64]
} {
  // TextIOWrapper instance layout (memref<8xi64>):
  //   [0] refcount   [1] class id (65)  [2] handle (fd or FILE*)
  //   [3] kind (0 = raw fd, 1 = FILE*)  [4] readable  [5] writable
  //   [6] closed     [7] reserved
  py.class @TextIOWrapper attributes {
    base_names = ["object"], ly.typing.final,
    method_names = ["write", "read", "readline", "flush", "close", "fileno", "readable", "writable", "seek", "tell", "seekable"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"_io.TextIOWrapper">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.int">]>,
      !py.callable<[!py.contract<"_io.TextIOWrapper">, !py.contract<"builtins.int">], arg_names = ["self", "size"], arg_defaults = [false, true], returns = [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"_io.TextIOWrapper">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"_io.TextIOWrapper">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"_io.TextIOWrapper">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"_io.TextIOWrapper">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"_io.TextIOWrapper">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"_io.TextIOWrapper">] -> [!py.contract<"builtins.bool">]>,
      !py.callable<[!py.contract<"_io.TextIOWrapper">, !py.contract<"builtins.int">, !py.contract<"builtins.int">], arg_names = ["self", "cookie", "whence"], arg_defaults = [false, false, true], returns = [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"_io.TextIOWrapper">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"_io.TextIOWrapper">] -> [!py.contract<"builtins.bool">]>
    ],
    method_kinds = ["instance", "instance", "instance", "instance", "instance", "instance", "instance", "instance", "instance", "instance", "instance"]
  } {}

  py.class @UnsupportedOperation attributes {base_names = ["OSError"]} {}

  // In-memory stream layout (both classes, memref<8xi64>):
  //   [0] refcount  [1] class id  [2] payload pointer (malloc'd, may be 0)
  //   [3] length    [4] capacity  [5] position  [6] closed  [7] reserved
  py.class @StringIO attributes {
    base_names = ["object"], ly.typing.final,
    method_names = ["__init__", "write", "getvalue", "read", "seek", "tell", "truncate", "seekable", "close", "readable", "writable"],
    method_contracts = [
      !py.callable<[!py.contract<"_io.StringIO">, !py.contract<"builtins.str">], arg_names = ["self", "initial_value"], arg_defaults = [false, true], returns = [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"_io.StringIO">, !py.contract<"builtins.str">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"_io.StringIO">] -> [!py.contract<"builtins.str">]>,
      !py.callable<[!py.contract<"_io.StringIO">, !py.contract<"builtins.int">], arg_names = ["self", "size"], arg_defaults = [false, true], returns = [!py.contract<"builtins.str">]>,
      !py.callable<[!py.contract<"_io.StringIO">, !py.contract<"builtins.int">, !py.contract<"builtins.int">], arg_names = ["self", "pos", "whence"], arg_defaults = [false, false, true], returns = [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"_io.StringIO">] -> [!py.contract<"builtins.int">]>,
      !py.callable<[!py.contract<"_io.StringIO">, !py.contract<"builtins.int">], arg_names = ["self", "pos"], arg_defaults = [false, true], returns = [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"_io.StringIO">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"_io.StringIO">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"_io.StringIO">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"_io.StringIO">] -> [!py.contract<"builtins.bool">]>
    ],
    method_kinds = ["instance", "instance", "instance", "instance", "instance", "instance", "instance", "instance", "instance", "instance", "instance"]
  } {}

  py.class @BytesIO attributes {
    base_names = ["object"], ly.typing.final,
    method_names = ["__init__", "write", "getvalue", "read", "seek", "tell", "truncate", "seekable", "close", "readable", "writable"],
    method_contracts = [
      !py.callable<[!py.contract<"_io.BytesIO">, !py.contract<"builtins.bytes">], arg_names = ["self", "initial_bytes"], arg_defaults = [false, true], returns = [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"_io.BytesIO">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"_io.BytesIO">] -> [!py.contract<"builtins.bytes">]>,
      !py.callable<[!py.contract<"_io.BytesIO">, !py.contract<"builtins.int">], arg_names = ["self", "size"], arg_defaults = [false, true], returns = [!py.contract<"builtins.bytes">]>,
      !py.callable<[!py.contract<"_io.BytesIO">, !py.contract<"builtins.int">, !py.contract<"builtins.int">], arg_names = ["self", "pos", "whence"], arg_defaults = [false, false, true], returns = [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"_io.BytesIO">] -> [!py.contract<"builtins.int">]>,
      !py.callable<[!py.contract<"_io.BytesIO">, !py.contract<"builtins.int">], arg_names = ["self", "pos"], arg_defaults = [false, true], returns = [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"_io.BytesIO">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"_io.BytesIO">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"_io.BytesIO">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"_io.BytesIO">] -> [!py.contract<"builtins.bool">]>
    ],
    method_kinds = ["instance", "instance", "instance", "instance", "instance", "instance", "instance", "instance", "instance", "instance", "instance"]
  } {}

  py.class @FileIO attributes {
    base_names = ["object"], ly.typing.final,
    method_names = ["__init__", "read", "write", "seek", "tell", "truncate", "seekable", "flush", "close", "fileno", "readable", "writable"],
    method_contracts = [
      !py.callable<[!py.contract<"_io.FileIO">, !py.contract<"builtins.str">, !py.contract<"builtins.str">], arg_names = ["self", "file", "mode"], arg_defaults = [false, false, true], returns = [!py.literal<None>]>,
      !py.callable<[!py.contract<"_io.FileIO">, !py.contract<"builtins.int">], arg_names = ["self", "size"], arg_defaults = [false, true], returns = [!py.contract<"builtins.bytes">]>,
      !py.protocol<"Callable", [!py.contract<"_io.FileIO">, !py.contract<"builtins.bytes">] -> [!py.contract<"builtins.int">]>,
      !py.callable<[!py.contract<"_io.FileIO">, !py.contract<"builtins.int">, !py.contract<"builtins.int">], arg_names = ["self", "pos", "whence"], arg_defaults = [false, false, true], returns = [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"_io.FileIO">] -> [!py.contract<"builtins.int">]>,
      !py.callable<[!py.contract<"_io.FileIO">, !py.contract<"builtins.int">], arg_names = ["self", "pos"], arg_defaults = [false, true], returns = [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"_io.FileIO">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"_io.FileIO">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"_io.FileIO">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"_io.FileIO">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"_io.FileIO">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"_io.FileIO">] -> [!py.contract<"builtins.bool">]>
    ],
    method_kinds = ["instance", "instance", "instance", "instance", "instance", "instance", "instance", "instance", "instance", "instance", "instance", "instance"]
  } {}

  // ===== shared runtime declarations (defined in builtins.mlir / support) =====
  func.func private @LyHost_WriteBytes(i32, memref<?xi8>, i64)
  func.func private @LyHost_FOpen(memref<?xi8>, i64, memref<?xi8>, i64) -> i64
  func.func private @LyHost_FRead(i64, memref<?xi8>, i64, i64) -> i64
  func.func private @LyHost_FGetc(i64) -> i32
  func.func private @LyHost_FWrite(i64, memref<?xi8>, i64) -> i64
  func.func private @LyHost_FClose(i64) -> i32
  func.func private @LyHost_FFlush(i64) -> i32
  func.func private @LyHost_Fileno(i64) -> i32
  func.func private @LyHost_FSeek(i64, i64, i32) -> i32
  func.func private @LyHost_FTell(i64) -> i64
  func.func private @LyHost_FUngetc(i64, i32) -> i32
  func.func private @LyHost_FTruncate(i64, i64) -> i32
  func.func private @LyLong_FromI64(%value: i64 {ly.runtime.default_i64 = 0 : i64}) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 1 : i64, ly.runtime.contract = "builtins.int", ly.runtime.initializer = "__new__"}
  func.func private @LyUnicode_FromBytes(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 4 : i64, ly.runtime.contract = "builtins.str", ly.runtime.initializer = "__new__"}
  func.func private @LyUnicode_Length(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> i64 attributes {ly.runtime.contract = "builtins.str", ly.runtime.primitive = "byte_length"}
  func.func private @LyUnicode_CodepointLength(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> i64 attributes {ly.runtime.contract = "builtins.str", ly.runtime.method = "__len__"}
  func.func private @LyBaseException_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 5 : i64, ly.runtime.contract = "builtins.BaseException", ly.runtime.initializer = "__new__"}
  func.func private @LyBaseException_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.BaseException", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"}
  func.func private @LyEH_ThrowException(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "raise"}
  func.func private @LyObject_ReleaseStorageToZero(memref<?xi64>) -> i1

  // ===== impls: UnsupportedOperation (OSError subclass, id 69) =====
  func.func private @LyUnsupportedOperation_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "_io.UnsupportedOperation", ly.runtime.shape}

  func.func @LyUnsupportedOperation_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 69 : i64, ly.runtime.contract = "_io.UnsupportedOperation", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnsupportedOperation_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "_io.UnsupportedOperation", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyUnsupportedOperation_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "_io.UnsupportedOperation", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyUnsupportedOperation_Str(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "_io.UnsupportedOperation", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnsupportedOperation_Repr(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "_io.UnsupportedOperation", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %view = memref.cast %message_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %message_header, %message_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func private @Ly_IncRef(%header: memref<2xi64, strided<[1], offset: ?>> {ly.ownership.object_header})

  // ===== raise helpers =====
  memref.global "private" constant @__ly_io_msg_closed : memref<29xi8> = dense<[73, 47, 79, 32, 111, 112, 101, 114, 97, 116, 105, 111, 110, 32, 111, 110, 32, 99, 108, 111, 115, 101, 100, 32, 102, 105, 108, 101, 46]>
  memref.global "private" constant @__ly_io_msg_not_readable : memref<12xi8> = dense<[110, 111, 116, 32, 114, 101, 97, 100, 97, 98, 108, 101]>
  memref.global "private" constant @__ly_io_msg_not_writable : memref<12xi8> = dense<[110, 111, 116, 32, 119, 114, 105, 116, 97, 98, 108, 101]>
  memref.global "private" constant @__ly_io_msg_binary_mode : memref<28xi8> = dense<[98, 105, 110, 97, 114, 121, 32, 109, 111, 100, 101, 32, 105, 115, 32, 110, 111, 116, 32, 115, 117, 112, 112, 111, 114, 116, 101, 100]>
  memref.global "private" constant @__ly_io_msg_invalid_mode : memref<12xi8> = dense<[105, 110, 118, 97, 108, 105, 100, 32, 109, 111, 100, 101]>
  memref.global "private" constant @__ly_io_msg_no_such_file : memref<27xi8> = dense<[78, 111, 32, 115, 117, 99, 104, 32, 102, 105, 108, 101, 32, 111, 114, 32, 100, 105, 114, 101, 99, 116, 111, 114, 121, 58, 32]>
  memref.global "private" constant @__ly_io_msg_neg_seek : memref<22xi8> = dense<[110, 101, 103, 97, 116, 105, 118, 101, 32, 115, 101, 101, 107, 32, 112, 111, 115, 105, 116, 105, 111, 110]>
  memref.global "private" constant @__ly_io_msg_bad_whence : memref<14xi8> = dense<[105, 110, 118, 97, 108, 105, 100, 32, 119, 104, 101, 110, 99, 101]>
  memref.global "private" constant @__ly_io_msg_nonzero_seek : memref<31xi8> = dense<[99, 97, 110, 39, 116, 32, 100, 111, 32, 110, 111, 110, 122, 101, 114, 111, 32, 114, 101, 108, 97, 116, 105, 118, 101, 32, 115, 101, 101, 107, 115]>
  memref.global "private" constant @__ly_io_msg_not_seekable : memref<33xi8> = dense<[117, 110, 100, 101, 114, 108, 121, 105, 110, 103, 32, 115, 116, 114, 101, 97, 109, 32, 105, 115, 32, 110, 111, 116, 32, 115, 101, 101, 107, 97, 98, 108, 101]>
  memref.global "private" constant @__ly_io_msg_neg_size : memref<19xi8> = dense<[110, 101, 103, 97, 116, 105, 118, 101, 32, 115, 105, 122, 101, 32, 118, 97, 108, 117, 101]>

  func.func private @__ly_io_raise(%class_id: i64, %message: memref<?xi8>, %length: i64) {
    %start = arith.constant 0 : index
    %exception:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    %message_header, %message_bytes = func.call @LyUnicode_FromBytes(%message, %start, %length) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %initialized:3 = func.call @LyBaseException_Init(%exception#0, %exception#1, %exception#2, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.call @LyEH_ThrowException(%initialized#0, %initialized#1, %initialized#2) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  // ValueError (53): "I/O operation on closed file."
  func.func private @__ly_io_raise_closed() {
    %class_id = arith.constant 53 : i64
    %length = arith.constant 29 : i64
    %static = memref.get_global @__ly_io_msg_closed : memref<29xi8>
    %message = memref.cast %static : memref<29xi8> to memref<?xi8>
    func.call @__ly_io_raise(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  // UnsupportedOperation (69): "not readable" / "not writable".
  func.func private @__ly_io_raise_not_readable() {
    %class_id = arith.constant 69 : i64
    %length = arith.constant 12 : i64
    %static = memref.get_global @__ly_io_msg_not_readable : memref<12xi8>
    %message = memref.cast %static : memref<12xi8> to memref<?xi8>
    func.call @__ly_io_raise(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @__ly_io_raise_not_writable() {
    %class_id = arith.constant 69 : i64
    %length = arith.constant 12 : i64
    %static = memref.get_global @__ly_io_msg_not_writable : memref<12xi8>
    %message = memref.cast %static : memref<12xi8> to memref<?xi8>
    func.call @__ly_io_raise(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  // ValueError (53) seek diagnostics.
  func.func private @__ly_io_raise_neg_seek() {
    %class_id = arith.constant 53 : i64
    %length = arith.constant 22 : i64
    %static = memref.get_global @__ly_io_msg_neg_seek : memref<22xi8>
    %message = memref.cast %static : memref<22xi8> to memref<?xi8>
    func.call @__ly_io_raise(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @__ly_io_raise_bad_whence() {
    %class_id = arith.constant 53 : i64
    %length = arith.constant 14 : i64
    %static = memref.get_global @__ly_io_msg_bad_whence : memref<14xi8>
    %message = memref.cast %static : memref<14xi8> to memref<?xi8>
    func.call @__ly_io_raise(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  // stringio.c raises OSError (not ValueError) for nonzero relative seeks.
  // UnsupportedOperation (69): seeking on a raw-fd text stream.
  func.func private @__ly_io_raise_not_seekable() {
    %class_id = arith.constant 69 : i64
    %length = arith.constant 33 : i64
    %static = memref.get_global @__ly_io_msg_not_seekable : memref<33xi8>
    %message = memref.cast %static : memref<33xi8> to memref<?xi8>
    func.call @__ly_io_raise(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  // ValueError (53): truncate with an explicit negative size.
  func.func private @__ly_io_raise_neg_size() {
    %class_id = arith.constant 53 : i64
    %length = arith.constant 19 : i64
    %static = memref.get_global @__ly_io_msg_neg_size : memref<19xi8>
    %message = memref.cast %static : memref<19xi8> to memref<?xi8>
    func.call @__ly_io_raise(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @__ly_io_raise_nonzero_seek() {
    %class_id = arith.constant 66 : i64
    %length = arith.constant 31 : i64
    %static = memref.get_global @__ly_io_msg_nonzero_seek : memref<31xi8>
    %message = memref.cast %static : memref<31xi8> to memref<?xi8>
    func.call @__ly_io_raise(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  // ===== impls: TextIOWrapper =====
  func.func private @LyTextIO_Shape() -> memref<8xi64> attributes {ly.runtime.contract = "_io.TextIOWrapper", ly.runtime.shape}

  func.func private @__ly_textio_check_open(%self: memref<8xi64>) {
    %closed_slot = arith.constant 6 : index
    %zero = arith.constant 0 : i64
    %closed = memref.load %self[%closed_slot] : memref<8xi64>
    %is_closed = arith.cmpi ne, %closed, %zero : i64
    scf.if %is_closed {
      func.call @__ly_io_raise_closed() : () -> ()
    }
    func.return
  }

  // write() returns the codepoint count (textio.c write semantics); the
  // byte length drives the underlying raw write.
  func.func @LyTextIO_Write(%self: memref<8xi64> {ly.ownership.object_header}, %str_header: memref<2xi64> {ly.ownership.object_header}, %str_bytes: memref<?xi8>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "_io.TextIOWrapper", ly.runtime.method = "write", ly.runtime.result_contract = "builtins.int"} {
    %handle_slot = arith.constant 2 : index
    %kind_slot = arith.constant 3 : index
    %writable_slot = arith.constant 5 : index
    %zero = arith.constant 0 : i64
    func.call @__ly_textio_check_open(%self) : (memref<8xi64>) -> ()
    %writable = memref.load %self[%writable_slot] : memref<8xi64>
    %not_writable = arith.cmpi eq, %writable, %zero : i64
    scf.if %not_writable {
      func.call @__ly_io_raise_not_writable() : () -> ()
    }
    %handle = memref.load %self[%handle_slot] : memref<8xi64>
    %kind = memref.load %self[%kind_slot] : memref<8xi64>
    %length = func.call @LyUnicode_Length(%str_header, %str_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %is_fd = arith.cmpi eq, %kind, %zero : i64
    scf.if %is_fd {
      %fd = arith.trunci %handle : i64 to i32
      func.call @LyHost_WriteBytes(%fd, %str_bytes, %length) : (i32, memref<?xi8>, i64) -> ()
    } else {
      %written = func.call @LyHost_FWrite(%handle, %str_bytes, %length) : (i64, memref<?xi8>, i64) -> i64
    }
    %codepoints = func.call @LyUnicode_CodepointLength(%str_header, %str_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %result:3 = func.call @LyLong_FromI64(%codepoints) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  // read() decodes the whole rest of the stream (clinic read(size=-1) with
  // the only supported size); the str representation IS UTF-8 bytes, so the
  // read bytes land in the str payload unchanged.
  func.func @LyTextIO_Read(%self: memref<8xi64> {ly.ownership.object_header}, %size: i64 {ly.runtime.default_i64 = -1 : i64}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "_io.TextIOWrapper", ly.runtime.method = "read", ly.runtime.result_contract = "builtins.str"} {
    %handle_slot = arith.constant 2 : index
    %readable_slot = arith.constant 4 : index
    %zero = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %chunk = arith.constant 4096 : i64
    func.call @__ly_textio_check_open(%self) : (memref<8xi64>) -> ()
    %readable = memref.load %self[%readable_slot] : memref<8xi64>
    %not_readable = arith.cmpi eq, %readable, %zero : i64
    scf.if %not_readable {
      func.call @__ly_io_raise_not_readable() : () -> ()
    }
    %handle = memref.load %self[%handle_slot] : memref<8xi64>
    %one = arith.constant 1 : i64
    %unlimited = arith.cmpi slt, %size, %zero : i64

    %result:2 = scf.if %unlimited -> (memref<2xi64>, memref<?xi8>) {
      %initial_cap = arith.constant 4096 : index
      %initial = memref.alloc(%initial_cap) : memref<?xi8>
      %final:3 = scf.while (%buffer = %initial, %len = %zero, %cap = %chunk) : (memref<?xi8>, i64, i64) -> (memref<?xi8>, i64, i64) {
        %room = arith.subi %cap, %len : i64
        %has_room = arith.cmpi sgt, %room, %zero : i64
        %grown:2 = scf.if %has_room -> (memref<?xi8>, i64) {
          scf.yield %buffer, %cap : memref<?xi8>, i64
        } else {
          %two = arith.constant 2 : i64
          %new_cap = arith.muli %cap, %two : i64
          %new_cap_index = arith.index_cast %new_cap : i64 to index
          %bigger = memref.alloc(%new_cap_index) : memref<?xi8>
          %len_index = arith.index_cast %len : i64 to index
          scf.for %i = %c0 to %len_index step %c1 {
            %byte = memref.load %buffer[%i] : memref<?xi8>
            memref.store %byte, %bigger[%i] : memref<?xi8>
          }
          memref.dealloc %buffer : memref<?xi8>
          scf.yield %bigger, %new_cap : memref<?xi8>, i64
        }
        %room_now = arith.subi %grown#1, %len : i64
        %count = func.call @LyHost_FRead(%handle, %grown#0, %len, %room_now) : (i64, memref<?xi8>, i64, i64) -> i64
        %got_data = arith.cmpi sgt, %count, %zero : i64
        %new_len = arith.addi %len, %count : i64
        scf.condition(%got_data) %grown#0, %new_len, %grown#1 : memref<?xi8>, i64, i64
      } do {
      ^bb0(%buffer: memref<?xi8>, %len: i64, %cap: i64):
        scf.yield %buffer, %len, %cap : memref<?xi8>, i64, i64
      }
      %header, %bytes = func.call @LyUnicode_FromBytes(%final#0, %c0, %final#1) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
      memref.dealloc %final#0 : memref<?xi8>
      scf.yield %header, %bytes : memref<2xi64>, memref<?xi8>
    } else {
      // size counts CHARACTERS (textio.c read(size)): consume bytes with
      // fgetc, counting lead bytes; the lead byte that would start
      // character size + 1 is pushed back with ungetc.
      %initial_cap_i64 = arith.constant 128 : i64
      %initial_cap = arith.constant 128 : index
      %initial = memref.alloc(%initial_cap) : memref<?xi8>
      %final:4 = scf.while (%buffer = %initial, %len = %zero, %cap = %initial_cap_i64, %chars = %zero) : (memref<?xi8>, i64, i64, i64) -> (memref<?xi8>, i64, i64, i64) {
        %ch = func.call @LyHost_FGetc(%handle) : (i64) -> i32
        %minus_one = arith.constant -1 : i32
        %at_eof = arith.cmpi eq, %ch, %minus_one : i32
        %mask = arith.constant 192 : i32
        %cont_tag = arith.constant 128 : i32
        %tag = arith.andi %ch, %mask : i32
        %is_cont = arith.cmpi eq, %tag, %cont_tag : i32
        %true_bit = arith.constant true
        %is_lead_raw = arith.xori %is_cont, %true_bit : i1
        %not_eof = arith.xori %at_eof, %true_bit : i1
        %is_lead = arith.andi %is_lead_raw, %not_eof : i1
        %at_limit = arith.cmpi eq, %chars, %size : i64
        %hit = arith.andi %is_lead, %at_limit : i1
        scf.if %hit {
          %pushed = func.call @LyHost_FUngetc(%handle, %ch) : (i64, i32) -> i32
        }
        %stop = arith.ori %at_eof, %hit : i1
        %continue = arith.xori %stop, %true_bit : i1
        %state:4 = scf.if %continue -> (memref<?xi8>, i64, i64, i64) {
          %full = arith.cmpi sge, %len, %cap : i64
          %sized:2 = scf.if %full -> (memref<?xi8>, i64) {
            %two = arith.constant 2 : i64
            %new_cap = arith.muli %cap, %two : i64
            %new_cap_index = arith.index_cast %new_cap : i64 to index
            %bigger = memref.alloc(%new_cap_index) : memref<?xi8>
            %len_index = arith.index_cast %len : i64 to index
            scf.for %i = %c0 to %len_index step %c1 {
              %byte = memref.load %buffer[%i] : memref<?xi8>
              memref.store %byte, %bigger[%i] : memref<?xi8>
            }
            memref.dealloc %buffer : memref<?xi8>
            scf.yield %bigger, %new_cap : memref<?xi8>, i64
          } else {
            scf.yield %buffer, %cap : memref<?xi8>, i64
          }
          %byte = arith.trunci %ch : i32 to i8
          %len_index = arith.index_cast %len : i64 to index
          memref.store %byte, %sized#0[%len_index] : memref<?xi8>
          %next_len = arith.addi %len, %one : i64
          %lead_count = arith.select %is_lead, %one, %zero : i64
          %next_chars = arith.addi %chars, %lead_count : i64
          scf.yield %sized#0, %next_len, %sized#1, %next_chars : memref<?xi8>, i64, i64, i64
        } else {
          scf.yield %buffer, %len, %cap, %chars : memref<?xi8>, i64, i64, i64
        }
        scf.condition(%continue) %state#0, %state#1, %state#2, %state#3 : memref<?xi8>, i64, i64, i64
      } do {
      ^bb0(%buffer: memref<?xi8>, %len: i64, %cap: i64, %chars: i64):
        scf.yield %buffer, %len, %cap, %chars : memref<?xi8>, i64, i64, i64
      }
      %header, %bytes = func.call @LyUnicode_FromBytes(%final#0, %c0, %final#1) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
      memref.dealloc %final#0 : memref<?xi8>
      scf.yield %header, %bytes : memref<2xi64>, memref<?xi8>
    }
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  // readline() keeps the trailing newline and returns "" at EOF (textio.c
  // readline with the FILE* doing the buffering).
  func.func @LyTextIO_ReadLine(%self: memref<8xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "_io.TextIOWrapper", ly.runtime.method = "readline", ly.runtime.result_contract = "builtins.str"} {
    %handle_slot = arith.constant 2 : index
    %readable_slot = arith.constant 4 : index
    %zero = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    func.call @__ly_textio_check_open(%self) : (memref<8xi64>) -> ()
    %readable = memref.load %self[%readable_slot] : memref<8xi64>
    %not_readable = arith.cmpi eq, %readable, %zero : i64
    scf.if %not_readable {
      func.call @__ly_io_raise_not_readable() : () -> ()
    }
    %handle = memref.load %self[%handle_slot] : memref<8xi64>

    %initial_cap_i64 = arith.constant 128 : i64
    %initial_cap = arith.constant 128 : index
    %initial = memref.alloc(%initial_cap) : memref<?xi8>
    %final:3 = scf.while (%buffer = %initial, %len = %zero, %cap = %initial_cap_i64) : (memref<?xi8>, i64, i64) -> (memref<?xi8>, i64, i64) {
      %ch = func.call @LyHost_FGetc(%handle) : (i64) -> i32
      %minus_one = arith.constant -1 : i32
      %at_eof = arith.cmpi eq, %ch, %minus_one : i32
      %grown:3 = scf.if %at_eof -> (memref<?xi8>, i64, i64) {
        scf.yield %buffer, %len, %cap : memref<?xi8>, i64, i64
      } else {
        %need = arith.addi %len, %zero : i64
        %full = arith.cmpi sge, %need, %cap : i64
        %sized:2 = scf.if %full -> (memref<?xi8>, i64) {
          %two = arith.constant 2 : i64
          %new_cap = arith.muli %cap, %two : i64
          %new_cap_index = arith.index_cast %new_cap : i64 to index
          %bigger = memref.alloc(%new_cap_index) : memref<?xi8>
          %len_index = arith.index_cast %len : i64 to index
          scf.for %i = %c0 to %len_index step %c1 {
            %byte = memref.load %buffer[%i] : memref<?xi8>
            memref.store %byte, %bigger[%i] : memref<?xi8>
          }
          memref.dealloc %buffer : memref<?xi8>
          scf.yield %bigger, %new_cap : memref<?xi8>, i64
        } else {
          scf.yield %buffer, %cap : memref<?xi8>, i64
        }
        %byte = arith.trunci %ch : i32 to i8
        %len_index = arith.index_cast %len : i64 to index
        memref.store %byte, %sized#0[%len_index] : memref<?xi8>
        %one = arith.constant 1 : i64
        %next_len = arith.addi %len, %one : i64
        scf.yield %sized#0, %next_len, %sized#1 : memref<?xi8>, i64, i64
      }
      %newline = arith.constant 10 : i32
      %is_newline = arith.cmpi eq, %ch, %newline : i32
      %stop = arith.ori %at_eof, %is_newline : i1
      %true_bit = arith.constant true
      %continue = arith.xori %stop, %true_bit : i1
      scf.condition(%continue) %grown#0, %grown#1, %grown#2 : memref<?xi8>, i64, i64
    } do {
    ^bb0(%buffer: memref<?xi8>, %len: i64, %cap: i64):
      scf.yield %buffer, %len, %cap : memref<?xi8>, i64, i64
    }
    %header, %bytes = func.call @LyUnicode_FromBytes(%final#0, %c0, %final#1) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    memref.dealloc %final#0 : memref<?xi8>
    func.return %header, %bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyTextIO_Flush(%self: memref<8xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "_io.TextIOWrapper", ly.runtime.method = "flush", ly.runtime.result_contract = "types.NoneType"} {
    %handle_slot = arith.constant 2 : index
    %kind_slot = arith.constant 3 : index
    %one = arith.constant 1 : i64
    func.call @__ly_textio_check_open(%self) : (memref<8xi64>) -> ()
    %kind = memref.load %self[%kind_slot] : memref<8xi64>
    %is_file = arith.cmpi eq, %kind, %one : i64
    scf.if %is_file {
      %handle = memref.load %self[%handle_slot] : memref<8xi64>
      %status = func.call @LyHost_FFlush(%handle) : (i64) -> i32
    }
    func.return
  }

  // close() flushes and closes the FILE*; further operations raise
  // ValueError, a second close is a no-op (IOBase semantics).
  func.func @LyTextIO_Close(%self: memref<8xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "_io.TextIOWrapper", ly.runtime.method = "close", ly.runtime.result_contract = "types.NoneType"} {
    %handle_slot = arith.constant 2 : index
    %kind_slot = arith.constant 3 : index
    %closed_slot = arith.constant 6 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %closed = memref.load %self[%closed_slot] : memref<8xi64>
    %already = arith.cmpi ne, %closed, %zero : i64
    scf.if %already {
    } else {
      %kind = memref.load %self[%kind_slot] : memref<8xi64>
      %is_file = arith.cmpi eq, %kind, %one : i64
      scf.if %is_file {
        %handle = memref.load %self[%handle_slot] : memref<8xi64>
        %status = func.call @LyHost_FClose(%handle) : (i64) -> i32
      }
      memref.store %one, %self[%closed_slot] : memref<8xi64>
    }
    func.return
  }

  func.func @LyTextIO_Fileno(%self: memref<8xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "_io.TextIOWrapper", ly.runtime.method = "fileno", ly.runtime.result_contract = "builtins.int"} {
    %handle_slot = arith.constant 2 : index
    %kind_slot = arith.constant 3 : index
    %zero = arith.constant 0 : i64
    func.call @__ly_textio_check_open(%self) : (memref<8xi64>) -> ()
    %handle = memref.load %self[%handle_slot] : memref<8xi64>
    %kind = memref.load %self[%kind_slot] : memref<8xi64>
    %is_fd = arith.cmpi eq, %kind, %zero : i64
    %fd = scf.if %is_fd -> (i64) {
      scf.yield %handle : i64
    } else {
      %fd32 = func.call @LyHost_Fileno(%handle) : (i64) -> i32
      %fd64 = arith.extsi %fd32 : i32 to i64
      scf.yield %fd64 : i64
    }
    %result:3 = func.call @LyLong_FromI64(%fd) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyTextIO_Readable(%self: memref<8xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "_io.TextIOWrapper", ly.runtime.method = "readable", ly.runtime.result_contract = "builtins.bool"} {
    %readable_slot = arith.constant 4 : index
    %zero = arith.constant 0 : i64
    %readable = memref.load %self[%readable_slot] : memref<8xi64>
    %flag = arith.cmpi ne, %readable, %zero : i64
    func.return %flag : i1
  }

  func.func @LyTextIO_Writable(%self: memref<8xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "_io.TextIOWrapper", ly.runtime.method = "writable", ly.runtime.result_contract = "builtins.bool"} {
    %writable_slot = arith.constant 5 : index
    %zero = arith.constant 0 : i64
    %writable = memref.load %self[%writable_slot] : memref<8xi64>
    %flag = arith.cmpi ne, %writable, %zero : i64
    func.return %flag : i1
  }

  // textio.c seek/tell: tell() returns an opaque cookie and seek() only
  // accepts cookies (or offset 0 for cur-/end-relative whence). This
  // wrapper keeps no incremental decoder state, so the cookie degenerates
  // to the byte offset; raw-fd streams (stdout/stderr) are not seekable.
  func.func @LyTextIO_Seek(%self: memref<8xi64> {ly.ownership.object_header}, %cookie: i64, %whence: i64 {ly.runtime.default_i64 = 0 : i64}) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "_io.TextIOWrapper", ly.runtime.method = "seek", ly.runtime.result_contract = "builtins.int"} {
    %handle_slot = arith.constant 2 : index
    %kind_slot = arith.constant 3 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    func.call @__ly_textio_check_open(%self) : (memref<8xi64>) -> ()
    %kind = memref.load %self[%kind_slot] : memref<8xi64>
    %is_fd = arith.cmpi eq, %kind, %zero : i64
    scf.if %is_fd {
      func.call @__ly_io_raise_not_seekable() : () -> ()
    }
    %handle = memref.load %self[%handle_slot] : memref<8xi64>
    %is_set = arith.cmpi eq, %whence, %zero : i64
    scf.if %is_set {
      %negative = arith.cmpi slt, %cookie, %zero : i64
      scf.if %negative {
        func.call @__ly_io_raise_neg_seek() : () -> ()
      }
      %whence32 = arith.constant 0 : i32
      %status = func.call @LyHost_FSeek(%handle, %cookie, %whence32) : (i64, i64, i32) -> i32
    } else {
      %is_cur = arith.cmpi eq, %whence, %one : i64
      %is_end = arith.cmpi eq, %whence, %two : i64
      %known = arith.ori %is_cur, %is_end : i1
      %true_bit = arith.constant true
      %unknown = arith.xori %known, %true_bit : i1
      scf.if %unknown {
        func.call @__ly_io_raise_bad_whence() : () -> ()
      }
      %nonzero = arith.cmpi ne, %cookie, %zero : i64
      scf.if %nonzero {
        func.call @__ly_io_raise_nonzero_seek() : () -> ()
      }
      scf.if %is_end {
        %end32 = arith.constant 2 : i32
        %status = func.call @LyHost_FSeek(%handle, %zero, %end32) : (i64, i64, i32) -> i32
      }
    }
    %where = func.call @LyHost_FTell(%handle) : (i64) -> i64
    %result:3 = func.call @LyLong_FromI64(%where) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyTextIO_Tell(%self: memref<8xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "_io.TextIOWrapper", ly.runtime.method = "tell", ly.runtime.result_contract = "builtins.int"} {
    %handle_slot = arith.constant 2 : index
    %kind_slot = arith.constant 3 : index
    %zero = arith.constant 0 : i64
    func.call @__ly_textio_check_open(%self) : (memref<8xi64>) -> ()
    %kind = memref.load %self[%kind_slot] : memref<8xi64>
    %is_fd = arith.cmpi eq, %kind, %zero : i64
    scf.if %is_fd {
      func.call @__ly_io_raise_not_seekable() : () -> ()
    }
    %handle = memref.load %self[%handle_slot] : memref<8xi64>
    %where = func.call @LyHost_FTell(%handle) : (i64) -> i64
    %result:3 = func.call @LyLong_FromI64(%where) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyTextIO_Seekable(%self: memref<8xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "_io.TextIOWrapper", ly.runtime.method = "seekable", ly.runtime.result_contract = "builtins.bool"} {
    %kind_slot = arith.constant 3 : index
    %zero = arith.constant 0 : i64
    %kind = memref.load %self[%kind_slot] : memref<8xi64>
    %flag = arith.cmpi ne, %kind, %zero : i64
    func.return %flag : i1
  }

  // Heap wrappers close their file when the last reference drops (fileio.c
  // dealloc closes the fd); the immortal singleton refcounts never reach
  // zero, so they never pass the release gate.
  func.func @LyTextIO_DecRef(%self: memref<8xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "_io.TextIOWrapper", ly.runtime.deallocator} {
    %storage = memref.cast %self : memref<8xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    scf.if %became_zero {
      func.call @LyTextIO_Close(%self) : (memref<8xi64>) -> ()
      memref.dealloc %self : memref<8xi64>
    }
    func.return
  }

  // ===== impls: StringIO / BytesIO (shared in-memory buffer core) =====
  func.func private @LyHost_BufferGrow(i64, i64) -> i64
  func.func private @LyHost_BufferCopyIn(i64, i64, memref<?xi8>, i64)
  func.func private @LyHost_BufferCopyOut(i64, i64, memref<?xi8>, i64)
  func.func private @LyHost_BufferFree(i64)
  func.func private @LyBytes_FromBytes(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 70 : i64, ly.runtime.contract = "builtins.bytes", ly.runtime.initializer = "__new__"}

  func.func private @LyStringIO_Shape() -> memref<8xi64> attributes {ly.runtime.contract = "_io.StringIO", ly.runtime.shape}
  func.func private @LyBytesIO_Shape() -> memref<8xi64> attributes {ly.runtime.contract = "_io.BytesIO", ly.runtime.shape}

  func.func private @__ly_membuf_new(%class_id: i64) -> memref<8xi64> attributes {ly.ownership.owned_results = [0]} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %self = memref.alloc() {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<8xi64>
    %refcount_slot = arith.constant 0 : index
    %class_slot = arith.constant 1 : index
    %ptr_slot = arith.constant 2 : index
    %len_slot = arith.constant 3 : index
    %cap_slot = arith.constant 4 : index
    %pos_slot = arith.constant 5 : index
    %closed_slot = arith.constant 6 : index
    %reserved_slot = arith.constant 7 : index
    memref.store %one, %self[%refcount_slot] : memref<8xi64>
    memref.store %class_id, %self[%class_slot] : memref<8xi64>
    memref.store %zero, %self[%ptr_slot] : memref<8xi64>
    memref.store %zero, %self[%len_slot] : memref<8xi64>
    memref.store %zero, %self[%cap_slot] : memref<8xi64>
    memref.store %zero, %self[%pos_slot] : memref<8xi64>
    memref.store %zero, %self[%closed_slot] : memref<8xi64>
    memref.store %zero, %self[%reserved_slot] : memref<8xi64>
    func.return %self : memref<8xi64>
  }

  func.func private @__ly_membuf_check_open(%self: memref<8xi64>) {
    %closed_slot = arith.constant 6 : index
    %zero = arith.constant 0 : i64
    %closed = memref.load %self[%closed_slot] : memref<8xi64>
    %is_closed = arith.cmpi ne, %closed, %zero : i64
    scf.if %is_closed {
      func.call @__ly_io_raise_closed() : () -> ()
    }
    func.return
  }

  // stringio.c/bytesio.c write: store at the stream position (overwriting),
  // extend the length past the end, advance the position.
  func.func private @__ly_membuf_write(%self: memref<8xi64>, %data: memref<?xi8>, %len: i64) {
    %ptr_slot = arith.constant 2 : index
    %len_slot = arith.constant 3 : index
    %cap_slot = arith.constant 4 : index
    %pos_slot = arith.constant 5 : index
    %zero = arith.constant 0 : i64
    func.call @__ly_membuf_check_open(%self) : (memref<8xi64>) -> ()
    %ptr = memref.load %self[%ptr_slot] : memref<8xi64>
    %cur_len = memref.load %self[%len_slot] : memref<8xi64>
    %cap = memref.load %self[%cap_slot] : memref<8xi64>
    %pos = memref.load %self[%pos_slot] : memref<8xi64>
    %need = arith.addi %pos, %len : i64
    %needs_grow = arith.cmpi slt, %cap, %need : i64
    %grown = scf.if %needs_grow -> (i64) {
      %two = arith.constant 2 : i64
      %minimum = arith.constant 64 : i64
      %doubled = arith.muli %cap, %two : i64
      %below_min = arith.cmpi slt, %doubled, %minimum : i64
      %base = arith.select %below_min, %minimum, %doubled : i64
      %below_need = arith.cmpi slt, %base, %need : i64
      %new_cap = arith.select %below_need, %need, %base : i64
      %new_ptr = func.call @LyHost_BufferGrow(%ptr, %new_cap) : (i64, i64) -> i64
      memref.store %new_cap, %self[%cap_slot] : memref<8xi64>
      scf.yield %new_ptr : i64
    } else {
      scf.yield %ptr : i64
    }
    memref.store %grown, %self[%ptr_slot] : memref<8xi64>
    func.call @LyHost_BufferCopyIn(%grown, %pos, %data, %len) : (i64, i64, memref<?xi8>, i64) -> ()
    %new_pos = arith.addi %pos, %len : i64
    %shorter = arith.cmpi slt, %cur_len, %new_pos : i64
    %extended = arith.select %shorter, %new_pos, %cur_len : i64
    memref.store %extended, %self[%len_slot] : memref<8xi64>
    memref.store %new_pos, %self[%pos_slot] : memref<8xi64>
    func.return
  }

  // read() consumes from the position to the end; getvalue() spans the
  // whole buffer without moving the position.
  func.func private @__ly_membuf_read_span(%self: memref<8xi64>) -> (i64, i64) {
    %len_slot = arith.constant 3 : index
    %pos_slot = arith.constant 5 : index
    func.call @__ly_membuf_check_open(%self) : (memref<8xi64>) -> ()
    %len = memref.load %self[%len_slot] : memref<8xi64>
    %pos = memref.load %self[%pos_slot] : memref<8xi64>
    %count = arith.subi %len, %pos : i64
    memref.store %len, %self[%pos_slot] : memref<8xi64>
    func.return %pos, %count : i64, i64
  }

  // stringio.c/bytesio.c truncate: clip the length, keep the position, and
  // return the requested size (which may exceed the buffer; no extension).
  // The INT64_MIN sentinel is the "no size given" clinic default (truncate()
  // truncates at the stream position).
  func.func private @__ly_membuf_truncate(%self: memref<8xi64>, %size: i64) -> i64 {
    %len_slot = arith.constant 3 : index
    %pos_slot = arith.constant 5 : index
    %zero = arith.constant 0 : i64
    %sentinel = arith.constant -9223372036854775808 : i64
    func.call @__ly_membuf_check_open(%self) : (memref<8xi64>) -> ()
    %pos = memref.load %self[%pos_slot] : memref<8xi64>
    %is_default = arith.cmpi eq, %size, %sentinel : i64
    %effective = arith.select %is_default, %pos, %size : i64
    %negative = arith.cmpi slt, %effective, %zero : i64
    scf.if %negative {
      func.call @__ly_io_raise_neg_size() : () -> ()
    }
    %len = memref.load %self[%len_slot] : memref<8xi64>
    %shrink = arith.cmpi slt, %effective, %len : i64
    %new_len = arith.select %shrink, %effective, %len : i64
    memref.store %new_len, %self[%len_slot] : memref<8xi64>
    func.return %effective : i64
  }

  func.func private @__ly_membuf_close(%self: memref<8xi64>) {
    %ptr_slot = arith.constant 2 : index
    %closed_slot = arith.constant 6 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %closed = memref.load %self[%closed_slot] : memref<8xi64>
    %already = arith.cmpi ne, %closed, %zero : i64
    scf.if %already {
    } else {
      %ptr = memref.load %self[%ptr_slot] : memref<8xi64>
      func.call @LyHost_BufferFree(%ptr) : (i64) -> ()
      memref.store %zero, %self[%ptr_slot] : memref<8xi64>
      memref.store %one, %self[%closed_slot] : memref<8xi64>
    }
    func.return
  }

  func.func private @__ly_membuf_decref(%self: memref<8xi64>) {
    %storage = memref.cast %self : memref<8xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    scf.if %became_zero {
      func.call @__ly_membuf_close(%self) : (memref<8xi64>) -> ()
      memref.dealloc %self : memref<8xi64>
    }
    func.return
  }

  // ---- StringIO (class id 68) ----
  func.func @LyStringIO_New(%class_id: i64 {ly.runtime.class_id_argument}) -> memref<8xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 68 : i64, ly.runtime.contract = "_io.StringIO", ly.runtime.initializer = "__new__"} {
    %self = func.call @__ly_membuf_new(%class_id) : (i64) -> memref<8xi64>
    func.return %self : memref<8xi64>
  }

  // stringio.c __init__: seed the buffer with the initial value and rewind
  // to position 0.
  func.func @LyStringIO_Init(%self: memref<8xi64> {ly.ownership.object_header}, %initial_header: memref<2xi64> {ly.ownership.object_header, ly.runtime.default_str = ""}, %initial_bytes: memref<?xi8>) -> memref<8xi64> attributes {ly.ownership.owned_results = [0], ly.ownership.transfer_args = [0], ly.runtime.contract = "_io.StringIO", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %pos_slot = arith.constant 5 : index
    %zero = arith.constant 0 : i64
    %length = func.call @LyUnicode_Length(%initial_header, %initial_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    func.call @__ly_membuf_write(%self, %initial_bytes, %length) : (memref<8xi64>, memref<?xi8>, i64) -> ()
    memref.store %zero, %self[%pos_slot] : memref<8xi64>
    func.return %self : memref<8xi64>
  }

  // write() returns the codepoint count (stringio.c counts characters).
  func.func @LyStringIO_Write(%self: memref<8xi64> {ly.ownership.object_header}, %str_header: memref<2xi64> {ly.ownership.object_header}, %str_bytes: memref<?xi8>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.ownership.owned_result_contracts = ["builtins.int"], ly.runtime.contract = "_io.StringIO", ly.runtime.method = "write", ly.runtime.result_contract = "builtins.int"} {
    %length = func.call @LyUnicode_Length(%str_header, %str_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    func.call @__ly_membuf_write(%self, %str_bytes, %length) : (memref<8xi64>, memref<?xi8>, i64) -> ()
    %codepoints = func.call @LyUnicode_CodepointLength(%str_header, %str_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %result:3 = func.call @LyLong_FromI64(%codepoints) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyStringIO_GetValue(%self: memref<8xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.owned_result_contracts = ["builtins.str"], ly.runtime.contract = "_io.StringIO", ly.runtime.method = "getvalue", ly.runtime.result_contract = "builtins.str"} {
    %ptr_slot = arith.constant 2 : index
    %len_slot = arith.constant 3 : index
    %zero = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    func.call @__ly_membuf_check_open(%self) : (memref<8xi64>) -> ()
    %ptr = memref.load %self[%ptr_slot] : memref<8xi64>
    %len = memref.load %self[%len_slot] : memref<8xi64>
    %len_index = arith.index_cast %len : i64 to index
    %temp = memref.alloc(%len_index) : memref<?xi8>
    func.call @LyHost_BufferCopyOut(%ptr, %zero, %temp, %len) : (i64, i64, memref<?xi8>, i64) -> ()
    %header, %bytes = func.call @LyUnicode_FromBytes(%temp, %c0, %len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    memref.dealloc %temp : memref<?xi8>
    func.return %header, %bytes : memref<2xi64>, memref<?xi8>
  }

  // read(size=-1): size counts CHARACTERS (stringio.c); the scan walks the
  // copied tail and stops on the lead byte that would start character
  // size + 1.
  func.func @LyStringIO_Read(%self: memref<8xi64> {ly.ownership.object_header}, %size: i64 {ly.runtime.default_i64 = -1 : i64}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.owned_result_contracts = ["builtins.str"], ly.runtime.contract = "_io.StringIO", ly.runtime.method = "read", ly.runtime.result_contract = "builtins.str"} {
    %ptr_slot = arith.constant 2 : index
    %len_slot = arith.constant 3 : index
    %pos_slot = arith.constant 5 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    func.call @__ly_membuf_check_open(%self) : (memref<8xi64>) -> ()
    %ptr = memref.load %self[%ptr_slot] : memref<8xi64>
    %len = memref.load %self[%len_slot] : memref<8xi64>
    %pos = memref.load %self[%pos_slot] : memref<8xi64>
    %avail = arith.subi %len, %pos : i64
    %avail_index = arith.index_cast %avail : i64 to index
    %temp = memref.alloc(%avail_index) : memref<?xi8>
    func.call @LyHost_BufferCopyOut(%ptr, %pos, %temp, %avail) : (i64, i64, memref<?xi8>, i64) -> ()
    %unlimited = arith.cmpi slt, %size, %zero : i64
    %take = scf.if %unlimited -> (i64) {
      scf.yield %avail : i64
    } else {
      %scan:2 = scf.while (%i = %zero, %chars = %zero) : (i64, i64) -> (i64, i64) {
        %in_range = arith.cmpi slt, %i, %avail : i64
        %byte = scf.if %in_range -> (i8) {
          %i_index = arith.index_cast %i : i64 to index
          %loaded = memref.load %temp[%i_index] : memref<?xi8>
          scf.yield %loaded : i8
        } else {
          %pad = arith.constant 0 : i8
          scf.yield %pad : i8
        }
        %mask = arith.constant 192 : i8
        %cont_tag = arith.constant 128 : i8
        %tag = arith.andi %byte, %mask : i8
        %is_cont = arith.cmpi eq, %tag, %cont_tag : i8
        %true_bit = arith.constant true
        %is_lead = arith.xori %is_cont, %true_bit : i1
        %at_limit = arith.cmpi eq, %chars, %size : i64
        %hit = arith.andi %is_lead, %at_limit : i1
        %not_hit = arith.xori %hit, %true_bit : i1
        %continue = arith.andi %in_range, %not_hit : i1
        %lead_count = arith.select %is_lead, %one, %zero : i64
        %next_chars_raw = arith.addi %chars, %lead_count : i64
        %next_chars = arith.select %continue, %next_chars_raw, %chars : i64
        %next_i_raw = arith.addi %i, %one : i64
        %next_i = arith.select %continue, %next_i_raw, %i : i64
        scf.condition(%continue) %next_i, %next_chars : i64, i64
      } do {
      ^bb0(%i: i64, %chars: i64):
        scf.yield %i, %chars : i64, i64
      }
      scf.yield %scan#0 : i64
    }
    %header, %bytes = func.call @LyUnicode_FromBytes(%temp, %c0, %take) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    memref.dealloc %temp : memref<?xi8>
    %new_pos = arith.addi %pos, %take : i64
    memref.store %new_pos, %self[%pos_slot] : memref<8xi64>
    func.return %header, %bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyStringIO_Truncate(%self: memref<8xi64> {ly.ownership.object_header}, %size: i64 {ly.runtime.default_i64 = -9223372036854775808 : i64}) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "_io.StringIO", ly.runtime.method = "truncate", ly.runtime.result_contract = "builtins.int"} {
    %effective = func.call @__ly_membuf_truncate(%self, %size) : (memref<8xi64>, i64) -> i64
    %result:3 = func.call @LyLong_FromI64(%effective) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyStringIO_Seekable(%self: memref<8xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "_io.StringIO", ly.runtime.method = "seekable", ly.runtime.result_contract = "builtins.bool"} {
    %true_bit = arith.constant true
    func.return %true_bit : i1
  }

  func.func @LyStringIO_Close(%self: memref<8xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "_io.StringIO", ly.runtime.method = "close", ly.runtime.result_contract = "types.NoneType"} {
    func.call @__ly_membuf_close(%self) : (memref<8xi64>) -> ()
    func.return
  }

  func.func @LyStringIO_Readable(%self: memref<8xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "_io.StringIO", ly.runtime.method = "readable", ly.runtime.result_contract = "builtins.bool"} {
    %true_bit = arith.constant true
    func.return %true_bit : i1
  }

  func.func @LyStringIO_Writable(%self: memref<8xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "_io.StringIO", ly.runtime.method = "writable", ly.runtime.result_contract = "builtins.bool"} {
    %true_bit = arith.constant true
    func.return %true_bit : i1
  }

  // stringio.c seek: absolute seeks take any non-negative position, but
  // cur-/end-relative seeks only accept offset 0 (the text stream has no
  // byte arithmetic on positions).
  func.func @LyStringIO_Seek(%self: memref<8xi64> {ly.ownership.object_header}, %pos: i64, %whence: i64 {ly.runtime.default_i64 = 0 : i64}) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "_io.StringIO", ly.runtime.method = "seek", ly.runtime.result_contract = "builtins.int"} {
    %len_slot = arith.constant 3 : index
    %pos_slot = arith.constant 5 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    func.call @__ly_membuf_check_open(%self) : (memref<8xi64>) -> ()
    %len = memref.load %self[%len_slot] : memref<8xi64>
    %cur = memref.load %self[%pos_slot] : memref<8xi64>
    %is_set = arith.cmpi eq, %whence, %zero : i64
    %new_pos = scf.if %is_set -> (i64) {
      %negative = arith.cmpi slt, %pos, %zero : i64
      scf.if %negative {
        func.call @__ly_io_raise_neg_seek() : () -> ()
      }
      scf.yield %pos : i64
    } else {
      %is_cur = arith.cmpi eq, %whence, %one : i64
      %inner = scf.if %is_cur -> (i64) {
        %nonzero = arith.cmpi ne, %pos, %zero : i64
        scf.if %nonzero {
          func.call @__ly_io_raise_nonzero_seek() : () -> ()
        }
        scf.yield %cur : i64
      } else {
        %is_end = arith.cmpi eq, %whence, %two : i64
        %innermost = scf.if %is_end -> (i64) {
          %nonzero = arith.cmpi ne, %pos, %zero : i64
          scf.if %nonzero {
            func.call @__ly_io_raise_nonzero_seek() : () -> ()
          }
          scf.yield %len : i64
        } else {
          func.call @__ly_io_raise_bad_whence() : () -> ()
          scf.yield %cur : i64
        }
        scf.yield %innermost : i64
      }
      scf.yield %inner : i64
    }
    memref.store %new_pos, %self[%pos_slot] : memref<8xi64>
    %result:3 = func.call @LyLong_FromI64(%new_pos) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyStringIO_Tell(%self: memref<8xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "_io.StringIO", ly.runtime.method = "tell", ly.runtime.result_contract = "builtins.int"} {
    %pos_slot = arith.constant 5 : index
    func.call @__ly_membuf_check_open(%self) : (memref<8xi64>) -> ()
    %pos = memref.load %self[%pos_slot] : memref<8xi64>
    %result:3 = func.call @LyLong_FromI64(%pos) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyStringIO_DecRef(%self: memref<8xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "_io.StringIO", ly.runtime.deallocator} {
    func.call @__ly_membuf_decref(%self) : (memref<8xi64>) -> ()
    func.return
  }

  // ---- BytesIO (class id 71) ----
  func.func @LyBytesIO_New(%class_id: i64 {ly.runtime.class_id_argument}) -> memref<8xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 71 : i64, ly.runtime.contract = "_io.BytesIO", ly.runtime.initializer = "__new__"} {
    %self = func.call @__ly_membuf_new(%class_id) : (i64) -> memref<8xi64>
    func.return %self : memref<8xi64>
  }

  // bytesio.c __init__: seed the buffer with the initial bytes and rewind
  // to position 0.
  func.func @LyBytesIO_Init(%self: memref<8xi64> {ly.ownership.object_header}, %initial_header: memref<2xi64> {ly.ownership.object_header, ly.runtime.default_bytes = ""}, %initial_bytes: memref<?xi8>) -> memref<8xi64> attributes {ly.ownership.owned_results = [0], ly.ownership.transfer_args = [0], ly.runtime.contract = "_io.BytesIO", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %pos_slot = arith.constant 5 : index
    %zero = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %dim = memref.dim %initial_bytes, %c0 : memref<?xi8>
    %length = arith.index_cast %dim : index to i64
    func.call @__ly_membuf_write(%self, %initial_bytes, %length) : (memref<8xi64>, memref<?xi8>, i64) -> ()
    memref.store %zero, %self[%pos_slot] : memref<8xi64>
    func.return %self : memref<8xi64>
  }

  // write() returns the byte count (bytesio.c).
  func.func @LyBytesIO_Write(%self: memref<8xi64> {ly.ownership.object_header}, %bytes_header: memref<2xi64> {ly.ownership.object_header}, %bytes_payload: memref<?xi8>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.ownership.owned_result_contracts = ["builtins.int"], ly.runtime.contract = "_io.BytesIO", ly.runtime.method = "write", ly.runtime.result_contract = "builtins.int"} {
    %c0 = arith.constant 0 : index
    %dim = memref.dim %bytes_payload, %c0 : memref<?xi8>
    %length = arith.index_cast %dim : index to i64
    func.call @__ly_membuf_write(%self, %bytes_payload, %length) : (memref<8xi64>, memref<?xi8>, i64) -> ()
    %result:3 = func.call @LyLong_FromI64(%length) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyBytesIO_GetValue(%self: memref<8xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.owned_result_contracts = ["builtins.bytes"], ly.runtime.contract = "_io.BytesIO", ly.runtime.method = "getvalue", ly.runtime.result_contract = "builtins.bytes"} {
    %ptr_slot = arith.constant 2 : index
    %len_slot = arith.constant 3 : index
    %zero = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    func.call @__ly_membuf_check_open(%self) : (memref<8xi64>) -> ()
    %ptr = memref.load %self[%ptr_slot] : memref<8xi64>
    %len = memref.load %self[%len_slot] : memref<8xi64>
    %len_index = arith.index_cast %len : i64 to index
    %temp = memref.alloc(%len_index) : memref<?xi8>
    func.call @LyHost_BufferCopyOut(%ptr, %zero, %temp, %len) : (i64, i64, memref<?xi8>, i64) -> ()
    %header, %bytes = func.call @LyBytes_FromBytes(%temp, %c0, %len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    memref.dealloc %temp : memref<?xi8>
    func.return %header, %bytes : memref<2xi64>, memref<?xi8>
  }

  // read(size=-1): size counts BYTES (bytesio.c).
  func.func @LyBytesIO_Read(%self: memref<8xi64> {ly.ownership.object_header}, %size: i64 {ly.runtime.default_i64 = -1 : i64}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.owned_result_contracts = ["builtins.bytes"], ly.runtime.contract = "_io.BytesIO", ly.runtime.method = "read", ly.runtime.result_contract = "builtins.bytes"} {
    %ptr_slot = arith.constant 2 : index
    %len_slot = arith.constant 3 : index
    %pos_slot = arith.constant 5 : index
    %zero = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    func.call @__ly_membuf_check_open(%self) : (memref<8xi64>) -> ()
    %ptr = memref.load %self[%ptr_slot] : memref<8xi64>
    %len = memref.load %self[%len_slot] : memref<8xi64>
    %pos = memref.load %self[%pos_slot] : memref<8xi64>
    %avail = arith.subi %len, %pos : i64
    %unlimited = arith.cmpi slt, %size, %zero : i64
    %clip = arith.cmpi slt, %size, %avail : i64
    %limited_take = arith.select %clip, %size, %avail : i64
    %take = arith.select %unlimited, %avail, %limited_take : i64
    %take_index = arith.index_cast %take : i64 to index
    %temp = memref.alloc(%take_index) : memref<?xi8>
    func.call @LyHost_BufferCopyOut(%ptr, %pos, %temp, %take) : (i64, i64, memref<?xi8>, i64) -> ()
    %header, %bytes = func.call @LyBytes_FromBytes(%temp, %c0, %take) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    memref.dealloc %temp : memref<?xi8>
    %new_pos = arith.addi %pos, %take : i64
    memref.store %new_pos, %self[%pos_slot] : memref<8xi64>
    func.return %header, %bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyBytesIO_Truncate(%self: memref<8xi64> {ly.ownership.object_header}, %size: i64 {ly.runtime.default_i64 = -9223372036854775808 : i64}) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "_io.BytesIO", ly.runtime.method = "truncate", ly.runtime.result_contract = "builtins.int"} {
    %effective = func.call @__ly_membuf_truncate(%self, %size) : (memref<8xi64>, i64) -> i64
    %result:3 = func.call @LyLong_FromI64(%effective) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyBytesIO_Seekable(%self: memref<8xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "_io.BytesIO", ly.runtime.method = "seekable", ly.runtime.result_contract = "builtins.bool"} {
    %true_bit = arith.constant true
    func.return %true_bit : i1
  }

  func.func @LyBytesIO_Close(%self: memref<8xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "_io.BytesIO", ly.runtime.method = "close", ly.runtime.result_contract = "types.NoneType"} {
    func.call @__ly_membuf_close(%self) : (memref<8xi64>) -> ()
    func.return
  }

  func.func @LyBytesIO_Readable(%self: memref<8xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "_io.BytesIO", ly.runtime.method = "readable", ly.runtime.result_contract = "builtins.bool"} {
    %true_bit = arith.constant true
    func.return %true_bit : i1
  }

  func.func @LyBytesIO_Writable(%self: memref<8xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "_io.BytesIO", ly.runtime.method = "writable", ly.runtime.result_contract = "builtins.bool"} {
    %true_bit = arith.constant true
    func.return %true_bit : i1
  }

  // bytesio.c seek: cur-/end-relative seeks take arbitrary offsets; the
  // resulting position just must not be negative.
  func.func @LyBytesIO_Seek(%self: memref<8xi64> {ly.ownership.object_header}, %pos: i64, %whence: i64 {ly.runtime.default_i64 = 0 : i64}) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "_io.BytesIO", ly.runtime.method = "seek", ly.runtime.result_contract = "builtins.int"} {
    %len_slot = arith.constant 3 : index
    %pos_slot = arith.constant 5 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    func.call @__ly_membuf_check_open(%self) : (memref<8xi64>) -> ()
    %len = memref.load %self[%len_slot] : memref<8xi64>
    %cur = memref.load %self[%pos_slot] : memref<8xi64>
    %is_set = arith.cmpi eq, %whence, %zero : i64
    %is_cur = arith.cmpi eq, %whence, %one : i64
    %is_end = arith.cmpi eq, %whence, %two : i64
    %known_a = arith.ori %is_set, %is_cur : i1
    %known = arith.ori %known_a, %is_end : i1
    %true_bit = arith.constant true
    %unknown = arith.xori %known, %true_bit : i1
    scf.if %unknown {
      func.call @__ly_io_raise_bad_whence() : () -> ()
    }
    %from_cur = arith.addi %cur, %pos : i64
    %from_end = arith.addi %len, %pos : i64
    %candidate_a = arith.select %is_cur, %from_cur, %pos : i64
    %candidate = arith.select %is_end, %from_end, %candidate_a : i64
    %negative = arith.cmpi slt, %candidate, %zero : i64
    scf.if %negative {
      func.call @__ly_io_raise_neg_seek() : () -> ()
    }
    memref.store %candidate, %self[%pos_slot] : memref<8xi64>
    %result:3 = func.call @LyLong_FromI64(%candidate) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyBytesIO_Tell(%self: memref<8xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "_io.BytesIO", ly.runtime.method = "tell", ly.runtime.result_contract = "builtins.int"} {
    %pos_slot = arith.constant 5 : index
    func.call @__ly_membuf_check_open(%self) : (memref<8xi64>) -> ()
    %pos = memref.load %self[%pos_slot] : memref<8xi64>
    %result:3 = func.call @LyLong_FromI64(%pos) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyBytesIO_DecRef(%self: memref<8xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "_io.BytesIO", ly.runtime.deallocator} {
    func.call @__ly_membuf_decref(%self) : (memref<8xi64>) -> ()
    func.return
  }

  // ---- FileIO (class id 72; fileio.c restricted to the FILE*-backed
  // surface: reads and writes are bytes, unlike TextIOWrapper's str) ----
  func.func private @LyFileIO_Shape() -> memref<8xi64> attributes {ly.runtime.contract = "_io.FileIO", ly.runtime.shape}

  func.func @LyFileIO_New(%class_id: i64 {ly.runtime.class_id_argument}) -> memref<8xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 72 : i64, ly.runtime.contract = "_io.FileIO", ly.runtime.initializer = "__new__"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %self = memref.alloc() {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<8xi64>
    %refcount_slot = arith.constant 0 : index
    %class_slot = arith.constant 1 : index
    %handle_slot = arith.constant 2 : index
    %kind_slot = arith.constant 3 : index
    %readable_slot = arith.constant 4 : index
    %writable_slot = arith.constant 5 : index
    %closed_slot = arith.constant 6 : index
    %reserved_slot = arith.constant 7 : index
    memref.store %one, %self[%refcount_slot] : memref<8xi64>
    memref.store %class_id, %self[%class_slot] : memref<8xi64>
    memref.store %zero, %self[%handle_slot] : memref<8xi64>
    memref.store %one, %self[%kind_slot] : memref<8xi64>
    memref.store %zero, %self[%readable_slot] : memref<8xi64>
    memref.store %zero, %self[%writable_slot] : memref<8xi64>
    memref.store %zero, %self[%closed_slot] : memref<8xi64>
    memref.store %zero, %self[%reserved_slot] : memref<8xi64>
    func.return %self : memref<8xi64>
  }

  // FileIO(file, mode): the binary twin of open()'s mode handling ('b' is
  // accepted and implied).
  func.func @LyFileIO_Init(%self: memref<8xi64> {ly.ownership.object_header}, %path_header: memref<2xi64> {ly.ownership.object_header}, %path_bytes: memref<?xi8>, %mode_header: memref<2xi64> {ly.ownership.object_header, ly.runtime.default_str = "r"}, %mode_bytes: memref<?xi8>) -> memref<8xi64> attributes {ly.ownership.owned_results = [0], ly.ownership.transfer_args = [0], ly.runtime.contract = "_io.FileIO", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %handle_slot = arith.constant 2 : index
    %readable_slot = arith.constant 4 : index
    %writable_slot = arith.constant 5 : index
    %true_bit = arith.constant true
    %base, %plus = func.call @__ly_io_parse_mode(%mode_header, %mode_bytes, %true_bit) : (memref<2xi64>, memref<?xi8>, i1) -> (i64, i64)
    %handle = func.call @__ly_io_fopen(%path_header, %path_bytes, %base, %plus, %true_bit) : (memref<2xi64>, memref<?xi8>, i64, i64, i1) -> i64
    %readable, %writable = func.call @__ly_io_mode_readable(%base, %plus) : (i64, i64) -> (i64, i64)
    memref.store %handle, %self[%handle_slot] : memref<8xi64>
    memref.store %readable, %self[%readable_slot] : memref<8xi64>
    memref.store %writable, %self[%writable_slot] : memref<8xi64>
    func.return %self : memref<8xi64>
  }

  // read() reads to EOF and returns bytes (clinic fileio.c read(size=-1)
  // with the only supported size).
  func.func @LyFileIO_Read(%self: memref<8xi64> {ly.ownership.object_header}, %size: i64 {ly.runtime.default_i64 = -1 : i64}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.bytes"], ly.ownership.owned_results = [0], ly.runtime.contract = "_io.FileIO", ly.runtime.method = "read", ly.runtime.result_contract = "builtins.bytes"} {
    %handle_slot = arith.constant 2 : index
    %readable_slot = arith.constant 4 : index
    %zero = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %chunk = arith.constant 4096 : i64
    func.call @__ly_textio_check_open(%self) : (memref<8xi64>) -> ()
    %readable = memref.load %self[%readable_slot] : memref<8xi64>
    %not_readable = arith.cmpi eq, %readable, %zero : i64
    scf.if %not_readable {
      func.call @__ly_io_raise_not_readable() : () -> ()
    }
    %handle = memref.load %self[%handle_slot] : memref<8xi64>
    %unlimited = arith.cmpi slt, %size, %zero : i64

    %result:2 = scf.if %unlimited -> (memref<2xi64>, memref<?xi8>) {
      %initial_cap = arith.constant 4096 : index
      %initial = memref.alloc(%initial_cap) : memref<?xi8>
      %final:3 = scf.while (%buffer = %initial, %len = %zero, %cap = %chunk) : (memref<?xi8>, i64, i64) -> (memref<?xi8>, i64, i64) {
        %room = arith.subi %cap, %len : i64
        %has_room = arith.cmpi sgt, %room, %zero : i64
        %grown:2 = scf.if %has_room -> (memref<?xi8>, i64) {
          scf.yield %buffer, %cap : memref<?xi8>, i64
        } else {
          %two = arith.constant 2 : i64
          %new_cap = arith.muli %cap, %two : i64
          %new_cap_index = arith.index_cast %new_cap : i64 to index
          %bigger = memref.alloc(%new_cap_index) : memref<?xi8>
          %len_index = arith.index_cast %len : i64 to index
          scf.for %i = %c0 to %len_index step %c1 {
            %byte = memref.load %buffer[%i] : memref<?xi8>
            memref.store %byte, %bigger[%i] : memref<?xi8>
          }
          memref.dealloc %buffer : memref<?xi8>
          scf.yield %bigger, %new_cap : memref<?xi8>, i64
        }
        %room_now = arith.subi %grown#1, %len : i64
        %count = func.call @LyHost_FRead(%handle, %grown#0, %len, %room_now) : (i64, memref<?xi8>, i64, i64) -> i64
        %got_data = arith.cmpi sgt, %count, %zero : i64
        %new_len = arith.addi %len, %count : i64
        scf.condition(%got_data) %grown#0, %new_len, %grown#1 : memref<?xi8>, i64, i64
      } do {
      ^bb0(%buffer: memref<?xi8>, %len: i64, %cap: i64):
        scf.yield %buffer, %len, %cap : memref<?xi8>, i64, i64
      }
      %header, %bytes = func.call @LyBytes_FromBytes(%final#0, %c0, %final#1) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
      memref.dealloc %final#0 : memref<?xi8>
      scf.yield %header, %bytes : memref<2xi64>, memref<?xi8>
    } else {
      // Bounded read: loop until size bytes or EOF (regular files only
      // short-read at EOF, but looping keeps pipes correct too).
      %size_index = arith.index_cast %size : i64 to index
      %temp = memref.alloc(%size_index) : memref<?xi8>
      %final = scf.while (%len = %zero) : (i64) -> (i64) {
        %room = arith.subi %size, %len : i64
        %want = arith.cmpi sgt, %room, %zero : i64
        %count = scf.if %want -> (i64) {
          %read = func.call @LyHost_FRead(%handle, %temp, %len, %room) : (i64, memref<?xi8>, i64, i64) -> i64
          scf.yield %read : i64
        } else {
          scf.yield %zero : i64
        }
        %got_data = arith.cmpi sgt, %count, %zero : i64
        %continue = arith.andi %want, %got_data : i1
        %new_len = arith.addi %len, %count : i64
        scf.condition(%continue) %new_len : i64
      } do {
      ^bb0(%len: i64):
        scf.yield %len : i64
      }
      %header, %bytes = func.call @LyBytes_FromBytes(%temp, %c0, %final) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
      memref.dealloc %temp : memref<?xi8>
      scf.yield %header, %bytes : memref<2xi64>, memref<?xi8>
    }
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  // write() returns the byte count reported by the raw write.
  func.func @LyFileIO_Write(%self: memref<8xi64> {ly.ownership.object_header}, %bytes_header: memref<2xi64> {ly.ownership.object_header}, %bytes_payload: memref<?xi8>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "_io.FileIO", ly.runtime.method = "write", ly.runtime.result_contract = "builtins.int"} {
    %handle_slot = arith.constant 2 : index
    %writable_slot = arith.constant 5 : index
    %zero = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    func.call @__ly_textio_check_open(%self) : (memref<8xi64>) -> ()
    %writable = memref.load %self[%writable_slot] : memref<8xi64>
    %not_writable = arith.cmpi eq, %writable, %zero : i64
    scf.if %not_writable {
      func.call @__ly_io_raise_not_writable() : () -> ()
    }
    %handle = memref.load %self[%handle_slot] : memref<8xi64>
    %dim = memref.dim %bytes_payload, %c0 : memref<?xi8>
    %length = arith.index_cast %dim : index to i64
    %written = func.call @LyHost_FWrite(%handle, %bytes_payload, %length) : (i64, memref<?xi8>, i64) -> i64
    %result:3 = func.call @LyLong_FromI64(%written) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyFileIO_Seek(%self: memref<8xi64> {ly.ownership.object_header}, %pos: i64, %whence: i64 {ly.runtime.default_i64 = 0 : i64}) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "_io.FileIO", ly.runtime.method = "seek", ly.runtime.result_contract = "builtins.int"} {
    %handle_slot = arith.constant 2 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    func.call @__ly_textio_check_open(%self) : (memref<8xi64>) -> ()
    %is_set = arith.cmpi eq, %whence, %zero : i64
    %is_cur = arith.cmpi eq, %whence, %one : i64
    %is_end = arith.cmpi eq, %whence, %two : i64
    %known_a = arith.ori %is_set, %is_cur : i1
    %known = arith.ori %known_a, %is_end : i1
    %true_bit = arith.constant true
    %unknown = arith.xori %known, %true_bit : i1
    scf.if %unknown {
      func.call @__ly_io_raise_bad_whence() : () -> ()
    }
    %handle = memref.load %self[%handle_slot] : memref<8xi64>
    %whence32 = arith.trunci %whence : i64 to i32
    %status = func.call @LyHost_FSeek(%handle, %pos, %whence32) : (i64, i64, i32) -> i32
    %where = func.call @LyHost_FTell(%handle) : (i64) -> i64
    %result:3 = func.call @LyLong_FromI64(%where) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyFileIO_Tell(%self: memref<8xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "_io.FileIO", ly.runtime.method = "tell", ly.runtime.result_contract = "builtins.int"} {
    %handle_slot = arith.constant 2 : index
    func.call @__ly_textio_check_open(%self) : (memref<8xi64>) -> ()
    %handle = memref.load %self[%handle_slot] : memref<8xi64>
    %where = func.call @LyHost_FTell(%handle) : (i64) -> i64
    %result:3 = func.call @LyLong_FromI64(%where) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyFileIO_Close(%self: memref<8xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "_io.FileIO", ly.runtime.method = "close", ly.runtime.result_contract = "types.NoneType"} {
    %handle_slot = arith.constant 2 : index
    %closed_slot = arith.constant 6 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %closed = memref.load %self[%closed_slot] : memref<8xi64>
    %already = arith.cmpi ne, %closed, %zero : i64
    scf.if %already {
    } else {
      %handle = memref.load %self[%handle_slot] : memref<8xi64>
      %has_handle = arith.cmpi ne, %handle, %zero : i64
      scf.if %has_handle {
        %status = func.call @LyHost_FClose(%handle) : (i64) -> i32
      }
      memref.store %one, %self[%closed_slot] : memref<8xi64>
    }
    func.return
  }

  func.func @LyFileIO_Fileno(%self: memref<8xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "_io.FileIO", ly.runtime.method = "fileno", ly.runtime.result_contract = "builtins.int"} {
    %handle_slot = arith.constant 2 : index
    func.call @__ly_textio_check_open(%self) : (memref<8xi64>) -> ()
    %handle = memref.load %self[%handle_slot] : memref<8xi64>
    %fd32 = func.call @LyHost_Fileno(%handle) : (i64) -> i32
    %fd = arith.extsi %fd32 : i32 to i64
    %result:3 = func.call @LyLong_FromI64(%fd) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyFileIO_Readable(%self: memref<8xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "_io.FileIO", ly.runtime.method = "readable", ly.runtime.result_contract = "builtins.bool"} {
    %readable_slot = arith.constant 4 : index
    %zero = arith.constant 0 : i64
    %readable = memref.load %self[%readable_slot] : memref<8xi64>
    %flag = arith.cmpi ne, %readable, %zero : i64
    func.return %flag : i1
  }

  func.func @LyFileIO_Writable(%self: memref<8xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "_io.FileIO", ly.runtime.method = "writable", ly.runtime.result_contract = "builtins.bool"} {
    %writable_slot = arith.constant 5 : index
    %zero = arith.constant 0 : i64
    %writable = memref.load %self[%writable_slot] : memref<8xi64>
    %flag = arith.cmpi ne, %writable, %zero : i64
    func.return %flag : i1
  }

  // truncate(size=None): flush, then ftruncate through the fd; the default
  // (INT64_MIN sentinel) truncates at the current position like fileio.c.
  func.func @LyFileIO_Truncate(%self: memref<8xi64> {ly.ownership.object_header}, %size: i64 {ly.runtime.default_i64 = -9223372036854775808 : i64}) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "_io.FileIO", ly.runtime.method = "truncate", ly.runtime.result_contract = "builtins.int"} {
    %handle_slot = arith.constant 2 : index
    %writable_slot = arith.constant 5 : index
    %zero = arith.constant 0 : i64
    %sentinel = arith.constant -9223372036854775808 : i64
    func.call @__ly_textio_check_open(%self) : (memref<8xi64>) -> ()
    %writable = memref.load %self[%writable_slot] : memref<8xi64>
    %not_writable = arith.cmpi eq, %writable, %zero : i64
    scf.if %not_writable {
      func.call @__ly_io_raise_not_writable() : () -> ()
    }
    %handle = memref.load %self[%handle_slot] : memref<8xi64>
    %is_default = arith.cmpi eq, %size, %sentinel : i64
    %current = func.call @LyHost_FTell(%handle) : (i64) -> i64
    %effective = arith.select %is_default, %current, %size : i64
    %negative = arith.cmpi slt, %effective, %zero : i64
    scf.if %negative {
      func.call @__ly_io_raise_neg_size() : () -> ()
    }
    %status = func.call @LyHost_FTruncate(%handle, %effective) : (i64, i64) -> i32
    %result:3 = func.call @LyLong_FromI64(%effective) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %result#0, %result#1, %result#2 : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyFileIO_Seekable(%self: memref<8xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "_io.FileIO", ly.runtime.method = "seekable", ly.runtime.result_contract = "builtins.bool"} {
    %true_bit = arith.constant true
    func.return %true_bit : i1
  }

  // CPython's FileIO inherits a no-op flush (the raw layer is unbuffered);
  // this FileIO sits on a FILE*, so flush() forwards to fflush to keep the
  // Buffered* wrappers' flush contract meaningful.
  func.func @LyFileIO_Flush(%self: memref<8xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "_io.FileIO", ly.runtime.method = "flush", ly.runtime.result_contract = "types.NoneType"} {
    %handle_slot = arith.constant 2 : index
    func.call @__ly_textio_check_open(%self) : (memref<8xi64>) -> ()
    %handle = memref.load %self[%handle_slot] : memref<8xi64>
    %status = func.call @LyHost_FFlush(%handle) : (i64) -> i32
    func.return
  }

  func.func @LyFileIO_DecRef(%self: memref<8xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "_io.FileIO", ly.runtime.deallocator} {
    %storage = memref.cast %self : memref<8xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    scf.if %became_zero {
      func.call @LyFileIO_Close(%self) : (memref<8xi64>) -> ()
      memref.dealloc %self : memref<8xi64>
    }
    func.return
  }

  // ===== impls: open / shared mode parsing =====
  // Scan the Python mode string: r/w/a/x pick the base mode, '+' adds
  // read-write, 't' is the default. 'b' is rejected unless the caller is a
  // binary stream (FileIO). Returns (base char, plus flag).
  func.func private @__ly_io_parse_mode(%mode_header: memref<2xi64>, %mode_bytes: memref<?xi8>, %binary_ok: i1) -> (i64, i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %mode_len = func.call @LyUnicode_Length(%mode_header, %mode_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %mode_len_index = arith.index_cast %mode_len : i64 to index
    %flags:3 = scf.for %i = %c0 to %mode_len_index step %c1 iter_args(%base = %zero, %plus = %zero, %bad = %zero) -> (i64, i64, i64) {
      %byte = memref.load %mode_bytes[%i] : memref<?xi8>
      %r = arith.constant 114 : i8
      %w = arith.constant 119 : i8
      %a = arith.constant 97 : i8
      %x = arith.constant 120 : i8
      %t = arith.constant 116 : i8
      %b = arith.constant 98 : i8
      %plus_ch = arith.constant 43 : i8
      %is_r = arith.cmpi eq, %byte, %r : i8
      %is_w = arith.cmpi eq, %byte, %w : i8
      %is_a = arith.cmpi eq, %byte, %a : i8
      %is_x = arith.cmpi eq, %byte, %x : i8
      %is_t = arith.cmpi eq, %byte, %t : i8
      %is_b = arith.cmpi eq, %byte, %b : i8
      %is_plus = arith.cmpi eq, %byte, %plus_ch : i8
      %true_bit = arith.constant true
      %is_base = arith.ori %is_r, %is_w : i1
      %is_base2 = arith.ori %is_base, %is_a : i1
      %is_base3 = arith.ori %is_base2, %is_x : i1
      %known = arith.ori %is_base3, %is_t : i1
      %known2 = arith.ori %known, %is_plus : i1
      %known3 = arith.ori %known2, %is_b : i1
      %base_set = arith.cmpi ne, %base, %zero : i64
      %dup_base = arith.andi %is_base3, %base_set : i1
      %byte_i64 = arith.extui %byte : i8 to i64
      %new_base = arith.select %is_base3, %byte_i64, %base : i64
      %new_plus = arith.select %is_plus, %one, %plus : i64
      // bad: 0 = ok, 1 = invalid mode, 2 = binary mode ('b' known but
      // unsupported until builtins.bytes exists).
      %binary_marker = arith.constant 2 : i64
      %bad_binary = arith.select %is_b, %binary_marker, %bad : i64
      %unknown = arith.xori %known3, %true_bit : i1
      %bad_unknown = arith.select %unknown, %one, %bad_binary : i64
      %bad_dup = arith.select %dup_base, %one, %bad_unknown : i64
      scf.yield %new_base, %new_plus, %bad_dup : i64, i64, i64
    }
    %no_base = arith.cmpi eq, %flags#0, %zero : i64
    %bad_unknown_or_dup = arith.cmpi eq, %flags#2, %one : i64
    %invalid = arith.ori %no_base, %bad_unknown_or_dup : i1
    scf.if %invalid {
      %class_id = arith.constant 53 : i64
      %length = arith.constant 12 : i64
      %static = memref.get_global @__ly_io_msg_invalid_mode : memref<12xi8>
      %message = memref.cast %static : memref<12xi8> to memref<?xi8>
      func.call @__ly_io_raise(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    }
    %two = arith.constant 2 : i64
    %binary = arith.cmpi eq, %flags#2, %two : i64
    %true_bit = arith.constant true
    %binary_rejected = arith.xori %binary_ok, %true_bit : i1
    %binary_bad = arith.andi %binary, %binary_rejected : i1
    scf.if %binary_bad {
      %class_id = arith.constant 53 : i64
      %length = arith.constant 28 : i64
      %static = memref.get_global @__ly_io_msg_binary_mode : memref<28xi8>
      %message = memref.cast %static : memref<28xi8> to memref<?xi8>
      func.call @__ly_io_raise(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    }
    func.return %flags#0, %flags#1 : i64, i64
  }

  // fopen the parsed mode ('b' appended for binary streams so Windows text
  // translation stays off), raising FileNotFoundError with the path when it
  // fails; returns the FILE* handle.
  func.func private @__ly_io_fopen(%path_header: memref<2xi64>, %path_bytes: memref<?xi8>, %base: i64, %plus: i64, %with_b: i1) -> i64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %cmode = memref.alloca() : memref<3xi8>
    %base_byte = arith.trunci %base : i64 to i8
    memref.store %base_byte, %cmode[%c0] : memref<3xi8>
    %has_plus = arith.cmpi ne, %plus, %zero : i64
    %plus_byte = arith.constant 43 : i8
    %b_byte = arith.constant 98 : i8
    %pad_byte = arith.constant 0 : i8
    %plus_or_pad = arith.select %has_plus, %plus_byte, %pad_byte : i8
    %second = arith.select %with_b, %b_byte, %plus_or_pad : i8
    %third_when_b = arith.select %has_plus, %plus_byte, %pad_byte : i8
    %third = arith.select %with_b, %third_when_b, %pad_byte : i8
    memref.store %second, %cmode[%c1] : memref<3xi8>
    memref.store %third, %cmode[%c2] : memref<3xi8>
    %plus_count = arith.select %has_plus, %one, %zero : i64
    %b_count = arith.select %with_b, %one, %zero : i64
    %len_a = arith.addi %one, %plus_count : i64
    %cmode_len = arith.addi %len_a, %b_count : i64
    %cmode_dyn = memref.cast %cmode : memref<3xi8> to memref<?xi8>

    %path_len = func.call @LyUnicode_Length(%path_header, %path_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %handle = func.call @LyHost_FOpen(%path_bytes, %path_len, %cmode_dyn, %cmode_len) : (memref<?xi8>, i64, memref<?xi8>, i64) -> i64
    %failed = arith.cmpi eq, %handle, %zero : i64
    scf.if %failed {
      // FileNotFoundError (67): "No such file or directory: <path>".
      %prefix_len = arith.constant 27 : index
      %prefix_len_i64 = arith.constant 27 : i64
      %path_len_index = arith.index_cast %path_len : i64 to index
      %total = arith.addi %prefix_len, %path_len_index : index
      %message = memref.alloc(%total) : memref<?xi8>
      %prefix_static = memref.get_global @__ly_io_msg_no_such_file : memref<27xi8>
      scf.for %i = %c0 to %prefix_len step %c1 {
        %byte = memref.load %prefix_static[%i] : memref<27xi8>
        memref.store %byte, %message[%i] : memref<?xi8>
      }
      scf.for %i = %c0 to %path_len_index step %c1 {
        %byte = memref.load %path_bytes[%i] : memref<?xi8>
        %dest = arith.addi %prefix_len, %i : index
        memref.store %byte, %message[%dest] : memref<?xi8>
      }
      %class_id = arith.constant 67 : i64
      %total_i64 = arith.addi %prefix_len_i64, %path_len : i64
      func.call @__ly_io_raise(%class_id, %message, %total_i64) : (i64, memref<?xi8>, i64) -> ()
    }
    func.return %handle : i64
  }

  // readable/writable per mode: r -> read (+ -> also write); w/a/x ->
  // write (+ -> also read).
  func.func private @__ly_io_mode_readable(%base: i64, %plus: i64) -> (i64, i64) {
    %one = arith.constant 1 : i64
    %r_byte = arith.constant 114 : i64
    %is_read_base = arith.cmpi eq, %base, %r_byte : i64
    %readable = arith.select %is_read_base, %one, %plus : i64
    %writable = arith.select %is_read_base, %plus, %one : i64
    func.return %readable, %writable : i64, i64
  }

  // open(file, mode) (clinic: _iomodule.c _io_open_impl restricted to the
  // text path): parse the mode string, fopen with the equivalent portable
  // C mode, wrap the FILE* in a TextIOWrapper.
  func.func @LyIO_Open(%path_header: memref<2xi64> {ly.ownership.object_header}, %path_bytes: memref<?xi8>, %mode_header: memref<2xi64> {ly.ownership.object_header, ly.runtime.default_str = "r"}, %mode_bytes: memref<?xi8>) -> memref<8xi64> attributes {ly.ownership.owned_result_contracts = ["_io.TextIOWrapper"], ly.ownership.owned_results = [0], ly.runtime.builtin = "_io.open", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.str", ly.runtime.primitive = "io_open", ly.runtime.result_contract = "_io.TextIOWrapper"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %false_bit = arith.constant false
    %base, %plus = func.call @__ly_io_parse_mode(%mode_header, %mode_bytes, %false_bit) : (memref<2xi64>, memref<?xi8>, i1) -> (i64, i64)
    %handle = func.call @__ly_io_fopen(%path_header, %path_bytes, %base, %plus, %false_bit) : (memref<2xi64>, memref<?xi8>, i64, i64, i1) -> i64
    %readable, %writable = func.call @__ly_io_mode_readable(%base, %plus) : (i64, i64) -> (i64, i64)

    %wrapper = memref.alloc() {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<8xi64>
    %refcount_slot = arith.constant 0 : index
    %class_slot = arith.constant 1 : index
    %handle_slot = arith.constant 2 : index
    %kind_slot = arith.constant 3 : index
    %readable_slot = arith.constant 4 : index
    %writable_slot = arith.constant 5 : index
    %closed_slot = arith.constant 6 : index
    %reserved_slot = arith.constant 7 : index
    %class_id = arith.constant 65 : i64
    memref.store %one, %wrapper[%refcount_slot] : memref<8xi64>
    memref.store %class_id, %wrapper[%class_slot] : memref<8xi64>
    memref.store %handle, %wrapper[%handle_slot] : memref<8xi64>
    memref.store %one, %wrapper[%kind_slot] : memref<8xi64>
    memref.store %readable, %wrapper[%readable_slot] : memref<8xi64>
    memref.store %writable, %wrapper[%writable_slot] : memref<8xi64>
    memref.store %zero, %wrapper[%closed_slot] : memref<8xi64>
    memref.store %zero, %wrapper[%reserved_slot] : memref<8xi64>
    func.return %wrapper : memref<8xi64>
  }

  // open(file, mode) with a 'b' mode: the binary arm the emitter selects
  // STATICALLY when the mode is a str literal containing 'b' (the return
  // type depends on the mode, which a runtime value cannot express in the
  // static surface). Returns the raw FileIO; CPython returns a Buffered*
  // wrapper, whose Lib/io.py port delegates to FileIO 1:1 here.
  func.func @LyIO_OpenBinary(%path_header: memref<2xi64> {ly.ownership.object_header}, %path_bytes: memref<?xi8>, %mode_header: memref<2xi64> {ly.ownership.object_header}, %mode_bytes: memref<?xi8>) -> memref<8xi64> attributes {ly.ownership.owned_result_contracts = ["_io.FileIO"], ly.ownership.owned_results = [0], ly.runtime.builtin = "_io.open_binary", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.str", ly.runtime.primitive = "io_open_binary", ly.runtime.result_contract = "_io.FileIO"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %true_bit = arith.constant true
    %base, %plus = func.call @__ly_io_parse_mode(%mode_header, %mode_bytes, %true_bit) : (memref<2xi64>, memref<?xi8>, i1) -> (i64, i64)
    %handle = func.call @__ly_io_fopen(%path_header, %path_bytes, %base, %plus, %true_bit) : (memref<2xi64>, memref<?xi8>, i64, i64, i1) -> i64
    %readable, %writable = func.call @__ly_io_mode_readable(%base, %plus) : (i64, i64) -> (i64, i64)

    %wrapper = memref.alloc() {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<8xi64>
    %refcount_slot = arith.constant 0 : index
    %class_slot = arith.constant 1 : index
    %handle_slot = arith.constant 2 : index
    %kind_slot = arith.constant 3 : index
    %readable_slot = arith.constant 4 : index
    %writable_slot = arith.constant 5 : index
    %closed_slot = arith.constant 6 : index
    %reserved_slot = arith.constant 7 : index
    %class_id = arith.constant 72 : i64
    memref.store %one, %wrapper[%refcount_slot] : memref<8xi64>
    memref.store %class_id, %wrapper[%class_slot] : memref<8xi64>
    memref.store %handle, %wrapper[%handle_slot] : memref<8xi64>
    memref.store %one, %wrapper[%kind_slot] : memref<8xi64>
    memref.store %readable, %wrapper[%readable_slot] : memref<8xi64>
    memref.store %writable, %wrapper[%writable_slot] : memref<8xi64>
    memref.store %zero, %wrapper[%closed_slot] : memref<8xi64>
    memref.store %zero, %wrapper[%reserved_slot] : memref<8xi64>
    func.return %wrapper : memref<8xi64>
  }
}
