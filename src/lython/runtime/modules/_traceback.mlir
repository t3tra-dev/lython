// Contract manifest AND runtime implementation for reading the traceback frame
// stack (CPython's Python/traceback.c counterpart, from the other side).
//
// WHY THIS IS `_traceback` AND NOT `traceback`, when CPython implements all of
// `traceback` in Python: the frames a traceback is made of are recorded at
// raise sites by the runtime's own stack (lowering/Common/TracebackSupportBuilder.cpp,
// `g_traceback_stack`), and a Python module cannot reach a process global. So
// the module is split the way `time` is: this file is the native half, and
// lib/traceback.py is the public layer that builds `types.TracebackType` links
// out of what it reads here. CPython has no `_traceback` module; the split is
// Lython's, and lib/traceback.py's docstring records it.
//
// ⭐ THE STACK IS THE IN-FLIGHT EXCEPTION'S. A frame is pushed where a raise
// passes through and the whole stack is cleared once the exception is handled,
// so `frame_count()` answers about the exception being handled RIGHT NOW --
// inside an `except` block -- and answers 0 outside one. That is the same
// state the uncaught-exception printer walks, which is why a caught traceback
// and a printed one cannot disagree.
//
// Deviations from CPython:
// - a frame carries its file, its function name and one line. CPython's frame
//   also carries the code object's bytecode, its globals and its locals; a
//   compiled program has none of those, and lib/types.py records the same
//   deviation for the classes built from these reads.
// - `frame_line` answers 0 and the two name reads answer '' for an index the
//   stack does not have, rather than raising IndexError. A reader that raced
//   the stack gets an empty frame, never a wild read.

module attributes {
  ly.typing.module = "_traceback",
  ly.typing.callable_exports = [
    "_traceback.frame_count",
    "_traceback.frame_file",
    "_traceback.frame_name",
    "_traceback.frame_line",
    "_traceback.frame_col",
    "_traceback.frame_end_col",
    "_traceback.frame_marker",
    "_traceback.exc_line",
    "_traceback.chain_count",
    "_traceback.chain_kind",
    "_traceback.chain_select"
  ],
  ly.typing.function_names = [
    "_traceback.frame_count",
    "_traceback.frame_file",
    "_traceback.frame_name",
    "_traceback.frame_line",
    "_traceback.frame_col",
    "_traceback.frame_end_col",
    "_traceback.frame_marker",
    "_traceback.exc_line",
    "_traceback.chain_count",
    "_traceback.chain_kind",
    "_traceback.chain_select"
  ],
  ly.typing.function_contracts = [
    !py.callable<[], arg_names = [], arg_defaults = [], returns = [!py.contract<"builtins.int">]>,
    !py.callable<[!py.contract<"builtins.int">], arg_names = ["index"], arg_defaults = [false], returns = [!py.contract<"builtins.str">]>,
    !py.callable<[!py.contract<"builtins.int">], arg_names = ["index"], arg_defaults = [false], returns = [!py.contract<"builtins.str">]>,
    !py.callable<[!py.contract<"builtins.int">], arg_names = ["index"], arg_defaults = [false], returns = [!py.contract<"builtins.int">]>,
    !py.callable<[!py.contract<"builtins.int">], arg_names = ["index"], arg_defaults = [false], returns = [!py.contract<"builtins.int">]>,
    !py.callable<[!py.contract<"builtins.int">], arg_names = ["index"], arg_defaults = [false], returns = [!py.contract<"builtins.int">]>,
    !py.callable<[!py.contract<"builtins.int">], arg_names = ["index"], arg_defaults = [false], returns = [!py.contract<"builtins.int">]>,
    !py.callable<[], arg_names = [], arg_defaults = [], returns = [!py.contract<"builtins.str">]>,
    !py.callable<[], arg_names = [], arg_defaults = [], returns = [!py.contract<"builtins.int">]>,
    !py.callable<[!py.contract<"builtins.int">], arg_names = ["level"], arg_defaults = [false], returns = [!py.contract<"builtins.int">]>,
    !py.callable<[!py.contract<"builtins.int">], arg_names = ["level"], arg_defaults = [false], returns = [!py.contract<"builtins.int">]>
  ]
} {
  func.func private @LyLong_FromI64(%value: i64) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 2 : i64, ly.runtime.contract = "builtins.int", ly.runtime.initializer = "__new__"}
  func.func private @LyUnicode_FromBytes(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 4 : i64, ly.runtime.contract = "builtins.str", ly.runtime.initializer = "__new__"}

  func.func private @LyTraceback_FrameCount() -> i64
  func.func private @LyTraceback_FrameLine(i64) -> i64
  func.func private @LyTraceback_FrameCol(i64) -> i64
  func.func private @LyTraceback_FrameEndCol(i64) -> i64
  func.func private @LyTraceback_FrameMarker(i64) -> i64
  func.func private @LyTraceback_FrameFileLen(i64) -> i64
  func.func private @LyTraceback_FrameNameLen(i64) -> i64
  func.func private @LyTraceback_FrameFileCopy(i64, memref<?xi8>, i64)
  func.func private @LyTraceback_FrameNameCopy(i64, memref<?xi8>, i64)
  func.func private @LyTraceback_ChainCount() -> i64
  func.func private @LyTraceback_ChainKind(i64) -> i64
  func.func private @LyTraceback_ChainSelect(i64)
  func.func private @LyTraceback_ExcLineLen() -> i64
  func.func private @LyTraceback_ExcLineCopy(memref<?xi8>, i64)

  // The chained exceptions ahead of the one being handled: how many sections
  // the uncaught printer would write before its own, how each is attached, and
  // which one the frame/exc-line accessors answer from. A negative level
  // restores the live stack.
  func.func @LyTracebackMod_ChainCount() -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "_traceback.chain_count", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.int", ly.runtime.primitive = "traceback_chain_count", ly.runtime.result_contract = "builtins.int"} {
    %count = func.call @LyTraceback_ChainCount() : () -> i64
    %h = func.call @LyLong_FromI64(%count) : (i64) -> memref<2xi64>
    func.return %h : memref<2xi64>
  }

  func.func @LyTracebackMod_ChainKind(%level: i64) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "_traceback.chain_kind", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.int", ly.runtime.primitive = "traceback_chain_kind", ly.runtime.result_contract = "builtins.int"} {
    %kind = func.call @LyTraceback_ChainKind(%level) : (i64) -> i64
    %h = func.call @LyLong_FromI64(%kind) : (i64) -> memref<2xi64>
    func.return %h : memref<2xi64>
  }

  func.func @LyTracebackMod_ChainSelect(%level: i64) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "_traceback.chain_select", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.int", ly.runtime.primitive = "traceback_chain_select", ly.runtime.result_contract = "builtins.int"} {
    func.call @LyTraceback_ChainSelect(%level) : (i64) -> ()
    %zero = arith.constant 0 : i64
    %h = func.call @LyLong_FromI64(%zero) : (i64) -> memref<2xi64>
    func.return %h : memref<2xi64>
  }

  func.func @LyTracebackMod_FrameCount() -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "_traceback.frame_count", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.int", ly.runtime.primitive = "traceback_frame_count", ly.runtime.result_contract = "builtins.int"} {
    %count = func.call @LyTraceback_FrameCount() : () -> i64
    %h = func.call @LyLong_FromI64(%count) : (i64) -> memref<2xi64>
    func.return %h : memref<2xi64>
  }

  func.func @LyTracebackMod_FrameLine(%index: i64) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "_traceback.frame_line", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.int", ly.runtime.primitive = "traceback_frame_line", ly.runtime.result_contract = "builtins.int"} {
    %line = func.call @LyTraceback_FrameLine(%index) : (i64) -> i64
    %h = func.call @LyLong_FromI64(%line) : (i64) -> memref<2xi64>
    func.return %h : memref<2xi64>
  }

  func.func @LyTracebackMod_FrameFile(%index: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "_traceback.frame_file", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.str", ly.runtime.primitive = "traceback_frame_file", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %len = func.call @LyTraceback_FrameFileLen(%index) : (i64) -> i64
    %len_index = arith.index_cast %len : i64 to index
    %buffer = memref.alloc(%len_index) : memref<?xi8>
    func.call @LyTraceback_FrameFileCopy(%index, %buffer, %len) : (i64, memref<?xi8>, i64) -> ()
    %header, %bytes = func.call @LyUnicode_FromBytes(%buffer, %c0, %len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    memref.dealloc %buffer : memref<?xi8>
    func.return %header, %bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyTracebackMod_FrameName(%index: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "_traceback.frame_name", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.str", ly.runtime.primitive = "traceback_frame_name", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %len = func.call @LyTraceback_FrameNameLen(%index) : (i64) -> i64
    %len_index = arith.index_cast %len : i64 to index
    %buffer = memref.alloc(%len_index) : memref<?xi8>
    func.call @LyTraceback_FrameNameCopy(%index, %buffer, %len) : (i64, memref<?xi8>, i64) -> ()
    %header, %bytes = func.call @LyUnicode_FromBytes(%buffer, %c0, %len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    memref.dealloc %buffer : memref<?xi8>
    func.return %header, %bytes : memref<2xi64>, memref<?xi8>
  }

  // The last line of a traceback -- "ValueError: boom" -- for the exception
  // being handled, or '' when there is none. Built from the class id the
  // handler dispatched on and the message re-encoder the uncaught printer
  // uses, so the two cannot word it differently.
  func.func @LyTracebackMod_ExcLine() -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "_traceback.exc_line", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.str", ly.runtime.primitive = "traceback_exc_line", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %len = func.call @LyTraceback_ExcLineLen() : () -> i64
    %len_index = arith.index_cast %len : i64 to index
    %buffer = memref.alloc(%len_index) : memref<?xi8>
    func.call @LyTraceback_ExcLineCopy(%buffer, %len) : (memref<?xi8>, i64) -> ()
    %header, %bytes = func.call @LyUnicode_FromBytes(%buffer, %c0, %len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    memref.dealloc %buffer : memref<?xi8>
    func.return %header, %bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyTracebackMod_FrameCol(%index: i64) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "_traceback.frame_col", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.int", ly.runtime.primitive = "traceback_frame_col", ly.runtime.result_contract = "builtins.int"} {
    %word = func.call @LyTraceback_FrameCol(%index) : (i64) -> i64
    %h = func.call @LyLong_FromI64(%word) : (i64) -> memref<2xi64>
    func.return %h : memref<2xi64>
  }

  func.func @LyTracebackMod_FrameEndCol(%index: i64) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "_traceback.frame_end_col", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.int", ly.runtime.primitive = "traceback_frame_end_col", ly.runtime.result_contract = "builtins.int"} {
    %word = func.call @LyTraceback_FrameEndCol(%index) : (i64) -> i64
    %h = func.call @LyLong_FromI64(%word) : (i64) -> memref<2xi64>
    func.return %h : memref<2xi64>
  }

  func.func @LyTracebackMod_FrameMarker(%index: i64) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "_traceback.frame_marker", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.int", ly.runtime.primitive = "traceback_frame_marker", ly.runtime.result_contract = "builtins.int"} {
    %word = func.call @LyTraceback_FrameMarker(%index) : (i64) -> i64
    %h = func.call @LyLong_FromI64(%word) : (i64) -> memref<2xi64>
    func.return %h : memref<2xi64>
  }
}
