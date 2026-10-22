// Contract manifest AND runtime implementation for reading the live call stack
// -- what CPython's `sys._getframe(level)` walks, cut to the three facts its
// readers in the standard library use: the file, the line and the module.
//
// WHY A MODULE OF ITS OWN and not `sys._getframe`: a frame object carries the
// code object, the globals and the locals, which a compiled program does not
// have, and `f_back` would make every frame reachable from the one asked for.
// lib/warnings.py and lib/logging read the three facts here instead; CPython
// has no `_frame` module, the split is Lython's, as `_traceback` is.
//
// ⭐ THE STACK HOLDS ONLY THE FRAMES OF FUNCTIONS THAT CAN REACH A READ. The
// frame pass writes a function's frame before each call that can reach one of
// these reads, and in no other function (lowering/Common/FrameSupportBuilder.cpp).
// A frame from a function that cannot reach one is never asked for: whatever
// asks is above it on the stack, and so reachable from it.
//
// Level 0 is the function that calls the read, at the line of that call;
// level 1 its caller, at the line of the call it is in; and so on.
// Deviations from CPython:
// - a level past the bottom answers `depth() <= level` rather than raising
//   ValueError; the readers test it first, as CPython's catch the error.
// - module code that is not the main module's has no frame name of its own
//   here; its frames are named `<module>` like the main module's.
module attributes {
  ly.typing.module = "_frame",
  ly.typing.callable_exports = ["_frame.depth", "_frame.file", "_frame.name", "_frame.module", "_frame.line"],
  ly.typing.function_names = ["_frame.depth", "_frame.file", "_frame.name", "_frame.module", "_frame.line"],
  ly.typing.function_contracts = [
    !py.callable<[], arg_names = [], arg_defaults = [], returns = [!py.contract<"builtins.int">]>,
    !py.callable<[!py.contract<"builtins.int">], arg_names = ["level"], arg_defaults = [false], returns = [!py.contract<"builtins.str">]>,
    !py.callable<[!py.contract<"builtins.int">], arg_names = ["level"], arg_defaults = [false], returns = [!py.contract<"builtins.str">]>,
    !py.callable<[!py.contract<"builtins.int">], arg_names = ["level"], arg_defaults = [false], returns = [!py.contract<"builtins.str">]>,
    !py.callable<[!py.contract<"builtins.int">], arg_names = ["level"], arg_defaults = [false], returns = [!py.contract<"builtins.int">]>
  ]
} {
  func.func private @__ly_unicode_from_valid_utf8(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]}
  func.func private @LyLong_FromI64(%value: i64) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class, ly.runtime.contract = "builtins.int", ly.runtime.initializer = "__new__"}

  func.func private @LyFrame_Size() -> i64
  func.func private @LyFrame_Line(i64) -> i64
  func.func private @LyFrame_TextLen(i64, i64) -> i64
  func.func private @LyFrame_TextCopy(i64, i64, memref<?xi8>, i64)

  // The frames above the read's caller and the caller's own: how deep a
  // level can go.
  func.func @LyFrameMod_Depth() -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "_frame.depth", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.int", ly.runtime.primitive = "frame_depth", ly.runtime.result_contract = "builtins.int"} {
    %size = func.call @LyFrame_Size() : () -> i64
    %h = func.call @LyLong_FromI64(%size) : (i64) -> memref<2xi64>
    func.return %h : memref<2xi64>
  }

  func.func @LyFrameMod_Line(%level: i64) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "_frame.line", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.int", ly.runtime.primitive = "frame_line", ly.runtime.result_contract = "builtins.int"} {
    %line = func.call @LyFrame_Line(%level) : (i64) -> i64
    %h = func.call @LyLong_FromI64(%line) : (i64) -> memref<2xi64>
    func.return %h : memref<2xi64>
  }

  func.func @LyFrameMod_File(%level: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "_frame.file", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.str", ly.runtime.primitive = "frame_file", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %field = arith.constant 0 : i64
    %len = func.call @LyFrame_TextLen(%level, %field) : (i64, i64) -> i64
    %len_index = arith.index_cast %len : i64 to index
    %buffer = memref.alloc(%len_index) : memref<?xi8>
    func.call @LyFrame_TextCopy(%level, %field, %buffer, %len) : (i64, i64, memref<?xi8>, i64) -> ()
    %header, %bytes = func.call @__ly_unicode_from_valid_utf8(%buffer, %c0, %len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    memref.dealloc %buffer : memref<?xi8>
    func.return %header, %bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyFrameMod_Name(%level: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "_frame.name", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.str", ly.runtime.primitive = "frame_name", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %field = arith.constant 1 : i64
    %len = func.call @LyFrame_TextLen(%level, %field) : (i64, i64) -> i64
    %len_index = arith.index_cast %len : i64 to index
    %buffer = memref.alloc(%len_index) : memref<?xi8>
    func.call @LyFrame_TextCopy(%level, %field, %buffer, %len) : (i64, i64, memref<?xi8>, i64) -> ()
    %header, %bytes = func.call @__ly_unicode_from_valid_utf8(%buffer, %c0, %len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    memref.dealloc %buffer : memref<?xi8>
    func.return %header, %bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @LyFrameMod_Module(%level: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "_frame.module", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.str", ly.runtime.primitive = "frame_module", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %field = arith.constant 2 : i64
    %len = func.call @LyFrame_TextLen(%level, %field) : (i64, i64) -> i64
    %len_index = arith.index_cast %len : i64 to index
    %buffer = memref.alloc(%len_index) : memref<?xi8>
    func.call @LyFrame_TextCopy(%level, %field, %buffer, %len) : (i64, i64, memref<?xi8>, i64) -> ()
    %header, %bytes = func.call @__ly_unicode_from_valid_utf8(%buffer, %c0, %len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    memref.dealloc %buffer : memref<?xi8>
    func.return %header, %bytes : memref<2xi64>, memref<?xi8>
  }
}
