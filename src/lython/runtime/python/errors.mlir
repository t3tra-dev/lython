// Raising -- CPython's Python/errors.c: PyErr_SetString (a class id and a
// message, `__ly_raise_static_message`) and PyErr_NoMemory.

module {
  // ===== declared here, defined in another runtime file or built by the lowering =====
  func.func private @LyBaseException_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.BaseException", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"}
  func.func private @LyBaseException_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 5 : i64, ly.runtime.contract = "builtins.BaseException", ly.runtime.initializer = "__new__"}
  func.func private @LyEH_ThrowException(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "raise"}
  func.func private @LyMemoryError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 109 : i64, ly.runtime.contract = "builtins.MemoryError", ly.runtime.initializer = "__new__"}
  func.func private @LyMemoryError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.MemoryError", ly.runtime.primitive = "raise"}
  func.func private @LyUnicode_FromBytes(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 4 : i64, ly.runtime.contract = "builtins.str", ly.runtime.initializer = "__new__"}

  // Raise `class_id` carrying an already-built str. The other manifests reach
  // this one: posix must free its formatted buffer between building the str
  // and throwing, and _time raises a strftime message it computed.
  func.func private @__ly_raise_message_object(%class_id: i64, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) {
    %exception:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    %initialized:3 = func.call @LyBaseException_Init(%exception#0, %exception#1, %exception#2, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.call @LyEH_ThrowException(%initialized#0, %initialized#1, %initialized#2) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  // ⭐ THE ONE STATIC-MESSAGE RAISE. `raise <class>("literal")` from a native
  // body is this: make the exception, make the str, init, throw. It was
  // written twice (once as `__ly_long_raise_message`, once as
  // `__ly_unicode_raise`) and inlined again in five argument-less helpers;
  // every runtime file declares this one instead.
  func.func private @__ly_raise_static_message(%class_id: i64, %message: memref<?xi8>, %length: i64) {
    %start = arith.constant 0 : index
    %message_header, %message_bytes = func.call @LyUnicode_FromBytes(%message, %start, %length) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    func.call @__ly_raise_message_object(%class_id, %message_header, %message_bytes) : (i64, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  // `MemoryError()`, with no arguments, as CPython's PyErr_NoMemory: what a
  // size the program computes past the allocator's reach raises
  // (`__ly_check_alloc_count`, the repeats).
  func.func private @LyErr_NoMemory() attributes {ly.runtime.contract = "builtins.MemoryError"} {
    %class_id = arith.constant 109 : i64
    %exception:3 = func.call @LyMemoryError_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.call @LyMemoryError_Raise(%exception#0, %exception#1, %exception#2) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }
}
