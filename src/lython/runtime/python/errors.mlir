// Raising -- CPython's Python/errors.c: PyErr_SetString (a class word and a
// message, `__ly_raise_static_message`) and PyErr_NoMemory.

module {
  // ===== declared here, defined in another runtime file or built by the lowering =====
  func.func private @__ly_global_view_i64(%pointer: i64, %size: i64) -> memref<?xi64>
  func.func private @LyObject_ReleaseBoxedPayloadRaw(%box: memref<5xi64>)
  func.func private @LyList_DecRef(%self: memref<5xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.list", ly.runtime.primitive = "decref"}
  func.func private @LySet_DecRef(%self: memref<9xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.set", ly.runtime.deallocator}
  func.func private @LyFrozenSet_DecRef(%self: memref<9xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.frozenset", ly.runtime.deallocator}
  func.func private @__ly_unicode_from_valid_utf8(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]}
  func.func private @LyBaseException_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.BaseException", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"}
  func.func private @LyBaseException_New(%class_word: i64 {ly.runtime.class_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class, ly.runtime.contract = "builtins.BaseException", ly.runtime.initializer = "__new__"}
  func.func private @LyEH_ThrowException(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "raise"}
  func.func private @LyMemoryError_New(%class_word: i64 {ly.runtime.class_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class, ly.runtime.contract = "builtins.MemoryError", ly.runtime.initializer = "__new__"}
  func.func private @LyMemoryError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.MemoryError", ly.runtime.primitive = "raise"}
  func.func private @LyUnicode_FromBytes(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class, ly.runtime.contract = "builtins.str", ly.runtime.initializer = "__new__"}

  // Raise `class_id` carrying an already-built str. The other manifests reach
  // this one: posix must free its formatted buffer between building the str
  // and throwing, and _time raises a strftime message it computed.
  func.func private @__ly_raise_message_object(%class_word: i64, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) {
    %exception:3 = func.call @LyBaseException_New(%class_word) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    %initialized:3 = func.call @LyBaseException_Init(%exception#0, %exception#1, %exception#2, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.call @LyEH_ThrowException(%initialized#0, %initialized#1, %initialized#2) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  // ⭐ THE ONE STATIC-MESSAGE RAISE. `raise <class>("literal")` from a native
  // body is this: make the exception, make the str, init, throw. It was
  // written twice (once as `__ly_long_raise_message`, once as
  // `__ly_unicode_raise`) and inlined again in five argument-less helpers;
  // every runtime file declares this one instead.
  func.func private @__ly_raise_static_message(%class_word: i64, %message: memref<?xi8>, %length: i64) {
    %start = arith.constant 0 : index
    %message_header, %message_bytes = func.call @__ly_unicode_from_valid_utf8(%message, %start, %length) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    func.call @__ly_raise_message_object(%class_word, %message_header, %message_bytes) : (i64, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  // ===== what a native body owes if an exception leaves it =====
  //
  // CPython's C functions release what they own on the error path before
  // returning NULL. A native body here cannot: an exception unwinds through
  // it, it has no landing pad, and what it held -- the copy `sorted` is
  // sorting, the key and value a dict insertion was handed -- was lost. So a
  // body REGISTERS what it would owe before the call that may raise, and pops
  // it once that call is past; a frame that catches releases every entry
  // registered below it (`LyEH_ReleasePendingBelow`, called by the catch
  // dispatch the EH lowering builds).
  //
  // An entry carries the address of a slot in the registering body's own
  // frame -- the "mark" -- which is what says which frame it belongs to: a
  // catcher releases the entries whose mark is below its stack pointer,
  // which are exactly the frames the exception abandoned. A registering body
  // is kept out of line (driver/lib/LLVMFinalize.cpp) so its mark is in a
  // frame of its own.
  //
  // ⛔ The entries live HERE and not in the registering frame: by the time a
  // catch runs, the abandoned frames are free stack, and the release's own
  // call would overwrite a list threaded through them.
  //
  // Kinds: 1 a box whose payload reference it owns (`value` is the box's
  // address), 2 a list it owns, 3 a set, 4 a frozenset (`value` is the
  // handle's address).
  memref.global "private" @__ly_pending_marks : memref<256xi64> = dense<0>
  memref.global "private" @__ly_pending_kinds : memref<256xi64> = dense<0>
  memref.global "private" @__ly_pending_values : memref<256xi64> = dense<0>
  memref.global "private" @__ly_pending_count : memref<1xi64> = dense<0>

  // Register; answers the entry's index, or -1 when the table is full -- an
  // entry not taken is a leak on the error path and nothing else.
  func.func private @__ly_pending_push(%mark: i64, %kind: i64, %value: i64) -> i64 {
    %c0 = arith.constant 0 : index
    %minus_one = arith.constant -1 : i64
    %capacity = arith.constant 256 : i64
    %one = arith.constant 1 : i64
    %count_global = memref.get_global @__ly_pending_count : memref<1xi64>
    %count = memref.load %count_global[%c0] : memref<1xi64>
    %full = arith.cmpi uge, %count, %capacity : i64
    %index = scf.if %full -> (i64) {
      scf.yield %minus_one : i64
    } else {
      %slot = arith.index_cast %count : i64 to index
      %marks = memref.get_global @__ly_pending_marks : memref<256xi64>
      %kinds = memref.get_global @__ly_pending_kinds : memref<256xi64>
      %values = memref.get_global @__ly_pending_values : memref<256xi64>
      memref.store %mark, %marks[%slot] : memref<256xi64>
      memref.store %kind, %kinds[%slot] : memref<256xi64>
      memref.store %value, %values[%slot] : memref<256xi64>
      %next = arith.addi %count, %one : i64
      memref.store %next, %count_global[%c0] : memref<1xi64>
      scf.yield %count : i64
    }
    func.return %index : i64
  }

  // Unregister the entry `push` answered and every one above it.
  func.func private @__ly_pending_pop(%index: i64) {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0 : i64
    %taken = arith.cmpi sge, %index, %zero : i64
    scf.if %taken {
      %count_global = memref.get_global @__ly_pending_count : memref<1xi64>
      memref.store %index, %count_global[%c0] : memref<1xi64>
    }
    func.return
  }

  func.func private @__ly_pending_release(%kind: i64, %value: i64) {
    %five = arith.constant 5 : i64
    %box_kind = arith.constant 1 : i64
    %list_kind = arith.constant 2 : i64
    %view = func.call @__ly_global_view_i64(%value, %five) : (i64, i64) -> memref<?xi64>
    %handle = memref.cast %view : memref<?xi64> to memref<5xi64>
    %is_box = arith.cmpi eq, %kind, %box_kind : i64
    scf.if %is_box {
      func.call @LyObject_ReleaseBoxedPayloadRaw(%handle) : (memref<5xi64>) -> ()
    } else {
      %is_list = arith.cmpi eq, %kind, %list_kind : i64
      scf.if %is_list {
        func.call @LyList_DecRef(%handle) : (memref<5xi64>) -> ()
      } else {
        %nine = arith.constant 9 : i64
        %set_view = func.call @__ly_global_view_i64(%value, %nine) : (i64, i64) -> memref<?xi64>
        %set_handle = memref.cast %set_view : memref<?xi64> to memref<9xi64>
        %set_kind = arith.constant 3 : i64
        %frozenset_kind = arith.constant 4 : i64
        %is_set = arith.cmpi eq, %kind, %set_kind : i64
        %is_frozenset = arith.cmpi eq, %kind, %frozenset_kind : i64
        scf.if %is_set {
          func.call @LySet_DecRef(%set_handle) : (memref<9xi64>) -> ()
        }
        scf.if %is_frozenset {
          func.call @LyFrozenSet_DecRef(%set_handle) : (memref<9xi64>) -> ()
        }
      }
    }
    func.return
  }

  // Release every entry whose mark is below `sp` -- the catching frame's
  // stack pointer -- newest first.
  func.func @LyEH_ReleasePendingBelow(%sp: i64) {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %count_global = memref.get_global @__ly_pending_count : memref<1xi64>
    %marks = memref.get_global @__ly_pending_marks : memref<256xi64>
    %kinds = memref.get_global @__ly_pending_kinds : memref<256xi64>
    %values = memref.get_global @__ly_pending_values : memref<256xi64>
    scf.while : () -> () {
      %count = memref.load %count_global[%c0] : memref<1xi64>
      %nonempty = arith.cmpi sgt, %count, %zero : i64
      %below = scf.if %nonempty -> (i1) {
        %top = arith.subi %count, %one : i64
        %slot = arith.index_cast %top : i64 to index
        %mark = memref.load %marks[%slot] : memref<256xi64>
        %deeper = arith.cmpi ult, %mark, %sp : i64
        scf.yield %deeper : i1
      } else {
        %no = arith.constant false
        scf.yield %no : i1
      }
      scf.condition(%below)
    } do {
      %count = memref.load %count_global[%c0] : memref<1xi64>
      %top = arith.subi %count, %one : i64
      %slot = arith.index_cast %top : i64 to index
      %kind = memref.load %kinds[%slot] : memref<256xi64>
      %value = memref.load %values[%slot] : memref<256xi64>
      memref.store %top, %count_global[%c0] : memref<1xi64>
      func.call @__ly_pending_release(%kind, %value) : (i64, i64) -> ()
      scf.yield
    }
    func.return
  }

  // `MemoryError()`, with no arguments, as CPython's PyErr_NoMemory: what a
  // size the program computes past the allocator's reach raises
  // (`__ly_check_alloc_count`, the repeats).
  func.func private @LyErr_NoMemory() attributes {ly.runtime.contract = "builtins.MemoryError"} {
    %class_word = arith.constant {ly.class_of = "builtins.MemoryError"} 109 : i64
    %exception:3 = func.call @LyMemoryError_New(%class_word) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.call @LyMemoryError_Raise(%exception#0, %exception#1, %exception#2) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }
}
