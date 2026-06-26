// Minimal CPython-shaped coroutine runtime object.
//
// A coroutine object is created by calling an async function. The function body
// is not executed at creation time; `py.await` resumes the coroutine frame.
// This file owns the object/state ABI; compiler lowering owns the exact frame
// body and currently materializes it from static call evidence. The saved resume
// index is part of the ABI so later async.coro.suspend lowering can spill the
// active continuation without inventing a second coroutine representation.

module attributes {ly.runtime.contracts = ["types.CoroutineType", "types.CoroutineAwaitIterator"]} {
  func.func private @Ly_IncRef(%header: memref<2xi64, strided<[1], offset: ?>> {ly.ownership.object_header})
  func.func private @LyObject_DecRefHeader(%header: memref<2xi64, strided<[1], offset: ?>> {ly.ownership.object_header}) -> i1

  func.func @LyCoroutine_New(%target_id: i64) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 9 : i64, ly.runtime.contract = "types.CoroutineType", ly.runtime.initializer = "__new__"} {
    %one = arith.constant 1 : i64
    %zero = arith.constant 0 : i64
    %layout_coroutine = arith.constant 9 : i64
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %state_slot = arith.constant 2 : index
    %target_slot = arith.constant 3 : index
    %resume_slot = arith.constant 4 : index

    %storage = memref.alloc() {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<5xi64>
    memref.store %one, %storage[%refcount_slot] : memref<5xi64>
    memref.store %layout_coroutine, %storage[%layout_slot] : memref<5xi64>
    // State 0 = created, 1 = running, 2 = suspended, 3 = completed, 4 = closed.
    memref.store %zero, %storage[%state_slot] : memref<5xi64>
    memref.store %target_id, %storage[%target_slot] : memref<5xi64>
    // Resume index 0 is the initial entry. Future suspension points write the
    // switched-resume continuation index here before handing control to a Task.
    memref.store %zero, %storage[%resume_slot] : memref<5xi64>
    func.return %storage : memref<5xi64>
  }

  func.func @LyCoroutine_ResumeBegin(%storage: memref<5xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "types.CoroutineType", ly.runtime.primitive = "resume.begin"} {
    %state_slot = arith.constant 2 : index
    %created = arith.constant 0 : i64
    %running = arith.constant 1 : i64
    %suspended = arith.constant 2 : i64
    %previous = memref.generic_atomic_rmw %storage[%state_slot] : memref<5xi64> {
    ^bb0(%current : i64):
      %is_created = arith.cmpi eq, %current, %created : i64
      %is_suspended = arith.cmpi eq, %current, %suspended : i64
      %can_enter = arith.ori %is_created, %is_suspended : i1
      %next = arith.select %can_enter, %running, %current : i1, i64
      memref.atomic_yield %next : i64
    } {ly.atomic.ordering = "acq_rel", ly.atomic.role = "coroutine.state.resume_begin"}
    %was_created = arith.cmpi eq, %previous, %created : i64
    %was_suspended = arith.cmpi eq, %previous, %suspended : i64
    %can_resume = arith.ori %was_created, %was_suspended : i1
    func.return %can_resume : i1
  }

  func.func @LyCoroutine_ResumeComplete(%storage: memref<5xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "types.CoroutineType", ly.runtime.primitive = "resume.complete"} {
    %state_slot = arith.constant 2 : index
    %running = arith.constant 1 : i64
    %completed = arith.constant 3 : i64
    %previous = memref.generic_atomic_rmw %storage[%state_slot] : memref<5xi64> {
    ^bb0(%current : i64):
      %is_running = arith.cmpi eq, %current, %running : i64
      %next = arith.select %is_running, %completed, %current : i1, i64
      memref.atomic_yield %next : i64
    } {ly.atomic.ordering = "acq_rel", ly.atomic.role = "coroutine.state.resume_complete"}
    %was_running = arith.cmpi eq, %previous, %running : i64
    cf.assert %was_running, "LyCoroutine_ResumeComplete called for a coroutine that is not running"
    func.return
  }

  func.func @LyCoroutine_Suspend(%storage: memref<5xi64> {ly.ownership.object_header}, %resume_index: i64) attributes {ly.runtime.contract = "types.CoroutineType", ly.runtime.primitive = "resume.suspend"} {
    %state_slot = arith.constant 2 : index
    %resume_slot = arith.constant 4 : index
    %running = arith.constant 1 : i64
    %suspended = arith.constant 2 : i64

    memref.store %resume_index, %storage[%resume_slot] : memref<5xi64>
    %previous = memref.generic_atomic_rmw %storage[%state_slot] : memref<5xi64> {
    ^bb0(%current : i64):
      %is_running = arith.cmpi eq, %current, %running : i64
      %next = arith.select %is_running, %suspended, %current : i1, i64
      memref.atomic_yield %next : i64
    } {ly.atomic.ordering = "acq_rel", ly.atomic.role = "coroutine.state.resume_suspend"}
    %was_running = arith.cmpi eq, %previous, %running : i64
    cf.assert %was_running, "LyCoroutine_Suspend called for a coroutine that is not running"
    func.return
  }

  func.func @LyCoroutine_Close(%storage: memref<5xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "types.CoroutineType", ly.runtime.method = "close", ly.runtime.result_contract = "types.NoneType"} {
    %state_slot = arith.constant 2 : index
    %running = arith.constant 1 : i64
    %closed = arith.constant 4 : i64

    %previous = memref.generic_atomic_rmw %storage[%state_slot] : memref<5xi64> {
    ^bb0(%current : i64):
      %is_running = arith.cmpi eq, %current, %running : i64
      %next = arith.select %is_running, %current, %closed : i1, i64
      memref.atomic_yield %next : i64
    } {ly.atomic.ordering = "acq_rel", ly.atomic.role = "coroutine.state.close"}
    %was_running = arith.cmpi eq, %previous, %running : i64
    %true = arith.constant true
    %can_close = arith.xori %was_running, %true : i1
    cf.assert %can_close, "cannot close a running coroutine"
    func.return
  }

  func.func @LyCoroutine_Await(%storage: memref<5xi64> {ly.ownership.object_header}) -> (memref<3xi64>, memref<5xi64>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "types.CoroutineType", ly.runtime.method = "__await__", ly.runtime.result_contract = "types.CoroutineAwaitIterator"} {
    %one = arith.constant 1 : i64
    %layout_await_iter = arith.constant 10 : i64
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %consumed_slot = arith.constant 2 : index
    %zero = arith.constant 0 : i64

    %iterator = memref.alloc() {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<3xi64>
    memref.store %one, %iterator[%refcount_slot] : memref<3xi64>
    memref.store %layout_await_iter, %iterator[%layout_slot] : memref<3xi64>
    memref.store %zero, %iterator[%consumed_slot] : memref<3xi64>

    %c0 = arith.constant 0 : index
    %header = memref.subview %storage[%c0] [2] [1] : memref<5xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%header) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %iterator, %storage : memref<3xi64>, memref<5xi64>
  }

  func.func @LyCoroutineAwaitIterator_Iter(%iterator: memref<3xi64> {ly.ownership.object_header}, %coroutine: memref<5xi64> {ly.ownership.object_header}) -> (memref<3xi64>, memref<5xi64>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "types.CoroutineAwaitIterator", ly.runtime.method = "__iter__", ly.runtime.result_contract = "types.CoroutineAwaitIterator"} {
    %c0 = arith.constant 0 : index
    %iterator_header = memref.subview %iterator[%c0] [2] [1] : memref<3xi64> to memref<2xi64, strided<[1], offset: ?>>
    %coroutine_header = memref.subview %coroutine[%c0] [2] [1] : memref<5xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%iterator_header) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.call @Ly_IncRef(%coroutine_header) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %iterator, %coroutine : memref<3xi64>, memref<5xi64>
  }

  func.func @LyCoroutine_DecRef(%storage: memref<5xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "types.CoroutineType", ly.runtime.deallocator} {
    %c0 = arith.constant 0 : index
    %header = memref.subview %storage[%c0] [2] [1] : memref<5xi64> to memref<2xi64, strided<[1], offset: ?>>
    %became_zero = func.call @LyObject_DecRefHeader(%header) : (memref<2xi64, strided<[1], offset: ?>>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    memref.dealloc %storage {ly.ownership.object_dealloc_part = "storage"} : memref<5xi64>
    cf.br ^done

  ^done:
    func.return
  }

  func.func @LyCoroutineAwaitIterator_DecRef(%iterator: memref<3xi64> {ly.ownership.object_header}, %coroutine: memref<5xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "types.CoroutineAwaitIterator", ly.runtime.deallocator} {
    %c0 = arith.constant 0 : index
    %header = memref.subview %iterator[%c0] [2] [1] : memref<3xi64> to memref<2xi64, strided<[1], offset: ?>>
    %became_zero = func.call @LyObject_DecRefHeader(%header) : (memref<2xi64, strided<[1], offset: ?>>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    func.call @LyCoroutine_DecRef(%coroutine) : (memref<5xi64>) -> ()
    memref.dealloc %iterator {ly.ownership.object_dealloc_part = "iterator"} : memref<3xi64>
    cf.br ^done

  ^done:
    func.return
  }
}
