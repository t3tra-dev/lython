// Contract manifest AND module-level runtime for stdlib `asyncio` public
// classes, re-exports, and free functions.

module attributes {
  ly.runtime.contracts = ["asyncio.AbstractEventLoop", "_asyncio.Future", "_asyncio.FutureIter", "_asyncio.Task", "_asyncio.TaskIter"],
  ly.typing.module = "asyncio",
  ly.typing.class_exports = [
    "asyncio.AbstractEventLoop=asyncio.AbstractEventLoop",
    "asyncio.CancelledError=asyncio.CancelledError",
    "asyncio.Handle=asyncio.Handle",
    "asyncio.TimerHandle=asyncio.TimerHandle",
    "asyncio.Future=_asyncio.Future",
    "asyncio.Task=_asyncio.Task",
    "asyncio.events.AbstractEventLoop=asyncio.AbstractEventLoop",
    "asyncio.exceptions.CancelledError=asyncio.CancelledError"
  ],
  // Manifest Callable contracts for asyncio free functions, replacing the C++
  // makeAsyncioSleepCallable / makeAsyncioGetEventLoopCallable factories so
  // imported asyncio callables are typed from the manifest.
  // Callable exports drive import binding (TypeSystem reads these tables;
  // no C++ import-table entries needed) and pair with the contracts below.
  ly.typing.callable_exports = [
    "asyncio.sleep",
    "asyncio.get_event_loop",
    "asyncio.run"
  ],
  ly.typing.function_names = ["asyncio.sleep", "asyncio.get_event_loop", "asyncio.run"],
  ly.typing.function_contracts = [
    !py.callable<[!py.contract<"typing.Any">, !py.contract<"typing.Any">], arg_names = ["delay", "result"], arg_defaults = [false, true], returns = [!py.contract<"types.CoroutineType", [!py.contract<"typing.Any">, !py.contract<"typing.Any">, !py.contract<"typing.Any">]>]>,
    !py.callable<[], returns = [!py.contract<"asyncio.AbstractEventLoop">]>,
    !py.callable<[!py.contract<"types.CoroutineType", [!py.contract<"typing.Any">, !py.contract<"typing.Any">, !py.contract<"typing.Any">]>], arg_names = ["main"], arg_defaults = [false], kwonly = [!py.union<!py.contract<"builtins.bool">, !py.literal<None>>], kw_names = ["debug"], kw_defaults = [true], returns = [!py.contract<"typing.Any">]>
  ]
} {
  // Module-level runtime: asyncio's free functions live with the module
  // manifest.
  memref.global "private" @__ly_asyncio_default_loop : memref<8xi64> = dense<[9223372036854775807, 13, 0, 0, 0, 0, 0, 0]>

  func.func private @LyAsyncio_Sleep_Builtin() -> memref<5xi64> attributes {ly.runtime.builtin = "asyncio.sleep", ly.runtime.builtin_lowering = "asyncio_sleep", ly.runtime.contract = "types.CoroutineType", ly.runtime.primitive = "builtin_sleep", ly.runtime.result_contract = "types.CoroutineType"}

  func.func @LyAsyncio_GetEventLoop() -> memref<8xi64> attributes {ly.runtime.builtin = "asyncio.get_event_loop", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "asyncio.AbstractEventLoop", ly.runtime.primitive = "get_event_loop", ly.runtime.result_contract = "asyncio.AbstractEventLoop"} {
    %running_slot = arith.constant 2 : index
    %ready_tail_slot = arith.constant 5 : index
    %loop = memref.get_global @__ly_asyncio_default_loop : memref<8xi64>
    memref.generic_atomic_rmw %loop[%running_slot] : memref<8xi64> {
    ^bb0(%current : i64):
      memref.atomic_yield %current : i64
    } {ly.atomic.ordering = "acq_rel", ly.atomic.role = "asyncio.loop.running.publish"}
    memref.generic_atomic_rmw %loop[%ready_tail_slot] : memref<8xi64> {
    ^bb0(%current : i64):
      memref.atomic_yield %current : i64
    } {ly.atomic.ordering = "acq_rel", ly.atomic.role = "asyncio.ready.tail.publish"}
    func.return %loop : memref<8xi64>
  }

  py.class @CancelledError attributes {base_names = ["BaseException"]} {}

  py.class @Handle attributes {
    base_names = ["object"],
    method_names = ["__init__", "cancel", "_run", "cancelled"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"asyncio.Handle">, !py.callable<[], vararg = !py.unpack<!py.typevartuple<"Ts">>, returns = [!py.contract<"builtins.object">]>, !py.protocol<"Sequence", [!py.contract<"typing.Any">]>, !py.contract<"asyncio.AbstractEventLoop">], kwonly = [!py.union<!py.contract<"contextvars.Context">, !py.literal<None>>], kw_names = ["context"], kw_defaults = [true] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"asyncio.Handle">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"asyncio.Handle">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"asyncio.Handle">] -> [!py.contract<"builtins.bool">]>
    ],
    method_kinds = ["instance", "instance", "instance", "instance"]
  } {}

  py.class @TimerHandle attributes {
    base_names = ["Handle"],
    method_names = ["__init__", "when"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"asyncio.TimerHandle">, !py.contract<"builtins.float">, !py.callable<[], vararg = !py.unpack<!py.typevartuple<"Ts">>, returns = [!py.contract<"builtins.object">]>, !py.protocol<"Sequence", [!py.contract<"typing.Any">]>, !py.contract<"asyncio.AbstractEventLoop">], kwonly = [!py.union<!py.contract<"contextvars.Context">, !py.literal<None>>], kw_names = ["context"], kw_defaults = [true] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"asyncio.TimerHandle">] -> [!py.contract<"builtins.float">]>
    ],
    method_kinds = ["instance", "instance"]
  } {}

  py.class @AbstractEventLoop attributes {
    base_names = ["object"], ly.typing.abstract,
    method_names = ["is_running", "stop", "call_soon", "call_later",
                    "call_at"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"asyncio.AbstractEventLoop">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"asyncio.AbstractEventLoop">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"asyncio.AbstractEventLoop">, !py.callable<[], vararg = !py.unpack<!py.typevartuple<"Ts">>, returns = [!py.contract<"builtins.object">]>], vararg = !py.unpack<!py.typevartuple<"Ts">>, kwonly = [!py.union<!py.contract<"contextvars.Context">, !py.literal<None>>], kw_names = ["context"], kw_defaults = [true], vararg_name = "args" -> [!py.contract<"asyncio.Handle">]>,
      !py.protocol<"Callable", [!py.contract<"asyncio.AbstractEventLoop">, !py.contract<"builtins.float">, !py.callable<[], vararg = !py.unpack<!py.typevartuple<"Ts">>, returns = [!py.contract<"builtins.object">]>], vararg = !py.unpack<!py.typevartuple<"Ts">>, kwonly = [!py.union<!py.contract<"contextvars.Context">, !py.literal<None>>], kw_names = ["context"], kw_defaults = [true], vararg_name = "args" -> [!py.contract<"asyncio.TimerHandle">]>,
      !py.protocol<"Callable", [!py.contract<"asyncio.AbstractEventLoop">, !py.contract<"builtins.float">, !py.callable<[], vararg = !py.unpack<!py.typevartuple<"Ts">>, returns = [!py.contract<"builtins.object">]>], vararg = !py.unpack<!py.typevartuple<"Ts">>, kwonly = [!py.union<!py.contract<"contextvars.Context">, !py.literal<None>>], kw_names = ["context"], kw_defaults = [true], vararg_name = "args" -> [!py.contract<"asyncio.TimerHandle">]>
    ],
    method_kinds = ["instance", "instance", "instance", "instance",
                    "instance"]
  } {}

  // ===========================================================
  // Runtime implementations (event loop, futures, tasks).
  // ===========================================================

  // ===== impls: asyncio =====
  func.func private @LyCoroutine_ResumeBegin(%storage: memref<5xi64> {ly.ownership.object_header}) -> i1
  func.func private @LyCoroutine_ResumeComplete(%storage: memref<5xi64> {ly.ownership.object_header})
  func.func private @LyCoroutine_DecRef(%storage: memref<5xi64> {ly.ownership.object_header})
  func.func private @LyLong_FromI64(%value: i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0]}
  func.func private @LyBaseException_New(%class_id: i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
  func.func private @LyBaseException_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
  func.func private @LyEH_ThrowException(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>)
  func.func private @LyUnicode_FromBytes(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0]}

  memref.global "private" constant @__ly_task_msg_set_result_unsupported : memref<42xi8> = dense<[84, 97, 115, 107, 32, 100, 111, 101, 115, 32, 110, 111, 116, 32, 115, 117, 112, 112, 111, 114, 116, 32, 115, 101, 116, 95, 114, 101, 115, 117, 108, 116, 32, 111, 112, 101, 114, 97, 116, 105, 111, 110]>
  memref.global "private" constant @__ly_task_msg_set_exception_unsupported : memref<45xi8> = dense<[84, 97, 115, 107, 32, 100, 111, 101, 115, 32, 110, 111, 116, 32, 115, 117, 112, 112, 111, 114, 116, 32, 115, 101, 116, 95, 101, 120, 99, 101, 112, 116, 105, 111, 110, 32, 111, 112, 101, 114, 97, 116, 105, 111, 110]>
  memref.global "private" constant @__ly_asyncio_msg_invalid_state : memref<13xi8> = dense<[105, 110, 118, 97, 108, 105, 100, 32, 115, 116, 97, 116, 101]>

  func.func private @LyEventLoop_Shape() -> memref<8xi64> attributes {ly.runtime.contract = "asyncio.AbstractEventLoop", ly.runtime.shape}

  func.func private @LyFuture_Shape() -> memref<10xi64> attributes {ly.runtime.contract = "_asyncio.Future", ly.runtime.shape}



  func.func private @LyFutureIter_Shape() -> (memref<3xi64>, memref<10xi64>) attributes {ly.runtime.contract = "_asyncio.FutureIter", ly.runtime.shape}

  func.func private @LyTask_Shape() -> (memref<12xi64>, memref<5xi64>) attributes {ly.runtime.contract = "_asyncio.Task", ly.runtime.shape}

  func.func private @LyTaskIter_Shape() -> (memref<3xi64>, memref<12xi64>, memref<5xi64>) attributes {ly.runtime.contract = "_asyncio.TaskIter", ly.runtime.shape}

  func.func private @__ly_asyncio_retain_storage(%storage: memref<?xi64>) {
    %slot = arith.constant 0 : index
    %zero = arith.constant 0 : i64
    %immortal = arith.constant 9223372036854775807 : i64

    %observed = memref.load %storage[%slot] {ly.atomic.ordering = "acquire", ly.atomic.role = "asyncio.object.refcount.load"} : memref<?xi64>
    %pre_immortal = arith.cmpi eq, %observed, %immortal : i64
    cf.cond_br %pre_immortal, ^done, ^mutate

  ^mutate:
    %previous = memref.generic_atomic_rmw %storage[%slot] : memref<?xi64> {
    ^bb0(%current : i64):
      %body_zero = arith.constant 0 : i64
      %body_one = arith.constant 1 : i64
      %body_immortal = arith.constant 9223372036854775807 : i64
      %body_positive = arith.cmpi sgt, %current, %body_zero : i64
      %body_immortal_check = arith.cmpi eq, %current, %body_immortal : i64
      %body_incremented = arith.addi %current, %body_one : i64
      %body_positive_next = arith.select %body_positive, %body_incremented, %current : i1, i64
      %body_next = arith.select %body_immortal_check, %current, %body_positive_next : i1, i64
      memref.atomic_yield %body_next : i64
    } {ly.atomic.ordering = "acq_rel", ly.atomic.retain_premise = "entry-borrowed", ly.atomic.role = "asyncio.object.refcount.retain"}
    %is_immortal = arith.cmpi eq, %previous, %immortal : i64
    cf.cond_br %is_immortal, ^done, ^check_positive

  ^check_positive:
    %positive = arith.cmpi sgt, %previous, %zero : i64
    cf.assert %positive, "asyncio retain observed non-positive refcount"
    cf.br ^done

  ^done:
    func.return
  }

  func.func private @__ly_asyncio_release_storage_to_zero(%storage: memref<?xi64>) -> i1 {
    %slot = arith.constant 0 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %immortal = arith.constant 9223372036854775807 : i64

    %observed = memref.load %storage[%slot] {ly.atomic.ordering = "acquire", ly.atomic.role = "asyncio.object.refcount.load"} : memref<?xi64>
    %pre_immortal = arith.cmpi eq, %observed, %immortal : i64
    cf.cond_br %pre_immortal, ^done, ^mutate

  ^mutate:
    %previous = memref.generic_atomic_rmw %storage[%slot] : memref<?xi64> {
    ^bb0(%current : i64):
      %body_zero = arith.constant 0 : i64
      %body_one = arith.constant 1 : i64
      %body_immortal = arith.constant 9223372036854775807 : i64
      %body_positive = arith.cmpi sgt, %current, %body_zero : i64
      %body_immortal_check = arith.cmpi eq, %current, %body_immortal : i64
      %body_decremented = arith.subi %current, %body_one : i64
      %body_positive_next = arith.select %body_positive, %body_decremented, %current : i1, i64
      %body_next = arith.select %body_immortal_check, %current, %body_positive_next : i1, i64
      memref.atomic_yield %body_next : i64
    } {ly.atomic.ordering = "acq_rel", ly.atomic.role = "asyncio.object.refcount.release"}
    %is_immortal = arith.cmpi eq, %previous, %immortal : i64
    cf.cond_br %is_immortal, ^done, ^check_positive

  ^check_positive:
    %positive = arith.cmpi sgt, %previous, %zero : i64
    cf.assert %positive, "asyncio release observed non-positive refcount"
    %became_zero = arith.cmpi eq, %previous, %one : i64
    func.return %became_zero : i1

  ^done:
    %false = arith.constant false
    func.return %false : i1
  }

  func.func private @__ly_task_raise_runtime_error(%message: memref<?xi8>, %length: i64) {
    %class_id_runtime_error = arith.constant 51 : i64
    %start = arith.constant 0 : index
    %exception:3 = func.call @LyBaseException_New(%class_id_runtime_error) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    %message_header, %message_bytes = func.call @LyUnicode_FromBytes(%message, %start, %length) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %initialized:3 = func.call @LyBaseException_Init(%exception#0, %exception#1, %exception#2, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.call @LyEH_ThrowException(%initialized#0, %initialized#1, %initialized#2) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @__ly_task_raise_set_result_unsupported() {
    %length = arith.constant 42 : i64
    %message_static = memref.get_global @__ly_task_msg_set_result_unsupported : memref<42xi8>
    %message = memref.cast %message_static : memref<42xi8> to memref<?xi8>
    func.call @__ly_task_raise_runtime_error(%message, %length) : (memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @__ly_task_raise_set_exception_unsupported() {
    %length = arith.constant 45 : i64
    %message_static = memref.get_global @__ly_task_msg_set_exception_unsupported : memref<45xi8>
    %message = memref.cast %message_static : memref<45xi8> to memref<?xi8>
    func.call @__ly_task_raise_runtime_error(%message, %length) : (memref<?xi8>, i64) -> ()
    func.return
  }

  func.func private @__ly_asyncio_raise_invalid_state() {
    %length = arith.constant 13 : i64
    %message_static = memref.get_global @__ly_asyncio_msg_invalid_state : memref<13xi8>
    %message = memref.cast %message_static : memref<13xi8> to memref<?xi8>
    func.call @__ly_task_raise_runtime_error(%message, %length) : (memref<?xi8>, i64) -> ()
    func.return
  }

  func.func @LyEventLoop_New() -> memref<8xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 13 : i64, ly.runtime.contract = "asyncio.AbstractEventLoop", ly.runtime.initializer = "__new__"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %layout = arith.constant 13 : i64
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %running_slot = arith.constant 2 : index
    %stopping_slot = arith.constant 3 : index
    %ready_head_slot = arith.constant 4 : index
    %ready_tail_slot = arith.constant 5 : index
    %timer_count_slot = arith.constant 6 : index
    %io_watch_count_slot = arith.constant 7 : index

    %loop = memref.alloc() {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<8xi64>
    memref.store %one, %loop[%refcount_slot] : memref<8xi64>
    memref.store %layout, %loop[%layout_slot] : memref<8xi64>
    memref.store %zero, %loop[%running_slot] {ly.atomic.ordering = "release", ly.atomic.role = "asyncio.loop.running.publish"} : memref<8xi64>
    memref.store %zero, %loop[%stopping_slot] : memref<8xi64>
    memref.store %zero, %loop[%ready_head_slot] : memref<8xi64>
    memref.store %zero, %loop[%ready_tail_slot] {ly.atomic.ordering = "release", ly.atomic.role = "asyncio.ready.tail.publish"} : memref<8xi64>
    memref.store %zero, %loop[%timer_count_slot] : memref<8xi64>
    memref.store %zero, %loop[%io_watch_count_slot] : memref<8xi64>
    func.return %loop : memref<8xi64>
  }

  func.func @LyEventLoop_EnqueueReady(%loop: memref<8xi64> {ly.ownership.object_header}) -> i64 attributes {ly.runtime.contract = "asyncio.AbstractEventLoop", ly.runtime.primitive = "callback.enqueue_ready"} {
    %tail_slot = arith.constant 5 : index
    %one = arith.constant 1 : i64
    %previous = memref.generic_atomic_rmw %loop[%tail_slot] : memref<8xi64> {
    ^bb0(%current : i64):
      %next = arith.addi %current, %one : i64
      memref.atomic_yield %next : i64
    } {ly.atomic.ordering = "acq_rel", ly.atomic.role = "asyncio.ready.enqueue"}
    %handle_id = arith.addi %previous, %one : i64
    func.return %handle_id : i64
  }

  func.func @LyEventLoop_PopReady(%loop: memref<8xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "asyncio.AbstractEventLoop", ly.runtime.primitive = "callback.pop_ready"} {
    %head_slot = arith.constant 4 : index
    %tail_slot = arith.constant 5 : index
    %tail = memref.load %loop[%tail_slot] {ly.atomic.ordering = "acquire", ly.atomic.role = "asyncio.ready.tail.load"} : memref<8xi64>
    %previous = memref.generic_atomic_rmw %loop[%head_slot] : memref<8xi64> {
    ^bb0(%current : i64):
      %has_ready = arith.cmpi ult, %current, %tail : i64
      %one = arith.constant 1 : i64
      %advanced = arith.addi %current, %one : i64
      %next = arith.select %has_ready, %advanced, %current : i1, i64
      memref.atomic_yield %next : i64
    } {ly.atomic.ordering = "acq_rel", ly.atomic.role = "asyncio.ready.pop"}
    %had_ready = arith.cmpi ult, %previous, %tail : i64
    func.return %had_ready : i1
  }

  func.func @LyEventLoop_RecordTimer(%loop: memref<8xi64> {ly.ownership.object_header}) -> i64 attributes {ly.runtime.contract = "asyncio.AbstractEventLoop", ly.runtime.primitive = "timer.record"} {
    %timer_count_slot = arith.constant 6 : index
    %one = arith.constant 1 : i64
    %previous = memref.generic_atomic_rmw %loop[%timer_count_slot] : memref<8xi64> {
    ^bb0(%current : i64):
      %next = arith.addi %current, %one : i64
      memref.atomic_yield %next : i64
    } {ly.atomic.ordering = "acq_rel", ly.atomic.role = "asyncio.timer.record"}
    %timer_id = arith.addi %previous, %one : i64
    func.return %timer_id : i64
  }

  func.func @LyEventLoop_DispatchDueTimer(%loop: memref<8xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "asyncio.AbstractEventLoop", ly.runtime.primitive = "timer.dispatch_due"} {
    %timer_count_slot = arith.constant 6 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %previous = memref.generic_atomic_rmw %loop[%timer_count_slot] : memref<8xi64> {
    ^bb0(%current : i64):
      %has_timer = arith.cmpi ugt, %current, %zero : i64
      %decremented = arith.subi %current, %one : i64
      %next = arith.select %has_timer, %decremented, %current : i1, i64
      memref.atomic_yield %next : i64
    } {ly.atomic.ordering = "acq_rel", ly.atomic.role = "asyncio.timer.dispatch_due"}
    %had_timer = arith.cmpi ugt, %previous, %zero : i64
    %ready = scf.if %had_timer -> (i1) {
      %handle_id = func.call @LyEventLoop_EnqueueReady(%loop) : (memref<8xi64>) -> i64
      %popped = func.call @LyEventLoop_PopReady(%loop) : (memref<8xi64>) -> i1
      scf.yield %popped : i1
    } else {
      %false = arith.constant false
      scf.yield %false : i1
    }
    func.return %ready : i1
  }

  func.func @LyEventLoop_RequestStop(%loop: memref<8xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "asyncio.AbstractEventLoop", ly.runtime.primitive = "stop.request"} {
    %stopping_slot = arith.constant 3 : index
    %one = arith.constant 1 : i64
    memref.store %one, %loop[%stopping_slot] {ly.atomic.ordering = "release", ly.atomic.role = "asyncio.loop.stop"} : memref<8xi64>
    func.return
  }

  func.func @LyEventLoop_IsRunning(%loop: memref<8xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "asyncio.AbstractEventLoop", ly.runtime.method = "is_running", ly.runtime.result_contract = "builtins.bool"} {
    %running_slot = arith.constant 2 : index
    %zero = arith.constant 0 : i64
    %running_value = memref.load %loop[%running_slot] {ly.atomic.ordering = "acquire", ly.atomic.role = "asyncio.loop.running.load"} : memref<8xi64>
    %running = arith.cmpi ne, %running_value, %zero : i64
    func.return %running : i1
  }

  func.func @LyEventLoop_Stop(%loop: memref<8xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "asyncio.AbstractEventLoop", ly.runtime.method = "stop", ly.runtime.result_contract = "types.NoneType"} {
    func.call @LyEventLoop_RequestStop(%loop) : (memref<8xi64>) -> ()
    func.return
  }

  func.func @LyEventLoop_DecRef(%loop: memref<8xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "asyncio.AbstractEventLoop", ly.runtime.deallocator} {
    %storage = memref.cast %loop : memref<8xi64> to memref<?xi64>
    %became_zero = func.call @__ly_asyncio_release_storage_to_zero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    memref.dealloc %loop : memref<8xi64>
    cf.br ^done

  ^done:
    func.return
  }

  func.func @LyFuture_New() -> memref<10xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 14 : i64, ly.runtime.contract = "_asyncio.Future", ly.runtime.initializer = "__new__"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %layout = arith.constant 14 : i64
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %state_slot = arith.constant 2 : index
    %cancel_requests_slot = arith.constant 3 : index
    %ready_callbacks_slot = arith.constant 4 : index
    %loop_id_slot = arith.constant 5 : index
    %waiter_count_slot = arith.constant 6 : index
    %result_token_slot = arith.constant 7 : index
    %exception_token_slot = arith.constant 8 : index
    %flags_slot = arith.constant 9 : index

    %future = memref.alloc() {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<10xi64>
    memref.store %one, %future[%refcount_slot] : memref<10xi64>
    memref.store %layout, %future[%layout_slot] : memref<10xi64>
    // State 0 = pending, 1 = finished, 2 = cancelled, 3 = publishing.
    memref.store %zero, %future[%state_slot] {ly.atomic.ordering = "release", ly.atomic.role = "asyncio.future.state.publish"} : memref<10xi64>
    memref.store %zero, %future[%cancel_requests_slot] : memref<10xi64>
    memref.store %zero, %future[%ready_callbacks_slot] : memref<10xi64>
    memref.store %zero, %future[%loop_id_slot] : memref<10xi64>
    memref.store %zero, %future[%waiter_count_slot] : memref<10xi64>
    memref.store %zero, %future[%result_token_slot] : memref<10xi64>
    memref.store %zero, %future[%exception_token_slot] : memref<10xi64>
    memref.store %zero, %future[%flags_slot] : memref<10xi64>
    func.return %future : memref<10xi64>
  }

  func.func @LyFuture_Init(%future: memref<10xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "_asyncio.Future", ly.runtime.method = "__init__", ly.runtime.result_contract = "types.NoneType"} {
    func.return
  }

  func.func @LyFuture_Done(%future: memref<10xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "_asyncio.Future", ly.runtime.method = "done", ly.runtime.result_contract = "builtins.bool"} {
    %state_slot = arith.constant 2 : index
    %finished = arith.constant 1 : i64
    %cancelled_state = arith.constant 2 : i64
    %state = memref.load %future[%state_slot] {ly.atomic.ordering = "acquire", ly.atomic.role = "asyncio.future.state.load"} : memref<10xi64>
    %is_finished = arith.cmpi eq, %state, %finished : i64
    %is_cancelled = arith.cmpi eq, %state, %cancelled_state : i64
    %done = arith.ori %is_finished, %is_cancelled : i1
    func.return %done : i1
  }

  func.func @LyFuture_Cancelled(%future: memref<10xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "_asyncio.Future", ly.runtime.method = "cancelled", ly.runtime.result_contract = "builtins.bool"} {
    %state_slot = arith.constant 2 : index
    %cancelled_state = arith.constant 2 : i64
    %state = memref.load %future[%state_slot] {ly.atomic.ordering = "acquire", ly.atomic.role = "asyncio.future.state.load"} : memref<10xi64>
    %cancelled = arith.cmpi eq, %state, %cancelled_state : i64
    func.return %cancelled : i1
  }

  func.func @LyFuture_RequestCancel(%future: memref<10xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "_asyncio.Future", ly.runtime.primitive = "cancel.request"} {
    %state_slot = arith.constant 2 : index
    %cancel_requests_slot = arith.constant 3 : index
    %pending = arith.constant 0 : i64
    %cancelled_state = arith.constant 2 : i64
    %previous = memref.generic_atomic_rmw %future[%state_slot] : memref<10xi64> {
    ^bb0(%current : i64):
      %is_pending = arith.cmpi eq, %current, %pending : i64
      %next = arith.select %is_pending, %cancelled_state, %current : i1, i64
      memref.atomic_yield %next : i64
    } {ly.atomic.ordering = "acq_rel", ly.atomic.role = "asyncio.future.cancel"}
    %was_pending = arith.cmpi eq, %previous, %pending : i64
    cf.cond_br %was_pending, ^record, ^done

  ^record:
    %one = arith.constant 1 : i64
    memref.store %one, %future[%cancel_requests_slot] {ly.atomic.ordering = "release", ly.atomic.role = "asyncio.future.cancel.requests"} : memref<10xi64>
    cf.br ^done

  ^done:
    func.return %was_pending : i1
  }

  func.func @LyFuture_Cancel(%future: memref<10xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "_asyncio.Future", ly.runtime.method = "cancel", ly.runtime.result_contract = "builtins.bool"} {
    %cancelled = func.call @LyFuture_RequestCancel(%future) : (memref<10xi64>) -> i1
    func.return %cancelled : i1
  }

  func.func private @LyFuture_ReserveFinish(%future: memref<10xi64> {ly.ownership.object_header}) -> i1 {
    %state_slot = arith.constant 2 : index
    %pending = arith.constant 0 : i64
    %publishing = arith.constant 3 : i64
    %previous = memref.generic_atomic_rmw %future[%state_slot] : memref<10xi64> {
    ^bb0(%current : i64):
      %is_pending = arith.cmpi eq, %current, %pending : i64
      %next = arith.select %is_pending, %publishing, %current : i1, i64
      memref.atomic_yield %next : i64
    } {ly.atomic.ordering = "acq_rel", ly.atomic.role = "asyncio.future.finish.reserve"}
    %was_pending = arith.cmpi eq, %previous, %pending : i64
    func.return %was_pending : i1
  }

  func.func private @LyFuture_CommitFinished(%future: memref<10xi64> {ly.ownership.object_header}) {
    %state_slot = arith.constant 2 : index
    %publishing = arith.constant 3 : i64
    %finished = arith.constant 1 : i64
    %previous = memref.generic_atomic_rmw %future[%state_slot] : memref<10xi64> {
    ^bb0(%current : i64):
      %is_publishing = arith.cmpi eq, %current, %publishing : i64
      %next = arith.select %is_publishing, %finished, %current : i1, i64
      memref.atomic_yield %next : i64
    } {ly.atomic.ordering = "acq_rel", ly.atomic.role = "asyncio.future.finish"}
    %was_publishing = arith.cmpi eq, %previous, %publishing : i64
    cf.assert %was_publishing, "LyFuture_CommitFinished called without a reserved future payload"
    func.return
  }

  func.func @LyFuture_MarkFinished(%future: memref<10xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "_asyncio.Future", ly.runtime.primitive = "finish.request"} {
    %state_slot = arith.constant 2 : index
    %pending = arith.constant 0 : i64
    %finished = arith.constant 1 : i64
    %previous = memref.generic_atomic_rmw %future[%state_slot] : memref<10xi64> {
    ^bb0(%current : i64):
      %is_pending = arith.cmpi eq, %current, %pending : i64
      %next = arith.select %is_pending, %finished, %current : i1, i64
      memref.atomic_yield %next : i64
    } {ly.atomic.ordering = "acq_rel", ly.atomic.role = "asyncio.future.finish"}
    %was_pending = arith.cmpi eq, %previous, %pending : i64
    func.return %was_pending : i1
  }

  func.func @LyFuture_SetResult(%future: memref<10xi64> {ly.ownership.object_header}, %result: memref<?xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "_asyncio.Future", ly.runtime.method = "set_result", ly.runtime.result_contract = "types.NoneType"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %result_token_slot = arith.constant 7 : index
    %exception_token_slot = arith.constant 8 : index
    %finished = func.call @LyFuture_ReserveFinish(%future) : (memref<10xi64>) -> i1
    cf.cond_br %finished, ^record, ^invalid

  ^record:
    memref.store %one, %future[%result_token_slot] {ly.atomic.ordering = "release", ly.atomic.role = "asyncio.future.result.token"} : memref<10xi64>
    memref.store %zero, %future[%exception_token_slot] {ly.atomic.ordering = "release", ly.atomic.role = "asyncio.future.exception.token.clear"} : memref<10xi64>
    func.call @LyFuture_CommitFinished(%future) : (memref<10xi64>) -> ()
    cf.br ^done

  ^invalid:
    func.call @__ly_asyncio_raise_invalid_state() : () -> ()
    cf.br ^done

  ^done:
    func.return
  }

  func.func @LyFuture_SetException(%future: memref<10xi64> {ly.ownership.object_header}, %exception: memref<?xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "_asyncio.Future", ly.runtime.method = "set_exception", ly.runtime.result_contract = "types.NoneType"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %result_token_slot = arith.constant 7 : index
    %exception_token_slot = arith.constant 8 : index
    %finished = func.call @LyFuture_ReserveFinish(%future) : (memref<10xi64>) -> i1
    cf.cond_br %finished, ^record, ^invalid

  ^record:
    memref.store %zero, %future[%result_token_slot] {ly.atomic.ordering = "release", ly.atomic.role = "asyncio.future.result.token.clear"} : memref<10xi64>
    memref.store %one, %future[%exception_token_slot] {ly.atomic.ordering = "release", ly.atomic.role = "asyncio.future.exception.token"} : memref<10xi64>
    func.call @LyFuture_CommitFinished(%future) : (memref<10xi64>) -> ()
    cf.br ^done

  ^invalid:
    func.call @__ly_asyncio_raise_invalid_state() : () -> ()
    cf.br ^done

  ^done:
    func.return
  }

  func.func @LyFuture_RecordCallback(%future: memref<10xi64> {ly.ownership.object_header}) -> i64 attributes {ly.runtime.contract = "_asyncio.Future", ly.runtime.primitive = "callback.record"} {
    %ready_callbacks_slot = arith.constant 4 : index
    %one = arith.constant 1 : i64
    %previous = memref.generic_atomic_rmw %future[%ready_callbacks_slot] : memref<10xi64> {
    ^bb0(%current : i64):
      %next = arith.addi %current, %one : i64
      memref.atomic_yield %next : i64
    } {ly.atomic.ordering = "acq_rel", ly.atomic.role = "asyncio.future.callback.record"}
    %count = arith.addi %previous, %one : i64
    func.return %count : i64
  }

  func.func private @LyFutureIter_New(%future: memref<10xi64> {ly.ownership.object_header}) -> (memref<3xi64>, memref<10xi64>) attributes {ly.ownership.owned_results = [0]} {
    %one = arith.constant 1 : i64
    %zero = arith.constant 0 : i64
    %layout_future_iter = arith.constant 16 : i64
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %consumed_slot = arith.constant 2 : index

    %iterator = memref.alloc() {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<3xi64>
    memref.store %one, %iterator[%refcount_slot] : memref<3xi64>
    memref.store %layout_future_iter, %iterator[%layout_slot] : memref<3xi64>
    memref.store %zero, %iterator[%consumed_slot] : memref<3xi64>

    %future_storage = memref.cast %future : memref<10xi64> to memref<?xi64>
    func.call @__ly_asyncio_retain_storage(%future_storage) : (memref<?xi64>) -> ()
    func.return %iterator, %future : memref<3xi64>, memref<10xi64>
  }

  func.func @LyFuture_Await(%future: memref<10xi64> {ly.ownership.object_header}) -> (memref<3xi64>, memref<10xi64>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "_asyncio.Future", ly.runtime.method = "__await__", ly.runtime.result_contract = "_asyncio.FutureIter"} {
    %iterator, %kept_future = func.call @LyFutureIter_New(%future) : (memref<10xi64>) -> (memref<3xi64>, memref<10xi64>)
    func.return %iterator, %kept_future : memref<3xi64>, memref<10xi64>
  }

  func.func @LyFuture_Iter(%future: memref<10xi64> {ly.ownership.object_header}) -> (memref<3xi64>, memref<10xi64>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "_asyncio.Future", ly.runtime.method = "__iter__", ly.runtime.result_contract = "_asyncio.FutureIter"} {
    %iterator, %kept_future = func.call @LyFutureIter_New(%future) : (memref<10xi64>) -> (memref<3xi64>, memref<10xi64>)
    func.return %iterator, %kept_future : memref<3xi64>, memref<10xi64>
  }

  func.func @LyFuture_AddDoneCallback(%future: memref<10xi64> {ly.ownership.object_header}, %callback: memref<?xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "_asyncio.Future", ly.runtime.method = "add_done_callback", ly.runtime.result_contract = "types.NoneType"} {
    %count = func.call @LyFuture_RecordCallback(%future) : (memref<10xi64>) -> i64
    func.return
  }

  func.func @LyFutureIter_Iter(%iterator: memref<3xi64> {ly.ownership.object_header}, %future: memref<10xi64> {ly.ownership.object_header}) -> (memref<3xi64>, memref<10xi64>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "_asyncio.FutureIter", ly.runtime.method = "__iter__", ly.runtime.result_contract = "_asyncio.FutureIter"} {
    %iterator_storage = memref.cast %iterator : memref<3xi64> to memref<?xi64>
    %future_storage = memref.cast %future : memref<10xi64> to memref<?xi64>
    func.call @__ly_asyncio_retain_storage(%iterator_storage) : (memref<?xi64>) -> ()
    func.call @__ly_asyncio_retain_storage(%future_storage) : (memref<?xi64>) -> ()
    func.return %iterator, %future : memref<3xi64>, memref<10xi64>
  }

  func.func @LyFuture_DecRef(%future: memref<10xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "_asyncio.Future", ly.runtime.deallocator} {
    %storage = memref.cast %future : memref<10xi64> to memref<?xi64>
    %became_zero = func.call @__ly_asyncio_release_storage_to_zero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    memref.dealloc %future : memref<10xi64>
    cf.br ^done

  ^done:
    func.return
  }

  func.func @LyFutureIter_DecRef(%iterator: memref<3xi64> {ly.ownership.object_header}, %future: memref<10xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "_asyncio.FutureIter", ly.runtime.deallocator} {
    %iterator_storage = memref.cast %iterator : memref<3xi64> to memref<?xi64>
    %became_zero = func.call @__ly_asyncio_release_storage_to_zero(%iterator_storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    func.call @LyFuture_DecRef(%future) : (memref<10xi64>) -> ()
    memref.dealloc %iterator : memref<3xi64>
    cf.br ^done

  ^done:
    func.return
  }

  func.func @LyTask_New(%coroutine: memref<5xi64> {ly.ownership.object_header}) -> (memref<12xi64>, memref<5xi64>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 15 : i64, ly.runtime.contract = "_asyncio.Task", ly.runtime.initializer = "__new__"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %layout = arith.constant 15 : i64
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %state_slot = arith.constant 2 : index
    %cancel_requests_slot = arith.constant 3 : index
    %ready_callbacks_slot = arith.constant 4 : index
    %loop_id_slot = arith.constant 5 : index
    %waiter_count_slot = arith.constant 6 : index
    %result_token_slot = arith.constant 7 : index
    %exception_token_slot = arith.constant 8 : index
    %flags_slot = arith.constant 9 : index
    %coroutine_target_slot = arith.constant 10 : index
    %step_epoch_slot = arith.constant 11 : index
    %source_target_slot = arith.constant 3 : index
    %coroutine_target_id = memref.load %coroutine[%source_target_slot] {ly.atomic.ordering = "acquire", ly.atomic.role = "asyncio.task.coroutine.target.load"} : memref<5xi64>

    %task = memref.alloc() {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<12xi64>
    memref.store %one, %task[%refcount_slot] : memref<12xi64>
    memref.store %layout, %task[%layout_slot] : memref<12xi64>
    // Task states: 0 = pending, 1 = running, 2 = finished, 3 = cancelled.
    memref.store %zero, %task[%state_slot] {ly.atomic.ordering = "release", ly.atomic.role = "asyncio.task.state.publish"} : memref<12xi64>
    memref.store %zero, %task[%cancel_requests_slot] {ly.atomic.ordering = "release", ly.atomic.role = "asyncio.task.cancel.requests.publish"} : memref<12xi64>
    memref.store %zero, %task[%ready_callbacks_slot] : memref<12xi64>
    memref.store %zero, %task[%loop_id_slot] : memref<12xi64>
    memref.store %zero, %task[%waiter_count_slot] : memref<12xi64>
    memref.store %zero, %task[%result_token_slot] : memref<12xi64>
    memref.store %zero, %task[%exception_token_slot] : memref<12xi64>
    memref.store %zero, %task[%flags_slot] : memref<12xi64>
    memref.store %coroutine_target_id, %task[%coroutine_target_slot] : memref<12xi64>
    memref.store %zero, %task[%step_epoch_slot] : memref<12xi64>

    %coroutine_storage = memref.cast %coroutine : memref<5xi64> to memref<?xi64>
    func.call @__ly_asyncio_retain_storage(%coroutine_storage) : (memref<?xi64>) -> ()
    func.return %task, %coroutine : memref<12xi64>, memref<5xi64>
  }

  func.func @LyTask_Init(%task: memref<12xi64> {ly.ownership.object_header}, %owned_coroutine: memref<5xi64> {ly.ownership.object_header}, %coroutine: memref<5xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "_asyncio.Task", ly.runtime.method = "__init__", ly.runtime.result_contract = "types.NoneType"} {
    func.return
  }

  func.func @LyTask_Done(%task: memref<12xi64> {ly.ownership.object_header}, %coroutine: memref<5xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "_asyncio.Task", ly.runtime.method = "done", ly.runtime.result_contract = "builtins.bool"} {
    %state_slot = arith.constant 2 : index
    %finished = arith.constant 2 : i64
    %cancelled_state = arith.constant 3 : i64
    %state = memref.load %task[%state_slot] {ly.atomic.ordering = "acquire", ly.atomic.role = "asyncio.task.state.load"} : memref<12xi64>
    %is_finished = arith.cmpi eq, %state, %finished : i64
    %is_cancelled = arith.cmpi eq, %state, %cancelled_state : i64
    %done = arith.ori %is_finished, %is_cancelled : i1
    func.return %done : i1
  }

  func.func @LyTask_Cancelled(%task: memref<12xi64> {ly.ownership.object_header}, %coroutine: memref<5xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "_asyncio.Task", ly.runtime.method = "cancelled", ly.runtime.result_contract = "builtins.bool"} {
    %state_slot = arith.constant 2 : index
    %cancelled_state = arith.constant 3 : i64
    %state = memref.load %task[%state_slot] {ly.atomic.ordering = "acquire", ly.atomic.role = "asyncio.task.state.load"} : memref<12xi64>
    %cancelled = arith.cmpi eq, %state, %cancelled_state : i64
    func.return %cancelled : i1
  }

  func.func @LyTask_RequestCancel(%task: memref<12xi64> {ly.ownership.object_header}, %coroutine: memref<5xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "_asyncio.Task", ly.runtime.primitive = "cancel.request"} {
    %state_slot = arith.constant 2 : index
    %cancel_requests_slot = arith.constant 3 : index
    %pending = arith.constant 0 : i64
    %running = arith.constant 1 : i64
    %cancelled_state = arith.constant 3 : i64
    %one = arith.constant 1 : i64
    %previous = memref.generic_atomic_rmw %task[%state_slot] : memref<12xi64> {
    ^bb0(%current : i64):
      %is_pending = arith.cmpi eq, %current, %pending : i64
      %is_running = arith.cmpi eq, %current, %running : i64
      %can_cancel = arith.ori %is_pending, %is_running : i1
      %next = arith.select %can_cancel, %cancelled_state, %current : i1, i64
      memref.atomic_yield %next : i64
    } {ly.atomic.ordering = "acq_rel", ly.atomic.role = "asyncio.task.cancel"}
    %was_pending = arith.cmpi eq, %previous, %pending : i64
    %was_running = arith.cmpi eq, %previous, %running : i64
    %did_cancel = arith.ori %was_pending, %was_running : i1
    cf.cond_br %did_cancel, ^record, ^done

  ^record:
    memref.generic_atomic_rmw %task[%cancel_requests_slot] : memref<12xi64> {
    ^bb0(%current : i64):
      %next = arith.addi %current, %one : i64
      memref.atomic_yield %next : i64
    } {ly.atomic.ordering = "acq_rel", ly.atomic.role = "asyncio.task.cancel.requests"}
    cf.br ^done

  ^done:
    func.return %did_cancel : i1
  }

  func.func @LyTask_Cancel(%task: memref<12xi64> {ly.ownership.object_header}, %coroutine: memref<5xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "_asyncio.Task", ly.runtime.method = "cancel", ly.runtime.result_contract = "builtins.bool"} {
    %cancelled = func.call @LyTask_RequestCancel(%task, %coroutine) : (memref<12xi64>, memref<5xi64>) -> i1
    func.return %cancelled : i1
  }

  func.func @LyTask_Cancelling(%task: memref<12xi64> {ly.ownership.object_header}, %coroutine: memref<5xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "_asyncio.Task", ly.runtime.method = "cancelling", ly.runtime.result_contract = "builtins.int"} {
    %cancel_requests_slot = arith.constant 3 : index
    %requests = memref.load %task[%cancel_requests_slot] {ly.atomic.ordering = "acquire", ly.atomic.role = "asyncio.task.cancel.requests.load"} : memref<12xi64>
    %header, %meta, %digits = func.call @LyLong_FromI64(%requests) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %header, %meta, %digits : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyTask_Uncancel(%task: memref<12xi64> {ly.ownership.object_header}, %coroutine: memref<5xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "_asyncio.Task", ly.runtime.method = "uncancel", ly.runtime.result_contract = "builtins.int"} {
    %cancel_requests_slot = arith.constant 3 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %previous = memref.generic_atomic_rmw %task[%cancel_requests_slot] : memref<12xi64> {
    ^bb0(%current : i64):
      %has_request = arith.cmpi sgt, %current, %zero : i64
      %decremented = arith.subi %current, %one : i64
      %next = arith.select %has_request, %decremented, %current : i1, i64
      memref.atomic_yield %next : i64
    } {ly.atomic.ordering = "acq_rel", ly.atomic.role = "asyncio.task.uncancel"}
    %had_request = arith.cmpi sgt, %previous, %zero : i64
    %decremented = arith.subi %previous, %one : i64
    %remaining = arith.select %had_request, %decremented, %previous : i1, i64
    %header, %meta, %digits = func.call @LyLong_FromI64(%remaining) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %header, %meta, %digits : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyTask_GetCoro(%task: memref<12xi64> {ly.ownership.object_header}, %coroutine: memref<5xi64> {ly.ownership.object_header}) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "_asyncio.Task", ly.runtime.method = "get_coro", ly.runtime.result_contract = "types.CoroutineType", ly.runtime.result_evidence = "receiver"} {
    %coroutine_storage = memref.cast %coroutine : memref<5xi64> to memref<?xi64>
    func.call @__ly_asyncio_retain_storage(%coroutine_storage) : (memref<?xi64>) -> ()
    func.return %coroutine : memref<5xi64>
  }

  func.func @LyTask_SetResult(%task: memref<12xi64> {ly.ownership.object_header}, %coroutine: memref<5xi64> {ly.ownership.object_header}, %result: memref<?xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "_asyncio.Task", ly.runtime.method = "set_result", ly.runtime.result_contract = "types.NoneType"} {
    func.call @__ly_task_raise_set_result_unsupported() : () -> ()
    func.return
  }

  func.func @LyTask_SetException(%task: memref<12xi64> {ly.ownership.object_header}, %coroutine: memref<5xi64> {ly.ownership.object_header}, %exception: memref<?xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "_asyncio.Task", ly.runtime.method = "set_exception", ly.runtime.result_contract = "types.NoneType"} {
    func.call @__ly_task_raise_set_exception_unsupported() : () -> ()
    func.return
  }

  func.func @LyTask_ResumeBegin(%task: memref<12xi64> {ly.ownership.object_header}, %coroutine: memref<5xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "_asyncio.Task", ly.runtime.primitive = "resume.begin"} {
    %state_slot = arith.constant 2 : index
    %pending = arith.constant 0 : i64
    %running = arith.constant 1 : i64
    %previous = memref.generic_atomic_rmw %task[%state_slot] : memref<12xi64> {
    ^bb0(%current : i64):
      %is_pending = arith.cmpi eq, %current, %pending : i64
      %next = arith.select %is_pending, %running, %current : i1, i64
      memref.atomic_yield %next : i64
    } {ly.atomic.ordering = "acq_rel", ly.atomic.role = "asyncio.task.resume.begin"}
    %was_pending = arith.cmpi eq, %previous, %pending : i64
    cf.cond_br %was_pending, ^resume_coroutine, ^done

  ^resume_coroutine:
    %can_resume = func.call @LyCoroutine_ResumeBegin(%coroutine) : (memref<5xi64>) -> i1
    cf.assert %can_resume, "Task resumed a coroutine that is not resumable"
    cf.br ^done

  ^done:
    func.return %was_pending : i1
  }

  func.func @LyTask_ResumeComplete(%task: memref<12xi64> {ly.ownership.object_header}, %coroutine: memref<5xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "_asyncio.Task", ly.runtime.primitive = "resume.complete"} {
    %state_slot = arith.constant 2 : index
    %running = arith.constant 1 : i64
    %finished = arith.constant 2 : i64

    func.call @LyCoroutine_ResumeComplete(%coroutine) : (memref<5xi64>) -> ()
    %previous = memref.generic_atomic_rmw %task[%state_slot] : memref<12xi64> {
    ^bb0(%current : i64):
      %is_running = arith.cmpi eq, %current, %running : i64
      %next = arith.select %is_running, %finished, %current : i1, i64
      memref.atomic_yield %next : i64
    } {ly.atomic.ordering = "acq_rel", ly.atomic.role = "asyncio.task.resume.complete"}
    %was_running = arith.cmpi eq, %previous, %running : i64
    cf.assert %was_running, "LyTask_ResumeComplete called for a task that is not running"
    func.return
  }

  func.func @LyTask_MarkFinished(%task: memref<12xi64> {ly.ownership.object_header}, %coroutine: memref<5xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "_asyncio.Task", ly.runtime.primitive = "finish.request"} {
    %state_slot = arith.constant 2 : index
    %pending = arith.constant 0 : i64
    %running = arith.constant 1 : i64
    %finished = arith.constant 2 : i64
    %previous = memref.generic_atomic_rmw %task[%state_slot] : memref<12xi64> {
    ^bb0(%current : i64):
      %is_pending = arith.cmpi eq, %current, %pending : i64
      %is_running = arith.cmpi eq, %current, %running : i64
      %can_finish = arith.ori %is_pending, %is_running : i1
      %next = arith.select %can_finish, %finished, %current : i1, i64
      memref.atomic_yield %next : i64
    } {ly.atomic.ordering = "acq_rel", ly.atomic.role = "asyncio.task.finish"}
    %was_pending = arith.cmpi eq, %previous, %pending : i64
    %was_running = arith.cmpi eq, %previous, %running : i64
    %finished_now = arith.ori %was_pending, %was_running : i1
    func.return %finished_now : i1
  }

  func.func private @LyTaskIter_New(%task: memref<12xi64> {ly.ownership.object_header}, %coroutine: memref<5xi64> {ly.ownership.object_header}) -> (memref<3xi64>, memref<12xi64>, memref<5xi64>) attributes {ly.ownership.owned_results = [0]} {
    %one = arith.constant 1 : i64
    %zero = arith.constant 0 : i64
    %layout_task_iter = arith.constant 17 : i64
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %consumed_slot = arith.constant 2 : index

    %iterator = memref.alloc() {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<3xi64>
    memref.store %one, %iterator[%refcount_slot] : memref<3xi64>
    memref.store %layout_task_iter, %iterator[%layout_slot] : memref<3xi64>
    memref.store %zero, %iterator[%consumed_slot] : memref<3xi64>

    %task_storage = memref.cast %task : memref<12xi64> to memref<?xi64>
    func.call @__ly_asyncio_retain_storage(%task_storage) : (memref<?xi64>) -> ()
    func.return %iterator, %task, %coroutine : memref<3xi64>, memref<12xi64>, memref<5xi64>
  }

  func.func @LyTask_Await(%task: memref<12xi64> {ly.ownership.object_header}, %coroutine: memref<5xi64> {ly.ownership.object_header}) -> (memref<3xi64>, memref<12xi64>, memref<5xi64>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "_asyncio.Task", ly.runtime.method = "__await__", ly.runtime.result_contract = "_asyncio.TaskIter"} {
    %iterator, %kept_task, %kept_coroutine = func.call @LyTaskIter_New(%task, %coroutine) : (memref<12xi64>, memref<5xi64>) -> (memref<3xi64>, memref<12xi64>, memref<5xi64>)
    func.return %iterator, %kept_task, %kept_coroutine : memref<3xi64>, memref<12xi64>, memref<5xi64>
  }

  func.func @LyTask_Iter(%task: memref<12xi64> {ly.ownership.object_header}, %coroutine: memref<5xi64> {ly.ownership.object_header}) -> (memref<3xi64>, memref<12xi64>, memref<5xi64>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "_asyncio.Task", ly.runtime.method = "__iter__", ly.runtime.result_contract = "_asyncio.TaskIter"} {
    %iterator, %kept_task, %kept_coroutine = func.call @LyTaskIter_New(%task, %coroutine) : (memref<12xi64>, memref<5xi64>) -> (memref<3xi64>, memref<12xi64>, memref<5xi64>)
    func.return %iterator, %kept_task, %kept_coroutine : memref<3xi64>, memref<12xi64>, memref<5xi64>
  }

  func.func @LyTask_AddDoneCallback(%task: memref<12xi64> {ly.ownership.object_header}, %coroutine: memref<5xi64> {ly.ownership.object_header}, %callback: memref<?xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "_asyncio.Task", ly.runtime.method = "add_done_callback", ly.runtime.result_contract = "types.NoneType"} {
    %ready_callbacks_slot = arith.constant 4 : index
    %one = arith.constant 1 : i64
    memref.generic_atomic_rmw %task[%ready_callbacks_slot] : memref<12xi64> {
    ^bb0(%current : i64):
      %next = arith.addi %current, %one : i64
      memref.atomic_yield %next : i64
    } {ly.atomic.ordering = "acq_rel", ly.atomic.role = "asyncio.task.callback.record"}
    func.return
  }

  func.func @LyTaskIter_Iter(%iterator: memref<3xi64> {ly.ownership.object_header}, %task: memref<12xi64> {ly.ownership.object_header}, %coroutine: memref<5xi64> {ly.ownership.object_header}) -> (memref<3xi64>, memref<12xi64>, memref<5xi64>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "_asyncio.TaskIter", ly.runtime.method = "__iter__", ly.runtime.result_contract = "_asyncio.TaskIter"} {
    %iterator_storage = memref.cast %iterator : memref<3xi64> to memref<?xi64>
    %task_storage = memref.cast %task : memref<12xi64> to memref<?xi64>
    func.call @__ly_asyncio_retain_storage(%iterator_storage) : (memref<?xi64>) -> ()
    func.call @__ly_asyncio_retain_storage(%task_storage) : (memref<?xi64>) -> ()
    func.return %iterator, %task, %coroutine : memref<3xi64>, memref<12xi64>, memref<5xi64>
  }

  func.func @LyTask_DecRef(%task: memref<12xi64> {ly.ownership.object_header}, %coroutine: memref<5xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "_asyncio.Task", ly.runtime.deallocator} {
    %task_storage = memref.cast %task : memref<12xi64> to memref<?xi64>
    %became_zero = func.call @__ly_asyncio_release_storage_to_zero(%task_storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    func.call @LyCoroutine_DecRef(%coroutine) : (memref<5xi64>) -> ()
    memref.dealloc %task : memref<12xi64>
    cf.br ^done

  ^done:
    func.return
  }

  func.func @LyTaskIter_DecRef(%iterator: memref<3xi64> {ly.ownership.object_header}, %task: memref<12xi64> {ly.ownership.object_header}, %coroutine: memref<5xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "_asyncio.TaskIter", ly.runtime.deallocator} {
    %iterator_storage = memref.cast %iterator : memref<3xi64> to memref<?xi64>
    %became_zero = func.call @__ly_asyncio_release_storage_to_zero(%iterator_storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    func.call @LyTask_DecRef(%task, %coroutine) : (memref<12xi64>, memref<5xi64>) -> ()
    memref.dealloc %iterator : memref<3xi64>
    cf.br ^done

  ^done:
    func.return
  }
}
