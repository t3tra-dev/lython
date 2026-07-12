// Contract manifest for Lython runtime helper classes.

module attributes {
  ly.runtime.contracts = ["lyrt.Counter", "lyrt.ReadyIntAwaitable", "lyrt.AsyncCounter", "lyrt.ReadyAsyncCounter"],
  ly.typing.module = "lyrt",
  ly.typing.class_exports = [
    "lyrt.ReadyIntAwaitable=lyrt.ReadyIntAwaitable"
  ]
} {
  py.class @ReadyIntAwaitable attributes {
    base_names = ["Awaitable"], ly.typing.final,
    ly.typing.base_args = [[!py.contract<"builtins.int">]],
    ly.runtime.contract = "lyrt.ReadyIntAwaitable", ly.runtime.required,
    ly.runtime.required_deallocator,
    ly.runtime.required_initializers = ["__new__"],
    ly.runtime.required_methods = ["__init__", "__await__"],
    method_names = ["__new__", "__init__", "__await__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.type<!py.contract<"lyrt.ReadyIntAwaitable">>, !py.contract<"builtins.int">] -> [!py.self]>,
      !py.protocol<"Callable", [!py.contract<"lyrt.ReadyIntAwaitable">, !py.contract<"builtins.int">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"lyrt.ReadyIntAwaitable">] -> [!py.contract<"_asyncio.FutureIter", [!py.contract<"builtins.int">]>]>
    ],
    method_kinds = ["classmethod", "instance", "instance"]
  } {}

  py.class @Counter attributes {
    base_names = ["Iterator"], ly.typing.final,
    ly.typing.base_args = [[!py.contract<"builtins.int">]],
    ly.runtime.contract = "lyrt.Counter", ly.runtime.required,
    ly.runtime.required_deallocator,
    ly.runtime.required_initializers = ["__new__"],
    ly.runtime.required_methods = ["__init__", "__iter__", "__next__"],
    method_names = ["__new__", "__init__", "__iter__", "__next__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.type<!py.contract<"lyrt.Counter">>, !py.contract<"builtins.int">] -> [!py.self]>,
      !py.protocol<"Callable", [!py.contract<"lyrt.Counter">, !py.contract<"builtins.int">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"lyrt.Counter">] -> [!py.contract<"lyrt.Counter">]>,
      !py.protocol<"Callable", [!py.contract<"lyrt.Counter">] -> [!py.contract<"builtins.int">]>
    ],
    method_kinds = ["classmethod", "instance", "instance", "instance"]
  } {}

  py.class @AsyncCounter attributes {
    base_names = ["AsyncIterator"], ly.typing.final,
    ly.typing.base_args = [[!py.contract<"builtins.int">]],
    ly.runtime.contract = "lyrt.AsyncCounter", ly.runtime.required,
    ly.runtime.required_deallocator,
    ly.runtime.required_initializers = ["__new__"],
    ly.runtime.required_methods = ["__init__", "__aiter__", "__anext__"],
    method_names = ["__new__", "__init__", "__aiter__", "__anext__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.type<!py.contract<"lyrt.AsyncCounter">>, !py.contract<"builtins.int">] -> [!py.self]>,
      !py.protocol<"Callable", [!py.contract<"lyrt.AsyncCounter">, !py.contract<"builtins.int">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"lyrt.AsyncCounter">] -> [!py.contract<"lyrt.AsyncCounter">]>,
      !py.protocol<"Callable", [!py.contract<"lyrt.AsyncCounter">] -> [!py.contract<"_asyncio.Future", [!py.contract<"builtins.int">]>]>
    ],
    method_kinds = ["classmethod", "instance", "instance", "instance"]
  } {}

  py.class @ReadyAsyncCounter attributes {
    base_names = ["AsyncIterator"], ly.typing.final,
    ly.typing.base_args = [[!py.contract<"builtins.int">]],
    ly.runtime.contract = "lyrt.ReadyAsyncCounter", ly.runtime.required,
    ly.runtime.required_deallocator,
    ly.runtime.required_initializers = ["__new__"],
    ly.runtime.required_methods = ["__init__", "__aiter__", "__anext__"],
    method_names = ["__new__", "__init__", "__aiter__", "__anext__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.type<!py.contract<"lyrt.ReadyAsyncCounter">>, !py.contract<"builtins.int">] -> [!py.self]>,
      !py.protocol<"Callable", [!py.contract<"lyrt.ReadyAsyncCounter">, !py.contract<"builtins.int">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"lyrt.ReadyAsyncCounter">] -> [!py.contract<"lyrt.ReadyAsyncCounter">]>,
      !py.protocol<"Callable", [!py.contract<"lyrt.ReadyAsyncCounter">] -> [!py.contract<"lyrt.ReadyIntAwaitable">]>
    ],
    method_kinds = ["classmethod", "instance", "instance", "instance"]
  } {}

  // ===========================================================
  // Runtime implementations (lyrt fixtures).
  // ===========================================================

  // ===== impls: lyrt_counter =====
  func.func private @Ly_IncRef(%header: memref<2xi64, strided<[1], offset: ?>> {ly.ownership.object_header})
  func.func private @LyObject_ReleaseStorageToZero(%storage: memref<?xi64>) -> i1
  func.func private @LyLong_FromI64(%value: i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0]}

  func.func @LyCounter_New(%limit: i64 {ly.runtime.default_i64 = 0 : i64}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 19 : i64, ly.runtime.contract = "lyrt.Counter", ly.runtime.initializer = "__new__"} {
    %one = arith.constant 1 : i64
    %zero = arith.constant 0 : i64
    %layout_counter = arith.constant 19 : i64
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %current_slot = arith.constant 2 : index
    %limit_slot = arith.constant 3 : index

    %counter = memref.alloc() {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<4xi64>
    memref.store %one, %counter[%refcount_slot] : memref<4xi64>
    memref.store %layout_counter, %counter[%layout_slot] : memref<4xi64>
    memref.store %zero, %counter[%current_slot] : memref<4xi64>
    memref.store %limit, %counter[%limit_slot] : memref<4xi64>
    func.return %counter : memref<4xi64>
  }

  func.func @LyCounter_Init(%counter: memref<4xi64> {ly.ownership.object_header}, %limit: i64 {ly.runtime.default_i64 = 0 : i64}) attributes {ly.runtime.contract = "lyrt.Counter", ly.runtime.method = "__init__", ly.runtime.result_contract = "types.NoneType"} {
    %zero = arith.constant 0 : i64
    %current_slot = arith.constant 2 : index
    %limit_slot = arith.constant 3 : index
    memref.store %zero, %counter[%current_slot] : memref<4xi64>
    memref.store %limit, %counter[%limit_slot] : memref<4xi64>
    func.return
  }

  func.func @LyCounter_Iter(%counter: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "lyrt.Counter", ly.runtime.method = "__iter__", ly.runtime.result_contract = "lyrt.Counter", ly.runtime.result_evidence = "receiver"} {
    %c0 = arith.constant 0 : index
    %header = memref.subview %counter[%c0] [2] [1] : memref<4xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%header) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %counter : memref<4xi64>
  }

  func.func @LyCounter_Next(%counter: memref<4xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>, i1, memref<4xi64>) attributes {ly.ownership.owned_results = [0, 4], ly.runtime.contract = "lyrt.Counter", ly.runtime.method = "__next__", ly.runtime.element_contract = "builtins.int", ly.runtime.next_contract = "lyrt.Counter", ly.runtime.next_evidence = "receiver", ly.runtime.valid_result_index = 3 : i64} {
    %current_slot = arith.constant 2 : index
    %limit_slot = arith.constant 3 : index
    %current = memref.load %counter[%current_slot] : memref<4xi64>
    %limit = memref.load %counter[%limit_slot] : memref<4xi64>
    %valid = arith.cmpi slt, %current, %limit : i64
    %one = arith.constant 1 : i64
    %zero = arith.constant 0 : i64
    %next_current_candidate = arith.addi %current, %one : i64
    %next_current = arith.select %valid, %next_current_candidate, %current : i1, i64
    %value = arith.select %valid, %current, %zero : i1, i64
    memref.store %next_current, %counter[%current_slot] : memref<4xi64>
    %result:3 = func.call @LyLong_FromI64(%value) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)

    %c0 = arith.constant 0 : index
    %header = memref.subview %counter[%c0] [2] [1] : memref<4xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%header) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %result#0, %result#1, %result#2, %valid, %counter : memref<2xi64>, memref<2xi64>, memref<?xi32>, i1, memref<4xi64>
  }

  func.func @LyCounter_DecRef(%counter: memref<4xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "lyrt.Counter", ly.runtime.deallocator} {
    %storage = memref.cast %counter : memref<4xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    memref.dealloc %counter : memref<4xi64>
    cf.br ^done

  ^done:
    func.return
  }

  // ===== impls: lyrt_awaitable =====
  func.func private @LyFuture_New() -> memref<10xi64> attributes {ly.ownership.owned_results = [0]}

  func.func @LyReadyIntAwaitable_New(%value: i64 {ly.runtime.default_i64 = 0 : i64}) -> memref<3xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 17 : i64, ly.runtime.contract = "lyrt.ReadyIntAwaitable", ly.runtime.initializer = "__new__"} {
    %one = arith.constant 1 : i64
    %layout_ready_int_awaitable = arith.constant 17 : i64
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %value_slot = arith.constant 2 : index

    %awaitable = memref.alloc() {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<3xi64>
    memref.store %one, %awaitable[%refcount_slot] : memref<3xi64>
    memref.store %layout_ready_int_awaitable, %awaitable[%layout_slot] : memref<3xi64>
    memref.store %value, %awaitable[%value_slot] : memref<3xi64>
    func.return %awaitable : memref<3xi64>
  }

  func.func @LyReadyIntAwaitable_Init(%awaitable: memref<3xi64> {ly.ownership.object_header}, %value: i64 {ly.runtime.default_i64 = 0 : i64}) attributes {ly.runtime.contract = "lyrt.ReadyIntAwaitable", ly.runtime.method = "__init__", ly.runtime.result_contract = "types.NoneType"} {
    %value_slot = arith.constant 2 : index
    memref.store %value, %awaitable[%value_slot] : memref<3xi64>
    func.return
  }

  func.func @LyReadyIntAwaitable_Await(%awaitable: memref<3xi64> {ly.ownership.object_header}) -> (memref<3xi64>, memref<10xi64>, memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0, 2], ly.runtime.contract = "lyrt.ReadyIntAwaitable", ly.runtime.method = "__await__", ly.runtime.result_contract = "_asyncio.FutureIter", ly.runtime.result_evidence_slots = ["asyncio.future.result"], ly.runtime.result_evidence_contracts = ["builtins.int"]} {
    %one = arith.constant 1 : i64
    %zero = arith.constant 0 : i64
    %layout_future_iter = arith.constant 16 : i64
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %consumed_slot = arith.constant 2 : index
    %value_slot = arith.constant 2 : index

    %future = func.call @LyFuture_New() : () -> memref<10xi64>
    %iterator = memref.alloc() {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<3xi64>
    memref.store %one, %iterator[%refcount_slot] : memref<3xi64>
    memref.store %layout_future_iter, %iterator[%layout_slot] : memref<3xi64>
    memref.store %zero, %iterator[%consumed_slot] : memref<3xi64>

    %value = memref.load %awaitable[%value_slot] : memref<3xi64>
    %result:3 = func.call @LyLong_FromI64(%value) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %iterator, %future, %result#0, %result#1, %result#2 : memref<3xi64>, memref<10xi64>, memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyReadyIntAwaitable_DecRef(%awaitable: memref<3xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "lyrt.ReadyIntAwaitable", ly.runtime.deallocator} {
    %storage = memref.cast %awaitable : memref<3xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    memref.dealloc %awaitable : memref<3xi64>
    cf.br ^done

  ^done:
    func.return
  }

  // ===== impls: lyrt_async_counter =====
  func.func private @LyStopAsyncIteration_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1]}
  func.func private @LyStopAsyncIteration_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1]}

  func.func @LyAsyncCounter_New(%limit: i64 {ly.runtime.default_i64 = 0 : i64}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 16 : i64, ly.runtime.contract = "lyrt.AsyncCounter", ly.runtime.initializer = "__new__"} {
    %one = arith.constant 1 : i64
    %zero = arith.constant 0 : i64
    %layout_async_counter = arith.constant 16 : i64
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %current_slot = arith.constant 2 : index
    %limit_slot = arith.constant 3 : index

    %counter = memref.alloc() {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<4xi64>
    memref.store %one, %counter[%refcount_slot] : memref<4xi64>
    memref.store %layout_async_counter, %counter[%layout_slot] : memref<4xi64>
    memref.store %zero, %counter[%current_slot] : memref<4xi64>
    memref.store %limit, %counter[%limit_slot] : memref<4xi64>
    func.return %counter : memref<4xi64>
  }

  func.func @LyAsyncCounter_Init(%counter: memref<4xi64> {ly.ownership.object_header}, %limit: i64 {ly.runtime.default_i64 = 0 : i64}) attributes {ly.runtime.contract = "lyrt.AsyncCounter", ly.runtime.method = "__init__", ly.runtime.result_contract = "types.NoneType"} {
    %zero = arith.constant 0 : i64
    %current_slot = arith.constant 2 : index
    %limit_slot = arith.constant 3 : index
    memref.store %zero, %counter[%current_slot] : memref<4xi64>
    memref.store %limit, %counter[%limit_slot] : memref<4xi64>
    func.return
  }

  func.func @LyAsyncCounter_AIter(%counter: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "lyrt.AsyncCounter", ly.runtime.method = "__aiter__", ly.runtime.result_contract = "lyrt.AsyncCounter"} {
    %c0 = arith.constant 0 : index
    %header = memref.subview %counter[%c0] [2] [1] : memref<4xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%header) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %counter : memref<4xi64>
  }

  func.func @LyAsyncCounter_ANext(%counter: memref<4xi64> {ly.ownership.object_header}) -> (memref<10xi64>, memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.contract = "lyrt.AsyncCounter", ly.runtime.method = "__anext__", ly.runtime.result_contract = "_asyncio.Future", ly.runtime.result_evidence_slots = ["asyncio.future.result"], ly.runtime.result_evidence_contracts = ["builtins.int"]} {
    %current_slot = arith.constant 2 : index
    %limit_slot = arith.constant 3 : index
    %current = memref.load %counter[%current_slot] : memref<4xi64>
    %limit = memref.load %counter[%limit_slot] : memref<4xi64>
    %valid = arith.cmpi slt, %current, %limit : i64
    %one = arith.constant 1 : i64
    %next_current_candidate = arith.addi %current, %one : i64
    %next_current = arith.select %valid, %next_current_candidate, %current : i1, i64
    memref.store %next_current, %counter[%current_slot] : memref<4xi64>

    %future = func.call @LyFuture_New() : () -> memref<10xi64>
    %zero = arith.constant 0 : i64
    %value = arith.select %valid, %current, %zero : i1, i64
    %result:3 = func.call @LyLong_FromI64(%value) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    cf.cond_br %valid, ^done, ^exhausted

  ^exhausted:
    %stop_class = arith.constant 58 : i64
    %stop:3 = func.call @LyStopAsyncIteration_New(%stop_class) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.call @LyStopAsyncIteration_Raise(%stop#0, %stop#1, %stop#2) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    cf.br ^done

  ^done:
    func.return %future, %result#0, %result#1, %result#2 : memref<10xi64>, memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyAsyncCounter_DecRef(%counter: memref<4xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "lyrt.AsyncCounter", ly.runtime.deallocator} {
    %storage = memref.cast %counter : memref<4xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    memref.dealloc %counter : memref<4xi64>
    cf.br ^done

  ^done:
    func.return
  }

  func.func @LyReadyAsyncCounter_New(%limit: i64 {ly.runtime.default_i64 = 0 : i64}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 18 : i64, ly.runtime.contract = "lyrt.ReadyAsyncCounter", ly.runtime.initializer = "__new__"} {
    %one = arith.constant 1 : i64
    %zero = arith.constant 0 : i64
    %layout_ready_async_counter = arith.constant 18 : i64
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %current_slot = arith.constant 2 : index
    %limit_slot = arith.constant 3 : index

    %counter = memref.alloc() {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<4xi64>
    memref.store %one, %counter[%refcount_slot] : memref<4xi64>
    memref.store %layout_ready_async_counter, %counter[%layout_slot] : memref<4xi64>
    memref.store %zero, %counter[%current_slot] : memref<4xi64>
    memref.store %limit, %counter[%limit_slot] : memref<4xi64>
    func.return %counter : memref<4xi64>
  }

  func.func @LyReadyAsyncCounter_Init(%counter: memref<4xi64> {ly.ownership.object_header}, %limit: i64 {ly.runtime.default_i64 = 0 : i64}) attributes {ly.runtime.contract = "lyrt.ReadyAsyncCounter", ly.runtime.method = "__init__", ly.runtime.result_contract = "types.NoneType"} {
    %zero = arith.constant 0 : i64
    %current_slot = arith.constant 2 : index
    %limit_slot = arith.constant 3 : index
    memref.store %zero, %counter[%current_slot] : memref<4xi64>
    memref.store %limit, %counter[%limit_slot] : memref<4xi64>
    func.return
  }

  func.func @LyReadyAsyncCounter_AIter(%counter: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "lyrt.ReadyAsyncCounter", ly.runtime.method = "__aiter__", ly.runtime.result_contract = "lyrt.ReadyAsyncCounter"} {
    %c0 = arith.constant 0 : index
    %header = memref.subview %counter[%c0] [2] [1] : memref<4xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%header) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %counter : memref<4xi64>
  }

  func.func @LyReadyAsyncCounter_ANext(%counter: memref<4xi64> {ly.ownership.object_header}) -> memref<3xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "lyrt.ReadyAsyncCounter", ly.runtime.method = "__anext__", ly.runtime.result_contract = "lyrt.ReadyIntAwaitable"} {
    %current_slot = arith.constant 2 : index
    %limit_slot = arith.constant 3 : index
    %current = memref.load %counter[%current_slot] : memref<4xi64>
    %limit = memref.load %counter[%limit_slot] : memref<4xi64>
    %valid = arith.cmpi slt, %current, %limit : i64
    %one = arith.constant 1 : i64
    %next_current_candidate = arith.addi %current, %one : i64
    %next_current = arith.select %valid, %next_current_candidate, %current : i1, i64
    memref.store %next_current, %counter[%current_slot] : memref<4xi64>

    %zero = arith.constant 0 : i64
    %value = arith.select %valid, %current, %zero : i1, i64
    %awaitable = func.call @LyReadyIntAwaitable_New(%value) : (i64) -> memref<3xi64>
    cf.cond_br %valid, ^done, ^exhausted

  ^exhausted:
    %stop_class = arith.constant 58 : i64
    %stop:3 = func.call @LyStopAsyncIteration_New(%stop_class) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.call @LyStopAsyncIteration_Raise(%stop#0, %stop#1, %stop#2) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    cf.br ^done

  ^done:
    func.return %awaitable : memref<3xi64>
  }

  func.func @LyReadyAsyncCounter_DecRef(%counter: memref<4xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "lyrt.ReadyAsyncCounter", ly.runtime.deallocator} {
    %storage = memref.cast %counter : memref<4xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    memref.dealloc %counter : memref<4xi64>
    cf.br ^done

  ^done:
    func.return
  }
}
