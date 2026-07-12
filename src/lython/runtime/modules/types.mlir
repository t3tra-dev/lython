// Contract manifest for stdlib `types`.

module attributes {
  ly.runtime.contracts = ["types.GeneratorType", "types.CoroutineType", "types.CoroutineAwaitIterator", "builtins.function"],ly.typing.module = "types"} {
  py.class @NoneType attributes {
    base_names = ["object"], ly.typing.final,
    method_names = ["__bool__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"types.NoneType">] -> [!py.literal<False>]>
    ],
    method_kinds = ["instance"]
  } {}

  py.class @NotImplementedType attributes {base_names = ["object"],
                                          ly.typing.final} {}
  py.class @EllipsisType attributes {base_names = ["object"],
                                    ly.typing.final} {}
  py.class @UnionType attributes {base_names = ["object"], ly.typing.final} {}

  py.class @TracebackType attributes {
    base_names = ["object"], ly.typing.final,
    field_names = ["tb_next", "tb_frame", "tb_lasti", "tb_lineno"],
    field_contract_types = [
      !py.union<!py.contract<"types.TracebackType">, !py.literal<None>>,
      !py.contract<"types.FrameType">,
      !py.contract<"builtins.int">,
      !py.contract<"builtins.int">
    ]
  } {}
  py.class @FrameType attributes {base_names = ["object"], ly.typing.abstract} {}
  py.class @GenericAlias attributes {base_names = ["object"], ly.typing.abstract} {}

  py.class @GeneratorType attributes {
    base_names = ["Generator"], ly.typing.final,
    ly.typing.params = ["Y", "S", "R"],
    ly.typing.param_variance = ["covariant", "contravariant", "covariant"],
    ly.typing.base_args = [[!py.contract<"$Y">, !py.contract<"$S">, !py.contract<"$R">]],
    ly.runtime.contract = "types.GeneratorType", ly.runtime.required,
    ly.runtime.required_deallocator,
    ly.runtime.required_initializers = ["__new__"],
    ly.runtime.required_methods = ["__iter__", "close"],
    ly.runtime.required_primitives = ["resume.begin", "resume.complete", "resume.suspend"],
    method_names = ["__iter__", "close"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"types.GeneratorType", [!py.contract<"$Y">, !py.contract<"$S">, !py.contract<"$R">]>] -> [!py.contract<"types.GeneratorType", [!py.contract<"$Y">, !py.contract<"$S">, !py.contract<"$R">]>]>,
      !py.protocol<"Callable", [!py.contract<"types.GeneratorType", [!py.contract<"$Y">, !py.contract<"$S">, !py.contract<"$R">]>] -> [!py.literal<None>]>
    ],
    method_kinds = ["instance", "instance"]
  } {}

  py.class @CoroutineAwaitIterator attributes {
    base_names = ["Generator"], ly.typing.final, ly.typing.params = ["R"],
    ly.typing.base_args = [[!py.contract<"typing.Any">, !py.literal<None>, !py.contract<"$R">]],
    method_names = ["__iter__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"types.CoroutineAwaitIterator", [!py.contract<"$R">]>] -> [!py.protocol<"Generator", [!py.contract<"typing.Any">, !py.literal<None>, !py.contract<"$R">]>]>
    ],
    method_kinds = ["instance"]
  } {}

  py.class @CoroutineType attributes {
    base_names = ["Coroutine"], ly.typing.final,
    ly.typing.params = ["Y", "S", "R"],
    ly.typing.param_variance = ["covariant", "contravariant", "covariant"],
    ly.typing.base_args = [[!py.contract<"$Y">, !py.contract<"$S">, !py.contract<"$R">]],
    method_names = ["__await__", "send", "throw", "throw", "close"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"types.CoroutineType", [!py.contract<"$Y">, !py.contract<"$S">, !py.contract<"$R">]>] -> [!py.protocol<"Generator", [!py.contract<"typing.Any">, !py.literal<None>, !py.contract<"$R">]>]>,
      !py.protocol<"Callable", [!py.contract<"types.CoroutineType", [!py.contract<"$Y">, !py.contract<"$S">, !py.contract<"$R">]>, !py.contract<"$S">] -> [!py.contract<"$Y">]>,
      !py.protocol<"Callable", [!py.contract<"types.CoroutineType", [!py.contract<"$Y">, !py.contract<"$S">, !py.contract<"$R">]>, !py.type<!py.contract<"builtins.BaseException">>, !py.union<!py.contract<"builtins.BaseException">, !py.contract<"builtins.object">>, !py.union<!py.contract<"types.TracebackType">, !py.literal<None>>] -> [!py.contract<"$Y">]>,
      !py.protocol<"Callable", [!py.contract<"types.CoroutineType", [!py.contract<"$Y">, !py.contract<"$S">, !py.contract<"$R">]>, !py.contract<"builtins.BaseException">, !py.literal<None>, !py.union<!py.contract<"types.TracebackType">, !py.literal<None>>] -> [!py.contract<"$Y">]>,
      !py.protocol<"Callable", [!py.contract<"types.CoroutineType", [!py.contract<"$Y">, !py.contract<"$S">, !py.contract<"$R">]>] -> [!py.literal<None>]>
    ],
    method_kinds = ["instance", "instance", "instance", "instance",
                    "instance"]
  } {}

  // ===========================================================
  // Runtime implementations (generators, coroutines, function objects).
  // ===========================================================

  // ===== impls: generator =====
  func.func private @Ly_IncRef(%header: memref<2xi64, strided<[1], offset: ?>> {ly.ownership.object_header})
  func.func private @LyObject_ReleaseStorageToZero(%storage: memref<?xi64>) -> i1

  func.func @LyGenerator_New(%target_id: i64) -> memref<24xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 63 : i64, ly.runtime.contract = "types.GeneratorType", ly.runtime.initializer = "__new__"} {
    %one = arith.constant 1 : i64
    %zero = arith.constant 0 : i64
    %layout_generator = arith.constant 63 : i64
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %state_slot = arith.constant 2 : index
    %target_slot = arith.constant 3 : index
    %resume_slot = arith.constant 4 : index
    %sent_valid_slot = arith.constant 5 : index
    %thrown_valid_slot = arith.constant 6 : index
    %close_requested_slot = arith.constant 7 : index

    %storage = memref.alloc() {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<24xi64>
    memref.store %one, %storage[%refcount_slot] : memref<24xi64>
    memref.store %layout_generator, %storage[%layout_slot] : memref<24xi64>
    // State 0 = created, 1 = running, 2 = suspended, 3 = exhausted, 4 = closed.
    memref.store %zero, %storage[%state_slot] : memref<24xi64>
    memref.store %target_id, %storage[%target_slot] {ly.atomic.ordering = "release", ly.atomic.role = "generator.target.publish"} : memref<24xi64>
    memref.store %zero, %storage[%resume_slot] : memref<24xi64>
    memref.store %zero, %storage[%sent_valid_slot] : memref<24xi64>
    memref.store %zero, %storage[%thrown_valid_slot] : memref<24xi64>
    memref.store %zero, %storage[%close_requested_slot] : memref<24xi64>
    // Frame slots (words 8..23): resume state-machine live values.
    %frame_lower = arith.constant 8 : index
    %frame_upper = arith.constant 24 : index
    %frame_step = arith.constant 1 : index
    scf.for %slot = %frame_lower to %frame_upper step %frame_step {
      memref.store %zero, %storage[%slot] : memref<24xi64>
    }
    func.return %storage : memref<24xi64>
  }

  func.func @LyGenerator_ResumeBegin(%storage: memref<24xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "types.GeneratorType", ly.runtime.primitive = "resume.begin"} {
    %state_slot = arith.constant 2 : index
    %created = arith.constant 0 : i64
    %running = arith.constant 1 : i64
    %suspended = arith.constant 2 : i64
    %previous = memref.generic_atomic_rmw %storage[%state_slot] : memref<24xi64> {
    ^bb0(%current : i64):
      %is_created = arith.cmpi eq, %current, %created : i64
      %is_suspended = arith.cmpi eq, %current, %suspended : i64
      %can_enter = arith.ori %is_created, %is_suspended : i1
      %next = arith.select %can_enter, %running, %current : i1, i64
      memref.atomic_yield %next : i64
    } {ly.atomic.ordering = "acq_rel", ly.atomic.role = "generator.state.resume_begin"}
    %was_created = arith.cmpi eq, %previous, %created : i64
    %was_suspended = arith.cmpi eq, %previous, %suspended : i64
    %can_resume = arith.ori %was_created, %was_suspended : i1
    func.return %can_resume : i1
  }

  func.func @LyGenerator_ResumeComplete(%storage: memref<24xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "types.GeneratorType", ly.runtime.primitive = "resume.complete"} {
    %state_slot = arith.constant 2 : index
    %running = arith.constant 1 : i64
    %exhausted = arith.constant 3 : i64
    %previous = memref.generic_atomic_rmw %storage[%state_slot] : memref<24xi64> {
    ^bb0(%current : i64):
      %is_running = arith.cmpi eq, %current, %running : i64
      %next = arith.select %is_running, %exhausted, %current : i1, i64
      memref.atomic_yield %next : i64
    } {ly.atomic.ordering = "acq_rel", ly.atomic.role = "generator.state.resume_complete"}
    %was_running = arith.cmpi eq, %previous, %running : i64
    cf.assert %was_running, "LyGenerator_ResumeComplete called for a generator that is not running"
    func.return
  }

  func.func @LyGenerator_Suspend(%storage: memref<24xi64> {ly.ownership.object_header}, %resume_index: i64) attributes {ly.runtime.contract = "types.GeneratorType", ly.runtime.primitive = "resume.suspend"} {
    %state_slot = arith.constant 2 : index
    %resume_slot = arith.constant 4 : index
    %running = arith.constant 1 : i64
    %suspended = arith.constant 2 : i64

    memref.store %resume_index, %storage[%resume_slot] : memref<24xi64>
    %previous = memref.generic_atomic_rmw %storage[%state_slot] : memref<24xi64> {
    ^bb0(%current : i64):
      %is_running = arith.cmpi eq, %current, %running : i64
      %next = arith.select %is_running, %suspended, %current : i1, i64
      memref.atomic_yield %next : i64
    } {ly.atomic.ordering = "acq_rel", ly.atomic.role = "generator.state.resume_suspend"}
    %was_running = arith.cmpi eq, %previous, %running : i64
    cf.assert %was_running, "LyGenerator_Suspend called for a generator that is not running"
    func.return
  }

  func.func @LyGenerator_Iter(%storage: memref<24xi64> {ly.ownership.object_header}) -> memref<24xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "types.GeneratorType", ly.runtime.method = "__iter__", ly.runtime.result_contract = "types.GeneratorType"} {
    %c0 = arith.constant 0 : index
    %header = memref.subview %storage[%c0] [2] [1] : memref<24xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%header) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %storage : memref<24xi64>
  }

  func.func @LyGenerator_Close(%storage: memref<24xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "types.GeneratorType", ly.runtime.method = "close", ly.runtime.result_contract = "types.NoneType"} {
    %state_slot = arith.constant 2 : index
    %close_requested_slot = arith.constant 7 : index
    %running = arith.constant 1 : i64
    %closed = arith.constant 4 : i64

    %previous = memref.generic_atomic_rmw %storage[%state_slot] : memref<24xi64> {
    ^bb0(%current : i64):
      %is_running = arith.cmpi eq, %current, %running : i64
      %next = arith.select %is_running, %current, %closed : i1, i64
      memref.atomic_yield %next : i64
    } {ly.atomic.ordering = "acq_rel", ly.atomic.role = "generator.state.close"}
    %was_running = arith.cmpi eq, %previous, %running : i64
    %true = arith.constant true
    %can_close = arith.xori %was_running, %true : i1
    cf.assert %can_close, "cannot close a running generator"
    %one = arith.constant 1 : i64
    memref.store %one, %storage[%close_requested_slot] : memref<24xi64>
    func.return
  }

  func.func @LyGenerator_DecRef(%storage: memref<24xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "types.GeneratorType", ly.runtime.deallocator} {
    %dynamic_storage = memref.cast %storage : memref<24xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%dynamic_storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    memref.dealloc %storage : memref<24xi64>
    cf.br ^done

  ^done:
    func.return
  }

  // ===== impls: coroutine =====
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
    memref.store %target_id, %storage[%target_slot] {ly.atomic.ordering = "release", ly.atomic.role = "coroutine.target.publish"} : memref<5xi64>
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
    %layout_await_iter = arith.constant 18 : i64
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
    %dynamic_storage = memref.cast %storage : memref<5xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%dynamic_storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    memref.dealloc %storage : memref<5xi64>
    cf.br ^done

  ^done:
    func.return
  }

  func.func @LyCoroutineAwaitIterator_DecRef(%iterator: memref<3xi64> {ly.ownership.object_header}, %coroutine: memref<5xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "types.CoroutineAwaitIterator", ly.runtime.deallocator} {
    %dynamic_iterator = memref.cast %iterator : memref<3xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%dynamic_iterator) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    func.call @LyCoroutine_DecRef(%coroutine) : (memref<5xi64>) -> ()
    memref.dealloc %iterator : memref<3xi64>
    cf.br ^done

  ^done:
    func.return
  }

  // ===== impls: function =====
  func.func @LyFunction_New(%target_id: i64, %defaults: i64, %kwdefaults: i64, %closure: i64, %annotations: i64, %module: i64) -> memref<8xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 6 : i64, ly.runtime.contract = "builtins.function", ly.runtime.initializer = "__new__"} {
    %one = arith.constant 1 : i64
    %layout_function = arith.constant 6 : i64
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %target_slot = arith.constant 2 : index
    %defaults_slot = arith.constant 3 : index
    %kwdefaults_slot = arith.constant 4 : index
    %closure_slot = arith.constant 5 : index
    %annotations_slot = arith.constant 6 : index
    %module_slot = arith.constant 7 : index

    %storage = memref.alloc() {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<8xi64>

    memref.store %one, %storage[%refcount_slot] : memref<8xi64>
    memref.store %layout_function, %storage[%layout_slot] : memref<8xi64>
    memref.store %target_id, %storage[%target_slot] : memref<8xi64>
    memref.store %defaults, %storage[%defaults_slot] : memref<8xi64>
    memref.store %kwdefaults, %storage[%kwdefaults_slot] : memref<8xi64>
    memref.store %closure, %storage[%closure_slot] : memref<8xi64>
    memref.store %annotations, %storage[%annotations_slot] : memref<8xi64>
    memref.store %module, %storage[%module_slot] : memref<8xi64>

    func.return %storage : memref<8xi64>
  }

  func.func @LyFunction_DecRef(%storage: memref<8xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.function", ly.runtime.deallocator} {
    %dynamic_storage = memref.cast %storage : memref<8xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%dynamic_storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    memref.dealloc %storage : memref<8xi64>
    cf.br ^done

  ^done:
    func.return
  }
}
