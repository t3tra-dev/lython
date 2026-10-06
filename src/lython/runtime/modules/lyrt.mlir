// Contract manifest for Lython runtime helper classes.

module attributes {
  ly.runtime.contracts = ["lyrt.Counter"],
  ly.typing.module = "lyrt"
} {
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

  // ===========================================================
  // Runtime implementations (lyrt fixtures).
  // ===========================================================

  // ===== impls: lyrt_counter =====
  func.func private @Ly_IncRef(%header: memref<2xi64, strided<[1], offset: ?>> {ly.ownership.object_header})
  func.func private @LyObject_ReleaseStorageToZero(%storage: memref<?xi64>) -> i1
  func.func private @LyLong_FromI64(%value: i64) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0]}

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

  func.func @LyCounter_Next(%counter: memref<4xi64> {ly.ownership.object_header}) -> (memref<2xi64>, i1, memref<4xi64>) attributes {ly.ownership.owned_result_contracts = ["builtins.int", "lyrt.Counter"], ly.ownership.owned_results = [0, 2], ly.runtime.contract = "lyrt.Counter", ly.runtime.method = "__next__", ly.runtime.element_contract = "builtins.int", ly.runtime.next_contract = "lyrt.Counter", ly.runtime.next_evidence = "receiver", ly.runtime.valid_result_index = 1 : i64} {
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
    %result = func.call @LyLong_FromI64(%value) : (i64) -> memref<2xi64>

    %c0 = arith.constant 0 : index
    %header = memref.subview %counter[%c0] [2] [1] : memref<4xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%header) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %result, %valid, %counter : memref<2xi64>, i1, memref<4xi64>
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

}
