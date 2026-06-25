// Runtime storage for builtins.function.
//
// Function objects are ordinary Python objects whose manifest contract contains
// `__call__`. Creating one is therefore runtime object initialization, not a
// PyDialect operation. Lowering is expected to map the resolved function symbol
// and optional defaults/kwdefaults/closure/annotations/module objects into
// stable runtime handles before calling this ABI.

module attributes {ly.runtime.contracts = ["builtins.function"]} {
  func.func private @LyObject_DecRefHeader(%header: memref<2xi64, strided<[1], offset: ?>> {ly.ownership.object_header}) -> i1 attributes {ly.ownership.object_release_to_zero, ly.runtime.contract = "builtins.object", ly.runtime.primitive = "release_to_zero"}

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
    %c0 = arith.constant 0 : index
    %header = memref.subview %storage[%c0] [2] [1] : memref<8xi64> to memref<2xi64, strided<[1], offset: ?>>
    %became_zero = func.call @LyObject_DecRefHeader(%header) : (memref<2xi64, strided<[1], offset: ?>>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    memref.dealloc %storage {ly.ownership.object_dealloc_part = "storage"} : memref<8xi64>
    cf.br ^done

  ^done:
    func.return
  }
}
