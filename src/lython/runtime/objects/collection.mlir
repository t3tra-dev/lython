// Minimal runtime ABI for Python collection objects whose full structural
// contracts are defined in typing.mlir.
//
// The current object ABI carries concrete objects as contract-specific physical
// value bundles. A fully correct tuple/dict payload therefore requires a generic
// boxed object handle before elements can be owned and recovered uniformly.
// This module provides the shared collection object shell and length semantics
// so variadic callable ABI can pass through the same runtime path as every other
// object. Lowering may attach local element evidence for statically visible
// literals/call packs, but ABI-crossing element payloads still require a generic
// boxed object handle before they can be owned and recovered uniformly.

module attributes {ly.runtime.contracts = ["builtins.tuple", "builtins.dict"]} {
  func.func private @LyObject_DecRefHeader(%header: memref<2xi64, strided<[1], offset: ?>> {ly.ownership.object_header}) -> i1 attributes {ly.ownership.object_release_to_zero, ly.runtime.contract = "builtins.object", ly.runtime.primitive = "release_to_zero"}

  func.func private @LyTuple_Shape() -> (memref<2xi64>, memref<1xi64>) attributes {ly.runtime.contract = "builtins.tuple", ly.runtime.shape}

  func.func private @LyDict_Shape() -> (memref<2xi64>, memref<1xi64>) attributes {ly.runtime.contract = "builtins.dict", ly.runtime.shape}

  func.func private @__ly_collection_alloc(%class_id: i64, %length: i64) -> (memref<2xi64>, memref<1xi64>) {
    %one = arith.constant 1 : i64
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %length_slot = arith.constant 0 : index

    %header = memref.alloc() {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<2xi64>
    %meta = memref.alloc() : memref<1xi64>

    memref.store %one, %header[%refcount_slot] : memref<2xi64>
    memref.store %class_id, %header[%layout_slot] : memref<2xi64>
    memref.store %length, %meta[%length_slot] : memref<1xi64>
    func.return %header, %meta : memref<2xi64>, memref<1xi64>
  }

  func.func @LyTuple_FromLength(%length: i64 {ly.runtime.default_i64 = 0 : i64}) -> (memref<2xi64>, memref<1xi64>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 11 : i64, ly.runtime.contract = "builtins.tuple", ly.runtime.initializer = "__new__"} {
    %class_id = arith.constant 11 : i64
    %header, %meta = func.call @__ly_collection_alloc(%class_id, %length) : (i64, i64) -> (memref<2xi64>, memref<1xi64>)
    func.return %header, %meta : memref<2xi64>, memref<1xi64>
  }

  func.func @LyDict_FromLength(%length: i64 {ly.runtime.default_i64 = 0 : i64}) -> (memref<2xi64>, memref<1xi64>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 12 : i64, ly.runtime.contract = "builtins.dict", ly.runtime.initializer = "__new__"} {
    %class_id = arith.constant 12 : i64
    %header, %meta = func.call @__ly_collection_alloc(%class_id, %length) : (i64, i64) -> (memref<2xi64>, memref<1xi64>)
    func.return %header, %meta : memref<2xi64>, memref<1xi64>
  }

  func.func @LyTuple_Len(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<1xi64>) -> i64 attributes {ly.runtime.contract = "builtins.tuple", ly.runtime.method = "__len__"} {
    %length_slot = arith.constant 0 : index
    %length = memref.load %meta[%length_slot] : memref<1xi64>
    func.return %length : i64
  }

  func.func @LyDict_Len(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<1xi64>) -> i64 attributes {ly.runtime.contract = "builtins.dict", ly.runtime.method = "__len__"} {
    %length_slot = arith.constant 0 : index
    %length = memref.load %meta[%length_slot] : memref<1xi64>
    func.return %length : i64
  }

  func.func @LyTuple_DecRef(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<1xi64>) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.tuple", ly.runtime.deallocator} {
    %header_view = memref.cast %header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    %became_zero = func.call @LyObject_DecRefHeader(%header_view) : (memref<2xi64, strided<[1], offset: ?>>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    memref.dealloc %meta {ly.ownership.object_dealloc_part = "meta"} : memref<1xi64>
    memref.dealloc %header {ly.ownership.object_dealloc_part = "header"} : memref<2xi64>
    cf.br ^done

  ^done:
    func.return
  }

  func.func @LyDict_DecRef(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<1xi64>) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.dict", ly.runtime.deallocator} {
    %header_view = memref.cast %header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    %became_zero = func.call @LyObject_DecRefHeader(%header_view) : (memref<2xi64, strided<[1], offset: ?>>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    memref.dealloc %meta {ly.ownership.object_dealloc_part = "meta"} : memref<1xi64>
    memref.dealloc %header {ly.ownership.object_dealloc_part = "header"} : memref<2xi64>
    cf.br ^done

  ^done:
    func.return
  }
}
