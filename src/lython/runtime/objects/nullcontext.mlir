module attributes {ly.runtime.contracts = ["contextlib.nullcontext"]} {
  func.func private @Ly_IncRef(%header: memref<2xi64, strided<[1], offset: ?>> {ly.ownership.object_header})
  func.func private @LyObject_ReleaseStorageToZero(%storage: memref<?xi64>) -> i1

  func.func @LyNullContext_New() -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 8 : i64, ly.runtime.contract = "contextlib.nullcontext", ly.runtime.initializer = "__new__"} {
    %header = memref.alloc() {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<2xi64>
    %one = arith.constant 1 : i64
    %layout_nullcontext = arith.constant 8 : i64
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    memref.store %one, %header[%refcount_slot] : memref<2xi64>
    memref.store %layout_nullcontext, %header[%layout_slot] : memref<2xi64>
    func.return %header : memref<2xi64>
  }

  func.func @LyNullContext_Init(%header: memref<2xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "contextlib.nullcontext", ly.runtime.method = "__init__"} {
    func.return
  }

  func.func @LyNullContext_Enter(%header: memref<2xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "contextlib.nullcontext", ly.runtime.method = "__enter__", ly.runtime.result_contract = "contextlib.nullcontext"} {
    %header_view = memref.cast %header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%header_view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %header : memref<2xi64>
  }

  func.func @LyNullContext_Exit(%header: memref<2xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "contextlib.nullcontext", ly.runtime.method = "__exit__", ly.runtime.result_contract = "builtins.bool"} {
    %false = arith.constant false
    func.return %false : i1
  }

  func.func @LyNullContext_DecRef(%header: memref<2xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "contextlib.nullcontext", ly.runtime.deallocator} {
    %storage = memref.cast %header : memref<2xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    memref.dealloc %header {ly.ownership.object_dealloc_part = "header"} : memref<2xi64>
    cf.br ^done

  ^done:
    func.return
  }
}
