module attributes {ly.runtime.contracts = ["builtins.float"]} {
  func.func private @LyObject_ReleaseStorageToZero(%storage: memref<?xi64>) -> i1
  func.func private @LyUnicode_FromF64(%value: f64) -> (memref<2xi64>, memref<?xi8>)

  func.func @LyFloat_FromF64(%value: f64 {ly.runtime.default_f64 = 0.0 : f64}) -> (memref<2xi64>, memref<1xf64>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 2 : i64, ly.runtime.contract = "builtins.float", ly.runtime.initializer = "__new__"} {
    %header = memref.alloc() {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<2xi64>
    %payload = memref.alloc() : memref<1xf64>
    %one = arith.constant 1 : i64
    %layout_float = arith.constant 2 : i64
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %value_slot = arith.constant 0 : index
    memref.store %one, %header[%refcount_slot] : memref<2xi64>
    memref.store %layout_float, %header[%layout_slot] : memref<2xi64>
    memref.store %value, %payload[%value_slot] : memref<1xf64>
    func.return %header, %payload : memref<2xi64>, memref<1xf64>
  }

  func.func @LyFloat_DecRef(%header: memref<2xi64> {ly.ownership.object_header}, %payload: memref<1xf64>) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.float", ly.runtime.deallocator} {
    %storage = memref.cast %header : memref<2xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    memref.dealloc %payload {ly.ownership.object_dealloc_part = "payload"} : memref<1xf64>
    memref.dealloc %header {ly.ownership.object_dealloc_part = "header"} : memref<2xi64>
    cf.br ^done

  ^done:
    func.return
  }

  func.func @LyFloat_Init(%header: memref<2xi64> {ly.ownership.object_header}, %payload: memref<1xf64>) attributes {ly.runtime.contract = "builtins.float", ly.runtime.method = "__init__"} {
    func.return
  }

  func.func @LyFloat_AsF64(%header: memref<2xi64> {ly.ownership.object_header}, %payload: memref<1xf64>) -> f64 attributes {ly.runtime.contract = "builtins.float", ly.runtime.method = "__float__", ly.runtime.primitive = "unbox.f64"} {
    %value_slot = arith.constant 0 : index
    %value = memref.load %payload[%value_slot] : memref<1xf64>
    func.return %value : f64
  }

  func.func @LyFloat_Bool(%header: memref<2xi64> {ly.ownership.object_header}, %payload: memref<1xf64>) -> i1 attributes {ly.runtime.contract = "builtins.float", ly.runtime.method = "__bool__"} {
    %value = func.call @LyFloat_AsF64(%header, %payload) : (memref<2xi64>, memref<1xf64>) -> f64
    %zero = arith.constant 0.0 : f64
    %truth = arith.cmpf une, %value, %zero : f64
    func.return %truth : i1
  }

  func.func @LyFloat_Repr(%header: memref<2xi64> {ly.ownership.object_header}, %payload: memref<1xf64>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.float", ly.runtime.method = "__repr__"} {
    %value = func.call @LyFloat_AsF64(%header, %payload) : (memref<2xi64>, memref<1xf64>) -> f64
    %str_header, %str_bytes = func.call @LyUnicode_FromF64(%value) : (f64) -> (memref<2xi64>, memref<?xi8>)
    func.return %str_header, %str_bytes : memref<2xi64>, memref<?xi8>
  }
}
