module attributes {ly.runtime.contracts = ["builtins.BaseException", "builtins.Exception", "builtins.RuntimeError", "builtins.TypeError", "builtins.ValueError", "builtins.ArithmeticError", "builtins.LookupError", "builtins.ZeroDivisionError", "builtins.KeyError", "builtins.IndexError", "builtins.AssertionError", "builtins.StopIteration", "builtins.StopAsyncIteration", "asyncio.CancelledError"]} {
  func.func private @LyEH_ThrowException(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0], ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "raise"}
  func.func private @LyUnicode_FromBytes(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>)
  func.func private @LyUnicode_DecRef(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>)

  func.func private @LyException_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.Exception", ly.runtime.shape}
  func.func private @LyRuntimeError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.RuntimeError", ly.runtime.shape}
  func.func private @LyTypeError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.TypeError", ly.runtime.shape}
  func.func private @LyValueError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.ValueError", ly.runtime.shape}
  func.func private @LyArithmeticError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.ArithmeticError", ly.runtime.shape}
  func.func private @LyLookupError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.LookupError", ly.runtime.shape}
  func.func private @LyZeroDivisionError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.ZeroDivisionError", ly.runtime.shape}
  func.func private @LyKeyError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.KeyError", ly.runtime.shape}
  func.func private @LyIndexError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.IndexError", ly.runtime.shape}
  func.func private @LyAssertionError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.AssertionError", ly.runtime.shape}
  func.func private @LyStopIteration_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.StopIteration", ly.runtime.shape}
  func.func private @LyStopAsyncIteration_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "builtins.StopAsyncIteration", ly.runtime.shape}
  func.func private @LyCancelledError_Shape() -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.runtime.contract = "asyncio.CancelledError", ly.runtime.shape}

  func.func @LyBaseException_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.release_args = [1], ly.ownership.transfer_args = [3], ly.runtime.contract = "builtins.BaseException", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    func.call @LyUnicode_DecRef(%old_message_header, %old_message_bytes) : (memref<2xi64>, memref<?xi8>) -> ()
    func.return %header, %message_header, %message_bytes : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyException_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.release_args = [1], ly.ownership.transfer_args = [3], ly.runtime.contract = "builtins.Exception", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyRuntimeError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.release_args = [1], ly.ownership.transfer_args = [3], ly.runtime.contract = "builtins.RuntimeError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyTypeError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.release_args = [1], ly.ownership.transfer_args = [3], ly.runtime.contract = "builtins.TypeError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyValueError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.release_args = [1], ly.ownership.transfer_args = [3], ly.runtime.contract = "builtins.ValueError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyArithmeticError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.release_args = [1], ly.ownership.transfer_args = [3], ly.runtime.contract = "builtins.ArithmeticError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyLookupError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.release_args = [1], ly.ownership.transfer_args = [3], ly.runtime.contract = "builtins.LookupError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyZeroDivisionError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.release_args = [1], ly.ownership.transfer_args = [3], ly.runtime.contract = "builtins.ZeroDivisionError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyKeyError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.release_args = [1], ly.ownership.transfer_args = [3], ly.runtime.contract = "builtins.KeyError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyIndexError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.release_args = [1], ly.ownership.transfer_args = [3], ly.runtime.contract = "builtins.IndexError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyAssertionError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.release_args = [1], ly.ownership.transfer_args = [3], ly.runtime.contract = "builtins.AssertionError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyStopIteration_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.release_args = [1], ly.ownership.transfer_args = [3], ly.runtime.contract = "builtins.StopIteration", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyStopAsyncIteration_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.release_args = [1], ly.ownership.transfer_args = [3], ly.runtime.contract = "builtins.StopAsyncIteration", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyCancelledError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.release_args = [1], ly.ownership.transfer_args = [3], ly.runtime.contract = "asyncio.CancelledError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"} {
    %result:3 = func.call @LyBaseException_Init(%header, %old_message_header, %old_message_bytes, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyBaseException_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 5 : i64, ly.runtime.contract = "builtins.BaseException", ly.runtime.initializer = "__new__"} {
    %one = arith.constant 1 : i64
    %layout_exception = arith.constant 5 : i64
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %class_slot = arith.constant 2 : index
    %zero_index = arith.constant 0 : index
    %zero_len = arith.constant 0 : i64

    %header = memref.alloc() {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<3xi64>
    %empty_bytes = memref.alloca(%zero_index) : memref<?xi8>
    %message_header, %message_bytes = func.call @LyUnicode_FromBytes(%empty_bytes, %zero_index, %zero_len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)

    memref.store %one, %header[%refcount_slot] : memref<3xi64>
    memref.store %layout_exception, %header[%layout_slot] : memref<3xi64>
    memref.store %class_id, %header[%class_slot] : memref<3xi64>

    func.return %header, %message_header, %message_bytes : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyException_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 50 : i64, ly.runtime.contract = "builtins.Exception", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyRuntimeError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 51 : i64, ly.runtime.contract = "builtins.RuntimeError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyTypeError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 52 : i64, ly.runtime.contract = "builtins.TypeError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyValueError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 53 : i64, ly.runtime.contract = "builtins.ValueError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyArithmeticError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 59 : i64, ly.runtime.contract = "builtins.ArithmeticError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyLookupError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 60 : i64, ly.runtime.contract = "builtins.LookupError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyZeroDivisionError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 61 : i64, ly.runtime.contract = "builtins.ZeroDivisionError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyKeyError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 54 : i64, ly.runtime.contract = "builtins.KeyError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyIndexError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 55 : i64, ly.runtime.contract = "builtins.IndexError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyAssertionError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 56 : i64, ly.runtime.contract = "builtins.AssertionError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyStopIteration_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 57 : i64, ly.runtime.contract = "builtins.StopIteration", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyStopAsyncIteration_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 58 : i64, ly.runtime.contract = "builtins.StopAsyncIteration", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyCancelledError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 1], ly.runtime.class_id = 62 : i64, ly.runtime.contract = "asyncio.CancelledError", ly.runtime.initializer = "__new__"} {
    %result:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.return %result#0, %result#1, %result#2 : memref<3xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func private @LyException_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0], ly.runtime.contract = "builtins.Exception", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @LyRuntimeError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0], ly.runtime.contract = "builtins.RuntimeError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @LyTypeError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0], ly.runtime.contract = "builtins.TypeError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @LyValueError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0], ly.runtime.contract = "builtins.ValueError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @LyArithmeticError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0], ly.runtime.contract = "builtins.ArithmeticError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @LyLookupError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0], ly.runtime.contract = "builtins.LookupError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @LyZeroDivisionError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0], ly.runtime.contract = "builtins.ZeroDivisionError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @LyKeyError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0], ly.runtime.contract = "builtins.KeyError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @LyIndexError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0], ly.runtime.contract = "builtins.IndexError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @LyAssertionError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0], ly.runtime.contract = "builtins.AssertionError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @LyStopIteration_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0], ly.runtime.contract = "builtins.StopIteration", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @LyStopAsyncIteration_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0], ly.runtime.contract = "builtins.StopAsyncIteration", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @LyCancelledError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0], ly.runtime.contract = "asyncio.CancelledError", ly.runtime.primitive = "raise"} {
    func.call @LyEH_ThrowException(%header, %message_header, %message_bytes) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

}
