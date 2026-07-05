module attributes {ly.runtime.contracts = ["builtins.str", "builtins.str_iterator"]} {
  func.func private @Ly_IncRef(%header: memref<2xi64, strided<[1], offset: ?>> {ly.ownership.object_header})
  func.func private @LyObject_ReleaseStorageToZero(%storage: memref<?xi64>) -> i1
  func.func private @LyUnicode_DecRef(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>)
  func.func private @LyUnicode_CodepointLength(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>) -> i64
  func.func private @LyUnicode_GetItem(%header: memref<2xi64> {ly.ownership.object_header}, %bytes: memref<?xi8>, %raw_index: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0]}
  func.func private @LyUnicode_FromBytes(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0]}

  func.func private @__ly_str_iterator_alloc(%position: i64, %length: i64) -> (memref<2xi64>, memref<2xi64>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 7 : i64, ly.runtime.contract = "builtins.str_iterator", ly.runtime.primitive = "alloc"} {
    %header = memref.alloc() {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<2xi64>
    %state = memref.alloc() : memref<2xi64>
    %one = arith.constant 1 : i64
    %layout_str_iterator = arith.constant 7 : i64
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %position_slot = arith.constant 0 : index
    %length_slot = arith.constant 1 : index
    memref.store %one, %header[%refcount_slot] : memref<2xi64>
    memref.store %layout_str_iterator, %header[%layout_slot] : memref<2xi64>
    memref.store %position, %state[%position_slot] : memref<2xi64>
    memref.store %length, %state[%length_slot] : memref<2xi64>
    func.return %header, %state : memref<2xi64>, memref<2xi64>
  }

  func.func @LyUnicode_Iter(%source_header: memref<2xi64> {ly.ownership.object_header}, %source_bytes: memref<?xi8>) -> (memref<2xi64>, memref<2xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__iter__", ly.runtime.result_contract = "builtins.str_iterator"} {
    %zero = arith.constant 0 : i64
    %length = func.call @LyUnicode_CodepointLength(%source_header, %source_bytes) : (memref<2xi64>, memref<?xi8>) -> i64
    %iter_header, %state = func.call @__ly_str_iterator_alloc(%zero, %length) : (i64, i64) -> (memref<2xi64>, memref<2xi64>)
    %source_header_view = memref.cast %source_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%source_header_view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %iter_header, %state, %source_header, %source_bytes : memref<2xi64>, memref<2xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicodeStrIterator_Iter(%iter_header: memref<2xi64> {ly.ownership.object_header}, %state: memref<2xi64>, %source_header: memref<2xi64> {ly.ownership.object_header}, %source_bytes: memref<?xi8>) -> (memref<2xi64>, memref<2xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str_iterator", ly.runtime.method = "__iter__", ly.runtime.result_contract = "builtins.str_iterator"} {
    %iter_header_view = memref.cast %iter_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%iter_header_view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %iter_header, %state, %source_header, %source_bytes : memref<2xi64>, memref<2xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicodeStrIterator_Next(%iter_header: memref<2xi64> {ly.ownership.object_header}, %state: memref<2xi64>, %source_header: memref<2xi64> {ly.ownership.object_header}, %source_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>, i1, memref<2xi64>, memref<2xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0, 3], ly.runtime.contract = "builtins.str_iterator", ly.runtime.method = "__next__", ly.runtime.element_contract = "builtins.str", ly.runtime.next_contract = "builtins.str_iterator", ly.runtime.valid_result_index = 2 : i64} {
    %position_slot = arith.constant 0 : index
    %length_slot = arith.constant 1 : index
    %position = memref.load %state[%position_slot] : memref<2xi64>
    %length = memref.load %state[%length_slot] : memref<2xi64>
    %valid = arith.cmpi slt, %position, %length : i64
    %one = arith.constant 1 : i64
    %next_position_candidate = arith.addi %position, %one : i64
    %next_position = arith.select %valid, %next_position_candidate, %position : i1, i64
    memref.store %next_position, %state[%position_slot] : memref<2xi64>
    %iter_header_view = memref.cast %iter_header : memref<2xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%iter_header_view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()

    %element:2 = scf.if %valid -> (memref<2xi64>, memref<?xi8>) {
      %item_header, %item_bytes = func.call @LyUnicode_GetItem(%source_header, %source_bytes, %position) : (memref<2xi64>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %item_header, %item_bytes : memref<2xi64>, memref<?xi8>
    } else {
      %start = arith.constant 0 : index
      %zero = arith.constant 0 : i64
      %empty_header, %empty_bytes = func.call @LyUnicode_FromBytes(%source_bytes, %start, %zero) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %empty_header, %empty_bytes : memref<2xi64>, memref<?xi8>
    }

    func.return %element#0, %element#1, %valid, %iter_header, %state, %source_header, %source_bytes : memref<2xi64>, memref<?xi8>, i1, memref<2xi64>, memref<2xi64>, memref<2xi64>, memref<?xi8>
  }

  func.func @LyUnicodeStrIterator_DecRef(%iter_header: memref<2xi64> {ly.ownership.object_header}, %state: memref<2xi64>, %source_header: memref<2xi64> {ly.ownership.object_header}, %source_bytes: memref<?xi8>) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.str_iterator", ly.runtime.deallocator} {
    %storage = memref.cast %iter_header : memref<2xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    func.call @LyUnicode_DecRef(%source_header, %source_bytes) : (memref<2xi64>, memref<?xi8>) -> ()
    memref.dealloc %state {ly.ownership.object_dealloc_part = "state"} : memref<2xi64>
    memref.dealloc %iter_header {ly.ownership.object_dealloc_part = "header"} : memref<2xi64>
    cf.br ^done

  ^done:
    func.return
  }
}
