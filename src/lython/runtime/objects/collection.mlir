// Runtime ABI for Python collection objects whose structural contracts are
// defined in typing.mlir. Payload slots are flattened 16-word boxed object
// handles: refcount, payload class id, payload header pointer, physical value
// pointer count, up to five physical value pointers, their one-dimensional
// sizes, an owned-payload flag, and one reserved word.
// The collection meta capacity counts logical slots; the payload arrays reserve
// sixteen i64 words for each logical slot.

module attributes {ly.runtime.contracts = ["builtins.list", "builtins.tuple", "builtins.dict"]} {
  func.func private @LyObject_ReleaseStorageToZero(%storage: memref<?xi64>) -> i1
  func.func private @LyObject_ReleaseBoxedPayloadArraySlotRaw(%payload: memref<?xi64>, %logical_index: i64)

  func.func private @LyList_Shape() -> (memref<2xi64>, memref<2xi64>, memref<?xi64>) attributes {ly.runtime.contract = "builtins.list", ly.runtime.shape}

  func.func private @LyTuple_Shape() -> (memref<2xi64>, memref<2xi64>, memref<?xi64>) attributes {ly.runtime.contract = "builtins.tuple", ly.runtime.shape}

  func.func private @LyDict_Shape() -> (memref<2xi64>, memref<2xi64>, memref<?xi64>, memref<?xi64>, memref<?xi64>) attributes {ly.runtime.contract = "builtins.dict", ly.runtime.shape}

  func.func private @__ly_sequence_alloc(%class_id: i64, %length: i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>) attributes {ly.ownership.owned_results = [0]} {
    %one = arith.constant 1 : i64
    %minimum_capacity = arith.constant 64 : i64
    %handle_words = arith.constant 16 : i64
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %length_slot = arith.constant 0 : index
    %capacity_slot = arith.constant 1 : index

    %header = memref.alloc() {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<2xi64>
    %meta = memref.alloc() : memref<2xi64>
    %needs_min_capacity = arith.cmpi slt, %length, %minimum_capacity : i64
    %capacity = arith.select %needs_min_capacity, %minimum_capacity, %length : i1, i64
    %payload_words = arith.muli %capacity, %handle_words : i64
    %payload_words_index = arith.index_cast %payload_words : i64 to index
    %items = memref.alloc(%payload_words_index) : memref<?xi64>

    memref.store %one, %header[%refcount_slot] : memref<2xi64>
    memref.store %class_id, %header[%layout_slot] : memref<2xi64>
    memref.store %length, %meta[%length_slot] : memref<2xi64>
    memref.store %capacity, %meta[%capacity_slot] : memref<2xi64>
    func.return %header, %meta, %items : memref<2xi64>, memref<2xi64>, memref<?xi64>
  }

  func.func private @__ly_dict_alloc(%length: i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>, memref<?xi64>, memref<?xi64>) attributes {ly.ownership.owned_results = [0]} {
    %one = arith.constant 1 : i64
    %minimum_capacity = arith.constant 64 : i64
    %handle_words = arith.constant 16 : i64
    %class_id = arith.constant 12 : i64
    %zero = arith.constant 0 : i64
    %lower = arith.constant 0 : index
    %step = arith.constant 1 : index
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %length_slot = arith.constant 0 : index
    %capacity_slot = arith.constant 1 : index

    %header = memref.alloc() {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<2xi64>
    %meta = memref.alloc() : memref<2xi64>
    %needs_min_capacity = arith.cmpi slt, %length, %minimum_capacity : i64
    %capacity = arith.select %needs_min_capacity, %minimum_capacity, %length : i1, i64
    %capacity_index = arith.index_cast %capacity : i64 to index
    %payload_words = arith.muli %capacity, %handle_words : i64
    %payload_words_index = arith.index_cast %payload_words : i64 to index
    %keys = memref.alloc(%payload_words_index) : memref<?xi64>
    %values = memref.alloc(%payload_words_index) : memref<?xi64>
    %present = memref.alloc(%capacity_index) : memref<?xi64>

    memref.store %one, %header[%refcount_slot] : memref<2xi64>
    memref.store %class_id, %header[%layout_slot] : memref<2xi64>
    memref.store %length, %meta[%length_slot] : memref<2xi64>
    memref.store %capacity, %meta[%capacity_slot] : memref<2xi64>
    scf.for %i = %lower to %capacity_index step %step {
      memref.store %zero, %present[%i] : memref<?xi64>
    }
    func.return %header, %meta, %keys, %values, %present : memref<2xi64>, memref<2xi64>, memref<?xi64>, memref<?xi64>, memref<?xi64>
  }

  func.func @LyList_FromLength(%length: i64 {ly.runtime.default_i64 = 0 : i64}) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 10 : i64, ly.runtime.contract = "builtins.list", ly.runtime.initializer = "__new__"} {
    %class_id = arith.constant 10 : i64
    %header, %meta, %items = func.call @__ly_sequence_alloc(%class_id, %length) : (i64, i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>)
    func.return %header, %meta, %items : memref<2xi64>, memref<2xi64>, memref<?xi64>
  }

  func.func @LyTuple_FromLength(%length: i64 {ly.runtime.default_i64 = 0 : i64}) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 11 : i64, ly.runtime.contract = "builtins.tuple", ly.runtime.initializer = "__new__"} {
    %class_id = arith.constant 11 : i64
    %header, %meta, %items = func.call @__ly_sequence_alloc(%class_id, %length) : (i64, i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>)
    func.return %header, %meta, %items : memref<2xi64>, memref<2xi64>, memref<?xi64>
  }

  func.func @LyDict_FromLength(%length: i64 {ly.runtime.default_i64 = 0 : i64}) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>, memref<?xi64>, memref<?xi64>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 12 : i64, ly.runtime.contract = "builtins.dict", ly.runtime.initializer = "__new__"} {
    %header, %meta, %keys, %values, %present = func.call @__ly_dict_alloc(%length) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>, memref<?xi64>, memref<?xi64>)
    func.return %header, %meta, %keys, %values, %present : memref<2xi64>, memref<2xi64>, memref<?xi64>, memref<?xi64>, memref<?xi64>
  }

  func.func @LyList_Len(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %items: memref<?xi64>) -> i64 attributes {ly.runtime.contract = "builtins.list", ly.runtime.method = "__len__"} {
    %length_slot = arith.constant 0 : index
    %length = memref.load %meta[%length_slot] : memref<2xi64>
    func.return %length : i64
  }

  func.func @LyTuple_Len(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %items: memref<?xi64>) -> i64 attributes {ly.runtime.contract = "builtins.tuple", ly.runtime.method = "__len__"} {
    %length_slot = arith.constant 0 : index
    %length = memref.load %meta[%length_slot] : memref<2xi64>
    func.return %length : i64
  }

  func.func @LyDict_Len(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %keys: memref<?xi64>, %values: memref<?xi64>, %present: memref<?xi64>) -> i64 attributes {ly.runtime.contract = "builtins.dict", ly.runtime.method = "__len__"} {
    %length_slot = arith.constant 0 : index
    %length = memref.load %meta[%length_slot] : memref<2xi64>
    func.return %length : i64
  }

  func.func private @__ly_sequence_ensure_capacity(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %items: memref<?xi64>, %required: i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>) {
    %minimum_capacity = arith.constant 64 : i64
    %handle_words = arith.constant 16 : i64
    %two = arith.constant 2 : i64
    %capacity_slot = arith.constant 1 : index
    %lower = arith.constant 0 : index
    %step = arith.constant 1 : index

    %capacity = memref.load %meta[%capacity_slot] : memref<2xi64>
    %needs_grow = arith.cmpi slt, %capacity, %required : i64
    %out_header, %out_meta, %out_items = scf.if %needs_grow -> (memref<2xi64>, memref<2xi64>, memref<?xi64>) {
      %doubled = arith.muli %capacity, %two : i64
      %below_min = arith.cmpi slt, %doubled, %minimum_capacity : i64
      %base_capacity = arith.select %below_min, %minimum_capacity, %doubled : i1, i64
      %below_required = arith.cmpi slt, %base_capacity, %required : i64
      %new_capacity = arith.select %below_required, %required, %base_capacity : i1, i64
      %old_words = arith.muli %capacity, %handle_words : i64
      %new_words = arith.muli %new_capacity, %handle_words : i64
      %old_words_index = arith.index_cast %old_words : i64 to index
      %new_words_index = arith.index_cast %new_words : i64 to index
      %new_items = memref.alloc(%new_words_index) : memref<?xi64>
      scf.for %i = %lower to %old_words_index step %step {
        %word = memref.load %items[%i] : memref<?xi64>
        memref.store %word, %new_items[%i] : memref<?xi64>
      }
      memref.dealloc %items {ly.ownership.object_dealloc_part = "items_realloc"} : memref<?xi64>
      memref.store %new_capacity, %meta[%capacity_slot] : memref<2xi64>
      scf.yield %header, %meta, %new_items : memref<2xi64>, memref<2xi64>, memref<?xi64>
    } else {
      scf.yield %header, %meta, %items : memref<2xi64>, memref<2xi64>, memref<?xi64>
    }
    func.return %out_header, %out_meta, %out_items : memref<2xi64>, memref<2xi64>, memref<?xi64>
  }

  func.func @LyList_EnsureCapacity(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %items: memref<?xi64>, %required: i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>) attributes {ly.ownership.transfer_args = [0], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.list", ly.runtime.primitive = "ensure_capacity"} {
    %out_header, %out_meta, %out_items = func.call @__ly_sequence_ensure_capacity(%header, %meta, %items, %required) : (memref<2xi64>, memref<2xi64>, memref<?xi64>, i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>)
    func.return %out_header, %out_meta, %out_items : memref<2xi64>, memref<2xi64>, memref<?xi64>
  }

  func.func @LyDict_EnsureCapacity(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %keys: memref<?xi64>, %values: memref<?xi64>, %present: memref<?xi64>, %required: i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi64>, memref<?xi64>, memref<?xi64>) attributes {ly.ownership.transfer_args = [0], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.dict", ly.runtime.primitive = "ensure_capacity"} {
    %minimum_capacity = arith.constant 64 : i64
    %handle_words = arith.constant 16 : i64
    %two = arith.constant 2 : i64
    %zero = arith.constant 0 : i64
    %capacity_slot = arith.constant 1 : index
    %lower = arith.constant 0 : index
    %step = arith.constant 1 : index

    %capacity = memref.load %meta[%capacity_slot] : memref<2xi64>
    %needs_grow = arith.cmpi slt, %capacity, %required : i64
    %out_header, %out_meta, %out_keys, %out_values, %out_present = scf.if %needs_grow -> (memref<2xi64>, memref<2xi64>, memref<?xi64>, memref<?xi64>, memref<?xi64>) {
      %doubled = arith.muli %capacity, %two : i64
      %below_min = arith.cmpi slt, %doubled, %minimum_capacity : i64
      %base_capacity = arith.select %below_min, %minimum_capacity, %doubled : i1, i64
      %below_required = arith.cmpi slt, %base_capacity, %required : i64
      %new_capacity = arith.select %below_required, %required, %base_capacity : i1, i64
      %old_words = arith.muli %capacity, %handle_words : i64
      %new_words = arith.muli %new_capacity, %handle_words : i64
      %old_words_index = arith.index_cast %old_words : i64 to index
      %new_words_index = arith.index_cast %new_words : i64 to index
      %old_capacity_index = arith.index_cast %capacity : i64 to index
      %new_capacity_index = arith.index_cast %new_capacity : i64 to index
      %new_keys = memref.alloc(%new_words_index) : memref<?xi64>
      %new_values = memref.alloc(%new_words_index) : memref<?xi64>
      %new_present = memref.alloc(%new_capacity_index) : memref<?xi64>
      scf.for %i = %lower to %new_capacity_index step %step {
        memref.store %zero, %new_present[%i] : memref<?xi64>
      }
      scf.for %i = %lower to %old_words_index step %step {
        %key_word = memref.load %keys[%i] : memref<?xi64>
        %value_word = memref.load %values[%i] : memref<?xi64>
        memref.store %key_word, %new_keys[%i] : memref<?xi64>
        memref.store %value_word, %new_values[%i] : memref<?xi64>
      }
      scf.for %i = %lower to %old_capacity_index step %step {
        %present_word = memref.load %present[%i] : memref<?xi64>
        memref.store %present_word, %new_present[%i] : memref<?xi64>
      }
      memref.dealloc %present {ly.ownership.object_dealloc_part = "present_realloc"} : memref<?xi64>
      memref.dealloc %values {ly.ownership.object_dealloc_part = "values_realloc"} : memref<?xi64>
      memref.dealloc %keys {ly.ownership.object_dealloc_part = "keys_realloc"} : memref<?xi64>
      memref.store %new_capacity, %meta[%capacity_slot] : memref<2xi64>
      scf.yield %header, %meta, %new_keys, %new_values, %new_present : memref<2xi64>, memref<2xi64>, memref<?xi64>, memref<?xi64>, memref<?xi64>
    } else {
      scf.yield %header, %meta, %keys, %values, %present : memref<2xi64>, memref<2xi64>, memref<?xi64>, memref<?xi64>, memref<?xi64>
    }
    func.return %out_header, %out_meta, %out_keys, %out_values, %out_present : memref<2xi64>, memref<2xi64>, memref<?xi64>, memref<?xi64>, memref<?xi64>
  }

  func.func @LyList_DecRef(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %items: memref<?xi64>) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.list", ly.runtime.deallocator} {
    %storage = memref.cast %header : memref<2xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    %length_slot = arith.constant 0 : index
    %lower = arith.constant 0 : index
    %step = arith.constant 1 : index
    %length = memref.load %meta[%length_slot] : memref<2xi64>
    %length_index = arith.index_cast %length : i64 to index
    scf.for %i = %lower to %length_index step %step {
      %logical_index = arith.index_cast %i : index to i64
      func.call @LyObject_ReleaseBoxedPayloadArraySlotRaw(%items, %logical_index) : (memref<?xi64>, i64) -> ()
    }
    memref.dealloc %items {ly.ownership.object_dealloc_part = "items"} : memref<?xi64>
    memref.dealloc %meta {ly.ownership.object_dealloc_part = "meta"} : memref<2xi64>
    memref.dealloc %header {ly.ownership.object_dealloc_part = "header"} : memref<2xi64>
    cf.br ^done

  ^done:
    func.return
  }

  func.func @LyTuple_DecRef(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %items: memref<?xi64>) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.tuple", ly.runtime.deallocator} {
    %storage = memref.cast %header : memref<2xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    %length_slot = arith.constant 0 : index
    %lower = arith.constant 0 : index
    %step = arith.constant 1 : index
    %length = memref.load %meta[%length_slot] : memref<2xi64>
    %length_index = arith.index_cast %length : i64 to index
    scf.for %i = %lower to %length_index step %step {
      %logical_index = arith.index_cast %i : index to i64
      func.call @LyObject_ReleaseBoxedPayloadArraySlotRaw(%items, %logical_index) : (memref<?xi64>, i64) -> ()
    }
    memref.dealloc %items {ly.ownership.object_dealloc_part = "items"} : memref<?xi64>
    memref.dealloc %meta {ly.ownership.object_dealloc_part = "meta"} : memref<2xi64>
    memref.dealloc %header {ly.ownership.object_dealloc_part = "header"} : memref<2xi64>
    cf.br ^done

  ^done:
    func.return
  }

  func.func @LyDict_DecRef(%header: memref<2xi64> {ly.ownership.object_header}, %meta: memref<2xi64>, %keys: memref<?xi64>, %values: memref<?xi64>, %present: memref<?xi64>) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.dict", ly.runtime.deallocator} {
    %storage = memref.cast %header : memref<2xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    %capacity_slot = arith.constant 1 : index
    %lower = arith.constant 0 : index
    %step = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %capacity = memref.load %meta[%capacity_slot] : memref<2xi64>
    %capacity_index = arith.index_cast %capacity : i64 to index
    scf.for %i = %lower to %capacity_index step %step {
      %occupied = memref.load %present[%i] : memref<?xi64>
      %is_present = arith.cmpi ne, %occupied, %zero : i64
      scf.if %is_present {
        %logical_index = arith.index_cast %i : index to i64
        func.call @LyObject_ReleaseBoxedPayloadArraySlotRaw(%keys, %logical_index) : (memref<?xi64>, i64) -> ()
        func.call @LyObject_ReleaseBoxedPayloadArraySlotRaw(%values, %logical_index) : (memref<?xi64>, i64) -> ()
      }
    }
    memref.dealloc %present {ly.ownership.object_dealloc_part = "present"} : memref<?xi64>
    memref.dealloc %values {ly.ownership.object_dealloc_part = "values"} : memref<?xi64>
    memref.dealloc %keys {ly.ownership.object_dealloc_part = "keys"} : memref<?xi64>
    memref.dealloc %meta {ly.ownership.object_dealloc_part = "meta"} : memref<2xi64>
    memref.dealloc %header {ly.ownership.object_dealloc_part = "header"} : memref<2xi64>
    cf.br ^done

  ^done:
    func.return
  }
}
