// `memoryview` and its iterator -- CPython's Objects/memoryobject.c.
//
// Signature source (1:1 correspondence target):
//   https://github.com/python/typeshed/blob/main/stdlib/builtins.pyi
//
// One dimension of unsigned bytes -- format 'B', itemsize 1 -- over a bytes,
// a bytearray, or another view of one. A view holds the address of its first
// item, its length and stride, a reference to the object, and for a bytearray
// one of its exports, which keeps the payload where the view reads it: a
// bytearray refuses to resize while exported, as CPython's does.
//
// Deviations from CPython:
// - Only format 'B' over bytes and bytearray: no cast(), no other formats, no
//   views of more than one dimension, no `obj`.
// - Each view, a slice included, holds an export of its own where CPython's
//   share one managed buffer; a resize is refused in exactly the same states.
// - __exit__ answers False where CPython's answers None.

module attributes {
  ly.typing.manifest,
  ly.runtime.contracts = ["builtins.memoryview", "builtins.memory_iterator"]
} {
  // ===== declared here, defined in another runtime file or built by the lowering =====
  func.func private @LyBytes_DecRef(%header: memref<4xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.deallocator}
  func.func private @LyBytes_EqBool(%lhs_header: memref<4xi64> {ly.ownership.object_header}, %rhs_header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__eq__"}
  func.func private @LyBytes_Hash(%header: memref<4xi64> {ly.ownership.object_header}) -> i64 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__hash__"}
  func.func private @LyBytes_Hex(%header: memref<4xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.method = "hex", ly.runtime.result_contract = "builtins.str"}
  func.func private @LyList_FromLength(%length: i64 {ly.runtime.default_i64 = 0 : i64}) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id, ly.runtime.contract = "builtins.list", ly.runtime.initializer = "__new__", ly.runtime.result_contract = "builtins.list"}
  func.func private @LyLong_FromI64(%value: i64 {ly.runtime.default_i64 = 0 : i64}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id, ly.runtime.contract = "builtins.int", ly.runtime.initializer = "__new__"}
  func.func private @LyLong_SlotWordFromI64(%value: i64) -> i64 attributes {ly.runtime.contract = "builtins.int", ly.runtime.primitive = "slot_word_from_i64"}
  func.func private @LyObject_ReleaseBoxedPayloadArraySlotRaw(%payload: memref<?xi64>, %logical_index: i64)
  func.func private @LyObject_ReleaseStorageToZero(%storage: memref<?xi64>) -> i1 attributes {ly.runtime.contract = "builtins.object", ly.runtime.primitive = "release_to_zero"}
  func.func private @Ly_IncRef(%header: memref<2xi64, strided<[1], offset: ?>> {ly.ownership.object_header}) attributes {ly.ownership.retain_args = [0], ly.runtime.primitive = "retain"}
  func.func private @__ly_box_store_entity(%items: memref<?xi64>, %slot: i64, %class_id: i64, %entity: i64)
  func.func private @__ly_bytearray_add_export(%self: memref<4xi64>, %delta: i64) attributes {ly.runtime.contract = "builtins.bytearray"}
  func.func private @__ly_bytes_alloc(%len: i64) -> memref<4xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.bytes"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.bytes", ly.runtime.primitive = "alloc"}
  func.func private @__ly_bytes_payload(%self: memref<4xi64>) -> memref<?xi8> attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.interior_word, ly.runtime.primitive = "payload_view"}
  func.func private @__ly_global_view_i64(%pointer: i64, %size: i64) -> memref<?xi64>
  func.func private @__ly_global_view_i8(%pointer: i64, %size: i64) -> memref<?xi8>
  func.func private @__ly_handle_retain_raw(%entity: i64)
  func.func private @__ly_list_items(%self: memref<5xi64>) -> memref<?xi64> attributes {ly.runtime.contract = "builtins.list", ly.runtime.interior_word, ly.runtime.primitive = "items_view"}
  func.func private @__ly_raise_static_message(%class_id: i64, %message: memref<?xi8>, %length: i64)
  func.func private @__ly_slice_adjust(%len: i64, %start_in: i64, %stop_in: i64, %step: i64, %mask: i64) -> (i64, i64)
  func.func private @__ly_slice_raise_zero_step()
  func.func private @__ly_slice_unpack(%self: memref<5xi64>) -> (i64, i64, i64, i64)
  func.func private @__ly_tuple_alloc(%length: i64) -> memref<5xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.tuple"], ly.ownership.owned_results = [0]}
  func.func private @__ly_tuple_items(%self: memref<5xi64>) -> memref<?xi64> attributes {ly.runtime.contract = "builtins.tuple", ly.runtime.interior_word, ly.runtime.primitive = "items_view"}
  func.func private @__ly_unicode_from_valid_utf8(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]}
  py.class @memoryview attributes {
    base_names = ["Sequence"],
    ly.typing.base_args = [[!py.contract<"builtins.int">]],
    ly.typing.final,
    ly.runtime.contract = "builtins.memoryview", ly.runtime.required,
    ly.runtime.required_deallocator,
    ly.runtime.required_initializers = ["__new__"],
    ly.runtime.required_methods = ["__len__"],
    field_names = ["readonly", "format", "itemsize", "nbytes", "ndim", "shape", "strides",
                   "contiguous", "c_contiguous", "f_contiguous"],
    field_contract_types = [!py.contract<"builtins.bool">, !py.contract<"builtins.str">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.tuple", [!py.contract<"builtins.int">]>, !py.contract<"builtins.tuple", [!py.contract<"builtins.int">]>, !py.contract<"builtins.bool">, !py.contract<"builtins.bool">, !py.contract<"builtins.bool">],
    method_names = ["__new__", "__new__", "__new__", "__init__", "__init__", "__init__",
                    "__len__", "__getitem__", "__getslice__", "__getslice__", "__setitem__",
                    "__setslice__", "__setslice__", "__setslice__", "__setslice__",
                    "__setslice__", "__setslice__", "__contains__", "__iter__", "__eq__",
                    "__ne__", "__hash__", "__repr__", "tobytes", "tolist", "hex", "toreadonly",
                    "release", "__enter__", "__exit__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.memoryview">>, !py.contract<"builtins.bytes">] -> [!py.self]>,
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.memoryview">>, !py.contract<"builtins.bytearray">] -> [!py.self]>,
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.memoryview">>, !py.contract<"builtins.memoryview">] -> [!py.self]>,
      !py.protocol<"Callable", [!py.contract<"builtins.memoryview">, !py.contract<"builtins.bytes">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.memoryview">, !py.contract<"builtins.bytearray">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.memoryview">, !py.contract<"builtins.memoryview">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.memoryview">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.memoryview">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.memoryview">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.memoryview">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.memoryview">, !py.contract<"builtins.slice">] -> [!py.contract<"builtins.memoryview">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.memoryview">, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.memoryview">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.bytes">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.memoryview">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.bytearray">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.memoryview">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.memoryview">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.memoryview">, !py.contract<"builtins.slice">, !py.contract<"builtins.bytes">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.memoryview">, !py.contract<"builtins.slice">, !py.contract<"builtins.bytearray">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.memoryview">, !py.contract<"builtins.slice">, !py.contract<"builtins.memoryview">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.memoryview">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.memoryview">] -> [!py.contract<"builtins.memory_iterator">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.memoryview">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.memoryview">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.memoryview">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.memoryview">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.memoryview">] -> [!py.contract<"builtins.bytes">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.memoryview">] -> [!py.contract<"builtins.list", [!py.contract<"builtins.int">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.memoryview">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.memoryview">] -> [!py.contract<"builtins.memoryview">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.memoryview">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.memoryview">] -> [!py.contract<"builtins.memoryview">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.memoryview">, !py.union<!py.type<!py.contract<"builtins.BaseException">>, !py.literal<None>>, !py.union<!py.contract<"builtins.BaseException">, !py.literal<None>>, !py.union<!py.contract<"types.TracebackType">, !py.literal<None>>] -> [!py.contract<"builtins.bool">]>
    ],
    method_kinds = ["classmethod", "classmethod", "classmethod", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance", "instance",
                    "instance"]
  } {}

  py.class @memory_iterator attributes {
    base_names = ["Iterator"],
    ly.typing.base_args = [[!py.contract<"builtins.int">]],
    method_names = ["__iter__", "__next__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"builtins.memory_iterator">] -> [!py.self]>,
      !py.protocol<"Callable", [!py.contract<"builtins.memory_iterator">] -> [!py.contract<"builtins.int">]>
    ],
    method_kinds = ["instance", "instance"]
  } {}

  // ===== the view (PyMemoryViewObject) =====
  //
  // Eight words: refcount, class 27, the address of item 0, the length in
  // items, the stride in bytes (negative for a reversed slice), the viewed
  // object (a reference of its own), flags (1 read-only, 2 released), and
  // whether the view holds one of a bytearray's exports. With an export held
  // the bytearray refuses to resize, so the address stays where the view
  // reads it; a bytes never moves.
  func.func private @LyMemoryView_Shape() -> memref<8xi64> attributes {ly.runtime.contract = "builtins.memoryview", ly.runtime.shape}
  func.func private @LyMemoryIterator_Shape() -> memref<4xi64> attributes {ly.runtime.contract = "builtins.memory_iterator", ly.runtime.shape}

  // "operation forbidden on released memoryview object"
  memref.global "private" constant @__ly_memoryview_msg_released : memref<49xi8> = dense<[111, 112, 101, 114, 97, 116, 105, 111, 110, 32, 102, 111, 114, 98, 105, 100, 100, 101, 110, 32, 111, 110, 32, 114, 101, 108, 101, 97, 115, 101, 100, 32, 109, 101, 109, 111, 114, 121, 118, 105, 101, 119, 32, 111, 98, 106, 101, 99, 116]>
  // "index out of bounds on dimension 1"
  memref.global "private" constant @__ly_memoryview_msg_index : memref<34xi8> = dense<[105, 110, 100, 101, 120, 32, 111, 117, 116, 32, 111, 102, 32, 98, 111, 117, 110, 100, 115, 32, 111, 110, 32, 100, 105, 109, 101, 110, 115, 105, 111, 110, 32, 49]>
  // "cannot modify read-only memory"
  memref.global "private" constant @__ly_memoryview_msg_readonly : memref<30xi8> = dense<[99, 97, 110, 110, 111, 116, 32, 109, 111, 100, 105, 102, 121, 32, 114, 101, 97, 100, 45, 111, 110, 108, 121, 32, 109, 101, 109, 111, 114, 121]>
  // "memoryview: invalid value for format 'B'"
  memref.global "private" constant @__ly_memoryview_msg_value : memref<40xi8> = dense<[109, 101, 109, 111, 114, 121, 118, 105, 101, 119, 58, 32, 105, 110, 118, 97, 108, 105, 100, 32, 118, 97, 108, 117, 101, 32, 102, 111, 114, 32, 102, 111, 114, 109, 97, 116, 32, 39, 66, 39]>
  // "memoryview assignment: lvalue and rvalue have different structures"
  memref.global "private" constant @__ly_memoryview_msg_structure : memref<66xi8> = dense<[109, 101, 109, 111, 114, 121, 118, 105, 101, 119, 32, 97, 115, 115, 105, 103, 110, 109, 101, 110, 116, 58, 32, 108, 118, 97, 108, 117, 101, 32, 97, 110, 100, 32, 114, 118, 97, 108, 117, 101, 32, 104, 97, 118, 101, 32, 100, 105, 102, 102, 101, 114, 101, 110, 116, 32, 115, 116, 114, 117, 99, 116, 117, 114, 101, 115]>
  // "cannot hash writable memoryview object"
  memref.global "private" constant @__ly_memoryview_msg_hash : memref<38xi8> = dense<[99, 97, 110, 110, 111, 116, 32, 104, 97, 115, 104, 32, 119, 114, 105, 116, 97, 98, 108, 101, 32, 109, 101, 109, 111, 114, 121, 118, 105, 101, 119, 32, 111, 98, 106, 101, 99, 116]>
  // "B"
  memref.global "private" constant @__ly_memoryview_format : memref<1xi8> = dense<[66]>
  // "<memory at 0x" / "<released memory at 0x"
  memref.global "private" constant @__ly_memoryview_repr_open : memref<13xi8> = dense<[60, 109, 101, 109, 111, 114, 121, 32, 97, 116, 32, 48, 120]>
  memref.global "private" constant @__ly_memoryview_repr_released : memref<22xi8> = dense<[60, 114, 101, 108, 101, 97, 115, 101, 100, 32, 109, 101, 109, 111, 114, 121, 32, 97, 116, 32, 48, 120]>

  func.func private @__ly_memoryview_raise(%class_id: i64, %message: memref<?xi8>, %length: i64) attributes {ly.runtime.contract = "builtins.memoryview"} {
    func.call @__ly_raise_static_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  // CHECK_RELEASED: ValueError on a released view.
  func.func private @__ly_memoryview_check(%self: memref<8xi64>) attributes {ly.runtime.contract = "builtins.memoryview"} {
    %zero = arith.constant 0 : i64
    %released_bit = arith.constant 2 : i64
    %flags_slot = arith.constant 6 : index
    %flags = memref.load %self[%flags_slot] : memref<8xi64>
    %bit = arith.andi %flags, %released_bit : i64
    %released = arith.cmpi ne, %bit, %zero : i64
    scf.if %released {
      %value_error = arith.constant {ly.class_id_of = "builtins.ValueError"} 53 : i64
      %length = arith.constant 49 : i64
      %static = memref.get_global @__ly_memoryview_msg_released : memref<49xi8>
      %message = memref.cast %static : memref<49xi8> to memref<?xi8>
      func.call @__ly_memoryview_raise(%value_error, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    }
    func.return
  }

  func.func private @__ly_memoryview_readonly(%self: memref<8xi64>) -> i1 attributes {ly.runtime.contract = "builtins.memoryview"} {
    %zero = arith.constant 0 : i64
    %readonly_bit = arith.constant 1 : i64
    %flags_slot = arith.constant 6 : index
    %flags = memref.load %self[%flags_slot] : memref<8xi64>
    %bit = arith.andi %flags, %readonly_bit : i64
    %readonly = arith.cmpi ne, %bit, %zero : i64
    func.return %readonly : i1
  }

  // The viewed bytearray's handle, from the object word.
  func.func private @__ly_memoryview_bytearray(%word: i64) -> memref<4xi64> attributes {ly.runtime.contract = "builtins.memoryview"} {
    %four = arith.constant 4 : i64
    %view = func.call @__ly_global_view_i64(%word, %four) : (i64, i64) -> memref<?xi64>
    %handle = memref.cast %view : memref<?xi64> to memref<4xi64>
    func.return %handle : memref<4xi64>
  }

  // A view of `length` items from `data`, `stride` bytes apart, holding a
  // reference to the object `source` addresses -- and one of its exports
  // when `exported`.
  func.func private @__ly_memoryview_alloc(%data: i64, %length: i64, %stride: i64, %source: i64, %readonly: i1, %exported: i1) -> memref<8xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.memoryview"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.memoryview"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %class_memoryview = arith.constant {ly.class_id_of = "builtins.memoryview"} 27 : i64
    %header_bytes = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %raw = memref.alloc(%header_bytes) {alignment = 16 : i64} : memref<?xi8>
    %self = memref.view %raw[%c0][] {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<?xi8> to memref<8xi64>
    %flags = arith.select %readonly, %one, %zero : i64
    %export_word = arith.select %exported, %one, %zero : i64
    %s0 = arith.constant 0 : index
    %s1 = arith.constant 1 : index
    %s2 = arith.constant 2 : index
    %s3 = arith.constant 3 : index
    %s4 = arith.constant 4 : index
    %s5 = arith.constant 5 : index
    %s6 = arith.constant 6 : index
    %s7 = arith.constant 7 : index
    memref.store %one, %self[%s0] : memref<8xi64>
    memref.store %class_memoryview, %self[%s1] : memref<8xi64>
    memref.store %data, %self[%s2] : memref<8xi64>
    memref.store %length, %self[%s3] : memref<8xi64>
    memref.store %stride, %self[%s4] : memref<8xi64>
    memref.store %source, %self[%s5] : memref<8xi64>
    memref.store %flags, %self[%s6] : memref<8xi64>
    memref.store %export_word, %self[%s7] : memref<8xi64>
    func.call @__ly_handle_retain_raw(%source) : (i64) -> ()
    scf.if %exported {
      %bytearray = func.call @__ly_memoryview_bytearray(%source) : (i64) -> memref<4xi64>
      func.call @__ly_bytearray_add_export(%bytearray, %one) : (memref<4xi64>, i64) -> ()
    }
    func.return %self : memref<8xi64>
  }

  // memory_release: the export given back and the object's reference
  // dropped, once; the view is released from then on.
  func.func private @__ly_memoryview_release(%self: memref<8xi64>) attributes {ly.runtime.contract = "builtins.memoryview"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %minus_one = arith.constant -1 : i64
    %released_bit = arith.constant 2 : i64
    %c0 = arith.constant 0 : i64
    %source_slot = arith.constant 5 : index
    %flags_slot = arith.constant 6 : index
    %export_slot = arith.constant 7 : index
    %flags = memref.load %self[%flags_slot] : memref<8xi64>
    %bit = arith.andi %flags, %released_bit : i64
    %live = arith.cmpi eq, %bit, %zero : i64
    scf.if %live {
      %source = memref.load %self[%source_slot] : memref<8xi64>
      %export = memref.load %self[%export_slot] : memref<8xi64>
      %holds_export = arith.cmpi ne, %export, %zero : i64
      scf.if %holds_export {
        %bytearray = func.call @__ly_memoryview_bytearray(%source) : (i64) -> memref<4xi64>
        func.call @__ly_bytearray_add_export(%bytearray, %minus_one) : (memref<4xi64>, i64) -> ()
      }
      %self_index = memref.extract_aligned_pointer_as_index %self : memref<8xi64> -> index
      %self_word = arith.index_cast %self_index : index to i64
      %source_word_addr = arith.constant 40 : i64
      %slot_addr = arith.addi %self_word, %source_word_addr : i64
      %slot = func.call @__ly_global_view_i64(%slot_addr, %one) : (i64, i64) -> memref<?xi64>
      func.call @LyObject_ReleaseBoxedPayloadArraySlotRaw(%slot, %c0) : (memref<?xi64>, i64) -> ()
      memref.store %zero, %self[%source_slot] : memref<8xi64>
      memref.store %zero, %self[%export_slot] : memref<8xi64>
      %released = arith.ori %flags, %released_bit : i64
      memref.store %released, %self[%flags_slot] : memref<8xi64>
    }
    func.return
  }

  // memoryview(b), memoryview(ba): one initializer, since bytes and bytearray
  // hand over the same four words; the class word tells them apart -- a
  // bytearray's view is writable and holds one of its exports.
  func.func @LyMemoryView_New(%source: memref<4xi64> {ly.ownership.object_header}) -> memref<8xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id, ly.runtime.contract = "builtins.memoryview", ly.runtime.initializer = "__new__"} {
    %one = arith.constant 1 : i64
    %bytearray_class = arith.constant {ly.class_id_of = "builtins.bytearray"} 26 : i64
    %true = arith.constant true
    %class_slot = arith.constant 1 : index
    %payload_slot = arith.constant 2 : index
    %length_slot = arith.constant 3 : index
    %class = memref.load %source[%class_slot] : memref<4xi64>
    %data = memref.load %source[%payload_slot] : memref<4xi64>
    %length = memref.load %source[%length_slot] : memref<4xi64>
    %writable = arith.cmpi eq, %class, %bytearray_class : i64
    %readonly = arith.xori %writable, %true : i1
    %source_index = memref.extract_aligned_pointer_as_index %source : memref<4xi64> -> index
    %source_word = arith.index_cast %source_index : index to i64
    %self = func.call @__ly_memoryview_alloc(%data, %length, %one, %source_word, %readonly, %writable) : (i64, i64, i64, i64, i1, i1) -> memref<8xi64>
    func.return %self : memref<8xi64>
  }

  func.func @LyMemoryView_Init(%self: memref<8xi64> {ly.ownership.object_header}, %source: memref<4xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "builtins.memoryview", ly.runtime.method = "__init__"} {
    func.return
  }

  // memoryview(view): a new view of the same memory, holding its own export.
  func.func @LyMemoryView_NewFromView(%source: memref<8xi64> {ly.ownership.object_header}) -> memref<8xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id, ly.runtime.contract = "builtins.memoryview", ly.runtime.initializer = "__new__"} {
    %zero = arith.constant 0 : i64
    %s2 = arith.constant 2 : index
    %s3 = arith.constant 3 : index
    %s4 = arith.constant 4 : index
    %s5 = arith.constant 5 : index
    %s7 = arith.constant 7 : index
    func.call @__ly_memoryview_check(%source) : (memref<8xi64>) -> ()
    %data = memref.load %source[%s2] : memref<8xi64>
    %length = memref.load %source[%s3] : memref<8xi64>
    %stride = memref.load %source[%s4] : memref<8xi64>
    %object = memref.load %source[%s5] : memref<8xi64>
    %export = memref.load %source[%s7] : memref<8xi64>
    %exported = arith.cmpi ne, %export, %zero : i64
    %readonly = func.call @__ly_memoryview_readonly(%source) : (memref<8xi64>) -> i1
    %self = func.call @__ly_memoryview_alloc(%data, %length, %stride, %object, %readonly, %exported) : (i64, i64, i64, i64, i1, i1) -> memref<8xi64>
    func.return %self : memref<8xi64>
  }

  func.func @LyMemoryView_InitFromView(%self: memref<8xi64> {ly.ownership.object_header}, %source: memref<8xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "builtins.memoryview", ly.runtime.method = "__init__"} {
    func.return
  }

  func.func @LyMemoryView_DecRef(%self: memref<8xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.memoryview", ly.runtime.deallocator} {
    %storage = memref.cast %self : memref<8xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    func.call @__ly_memoryview_release(%self) : (memref<8xi64>) -> ()
    memref.dealloc %self : memref<8xi64>
    cf.br ^done

  ^done:
    func.return
  }

  func.func @LyMemoryView_Release(%self: memref<8xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "builtins.memoryview", ly.runtime.method = "release"} {
    func.call @__ly_memoryview_release(%self) : (memref<8xi64>) -> ()
    func.return
  }

  func.func @LyMemoryView_Enter(%self: memref<8xi64> {ly.ownership.object_header}) -> memref<8xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.memoryview", ly.runtime.method = "__enter__", ly.runtime.result_contract = "builtins.memoryview"} {
    func.call @__ly_memoryview_check(%self) : (memref<8xi64>) -> ()
    %refcount_view = memref.subview %self[0] [2] [1] : memref<8xi64> to memref<2xi64, strided<[1]>>
    %retained = memref.cast %refcount_view : memref<2xi64, strided<[1]>> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%retained) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %self : memref<8xi64>
  }

  func.func @LyMemoryView_Exit(%self: memref<8xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.memoryview", ly.runtime.method = "__exit__", ly.runtime.result_contract = "builtins.bool"} {
    %false = arith.constant false
    func.call @__ly_memoryview_release(%self) : (memref<8xi64>) -> ()
    func.return %false : i1
  }

  // ===== the items (memory_item, memory_subscript, memory_ass_sub) =====

  // The address of item `index`, from the end when negative; IndexError
  // "index out of bounds on dimension 1" past either end.
  func.func private @__ly_memoryview_item_address(%self: memref<8xi64>, %raw_index: i64) -> i64 attributes {ly.runtime.contract = "builtins.memoryview"} {
    %zero = arith.constant 0 : i64
    %s2 = arith.constant 2 : index
    %s3 = arith.constant 3 : index
    %s4 = arith.constant 4 : index
    %data = memref.load %self[%s2] : memref<8xi64>
    %length = memref.load %self[%s3] : memref<8xi64>
    %stride = memref.load %self[%s4] : memref<8xi64>
    %negative = arith.cmpi slt, %raw_index, %zero : i64
    %from_end = arith.addi %raw_index, %length : i64
    %index = arith.select %negative, %from_end, %raw_index : i64
    %low = arith.cmpi sge, %index, %zero : i64
    %high = arith.cmpi slt, %index, %length : i64
    %valid = arith.andi %low, %high : i1
    scf.if %valid {
    } else {
      %index_error = arith.constant {ly.class_id_of = "builtins.IndexError"} 55 : i64
      %message_length = arith.constant 34 : i64
      %static = memref.get_global @__ly_memoryview_msg_index : memref<34xi8>
      %message = memref.cast %static : memref<34xi8> to memref<?xi8>
      func.call @__ly_memoryview_raise(%index_error, %message, %message_length) : (i64, memref<?xi8>, i64) -> ()
    }
    %offset = arith.muli %index, %stride : i64
    %address = arith.addi %data, %offset : i64
    func.return %address : i64
  }

  func.func private @__ly_memoryview_load(%address: i64) -> i64 attributes {ly.runtime.contract = "builtins.memoryview"} {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1 : i64
    %view = func.call @__ly_global_view_i8(%address, %one) : (i64, i64) -> memref<?xi8>
    %byte = memref.load %view[%c0] : memref<?xi8>
    %wide = arith.extui %byte : i8 to i64
    func.return %wide : i64
  }

  func.func private @__ly_memoryview_store(%address: i64, %value: i64) attributes {ly.runtime.contract = "builtins.memoryview"} {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1 : i64
    %view = func.call @__ly_global_view_i8(%address, %one) : (i64, i64) -> memref<?xi8>
    %byte = arith.trunci %value : i64 to i8
    memref.store %byte, %view[%c0] : memref<?xi8>
    func.return
  }

  func.func @LyMemoryView_Len(%self: memref<8xi64> {ly.ownership.object_header}) -> i64 attributes {ly.runtime.contract = "builtins.memoryview", ly.runtime.method = "__len__"} {
    %s3 = arith.constant 3 : index
    func.call @__ly_memoryview_check(%self) : (memref<8xi64>) -> ()
    %length = memref.load %self[%s3] : memref<8xi64>
    func.return %length : i64
  }

  func.func @LyMemoryView_GetItem(%self: memref<8xi64> {ly.ownership.object_header}, %raw_index: i64) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.memoryview", ly.runtime.method = "__getitem__", ly.runtime.result_contract = "builtins.int"} {
    func.call @__ly_memoryview_check(%self) : (memref<8xi64>) -> ()
    %address = func.call @__ly_memoryview_item_address(%self, %raw_index) : (memref<8xi64>, i64) -> i64
    %value = func.call @__ly_memoryview_load(%address) : (i64) -> i64
    %result = func.call @LyLong_FromI64(%value) : (i64) -> memref<2xi64>
    func.return %result : memref<2xi64>
  }

  // A slice is a view of the same memory: its first item, `step` items
  // apart.
  func.func @LyMemoryView_GetSlice(%self: memref<8xi64> {ly.ownership.object_header}, %start_raw: i64 {ly.runtime.clip_i64}, %stop_raw: i64 {ly.runtime.clip_i64}, %step_raw: i64 {ly.runtime.clip_i64}, %mask: i64) -> memref<8xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.memoryview"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.memoryview", ly.runtime.method = "__getslice__"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %s2 = arith.constant 2 : index
    %s3 = arith.constant 3 : index
    %s4 = arith.constant 4 : index
    %s5 = arith.constant 5 : index
    %s7 = arith.constant 7 : index
    func.call @__ly_memoryview_check(%self) : (memref<8xi64>) -> ()
    %step_zero = arith.cmpi eq, %step_raw, %zero : i64
    scf.if %step_zero {
      func.call @__ly_slice_raise_zero_step() : () -> ()
    }
    %step = arith.select %step_zero, %one, %step_raw : i64
    %data = memref.load %self[%s2] : memref<8xi64>
    %length = memref.load %self[%s3] : memref<8xi64>
    %stride = memref.load %self[%s4] : memref<8xi64>
    %object = memref.load %self[%s5] : memref<8xi64>
    %export = memref.load %self[%s7] : memref<8xi64>
    %start, %count = func.call @__ly_slice_adjust(%length, %start_raw, %stop_raw, %step, %mask) : (i64, i64, i64, i64, i64) -> (i64, i64)
    %offset = arith.muli %start, %stride : i64
    %first = arith.addi %data, %offset : i64
    %new_stride = arith.muli %stride, %step : i64
    %readonly = func.call @__ly_memoryview_readonly(%self) : (memref<8xi64>) -> i1
    %exported = arith.cmpi ne, %export, %zero : i64
    %view = func.call @__ly_memoryview_alloc(%first, %count, %new_stride, %object, %readonly, %exported) : (i64, i64, i64, i64, i1, i1) -> memref<8xi64>
    func.return %view : memref<8xi64>
  }

  func.func @LyMemoryView_SliceSubscript(%self: memref<8xi64> {ly.ownership.object_header}, %slice: memref<5xi64> {ly.ownership.object_header}) -> memref<8xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.memoryview"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.memoryview", ly.runtime.method = "__getslice__"} {
    func.call @__ly_memoryview_check(%self) : (memref<8xi64>) -> ()
    %start, %stop, %step, %mask = func.call @__ly_slice_unpack(%slice) : (memref<5xi64>) -> (i64, i64, i64, i64)
    %view = func.call @LyMemoryView_GetSlice(%self, %start, %stop, %step, %mask) : (memref<8xi64>, i64, i64, i64, i64) -> memref<8xi64>
    func.return %view : memref<8xi64>
  }

  // TypeError "cannot modify read-only memory".
  func.func private @__ly_memoryview_check_writable(%self: memref<8xi64>) attributes {ly.runtime.contract = "builtins.memoryview"} {
    %readonly = func.call @__ly_memoryview_readonly(%self) : (memref<8xi64>) -> i1
    scf.if %readonly {
      %type_error = arith.constant {ly.class_id_of = "builtins.TypeError"} 52 : i64
      %length = arith.constant 30 : i64
      %static = memref.get_global @__ly_memoryview_msg_readonly : memref<30xi8>
      %message = memref.cast %static : memref<30xi8> to memref<?xi8>
      func.call @__ly_memoryview_raise(%type_error, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    }
    func.return
  }

  // m[i] = v: released, then read-only, then the index, then the value.
  func.func @LyMemoryView_SetItem(%self: memref<8xi64> {ly.ownership.object_header}, %raw_index: i64, %value: i64 {ly.runtime.clip_i64}) attributes {ly.runtime.contract = "builtins.memoryview", ly.runtime.method = "__setitem__"} {
    %zero = arith.constant 0 : i64
    %end = arith.constant 256 : i64
    func.call @__ly_memoryview_check(%self) : (memref<8xi64>) -> ()
    func.call @__ly_memoryview_check_writable(%self) : (memref<8xi64>) -> ()
    %address = func.call @__ly_memoryview_item_address(%self, %raw_index) : (memref<8xi64>, i64) -> i64
    %low = arith.cmpi sge, %value, %zero : i64
    %high = arith.cmpi slt, %value, %end : i64
    %fits = arith.andi %low, %high : i1
    scf.if %fits {
    } else {
      %value_error = arith.constant {ly.class_id_of = "builtins.ValueError"} 53 : i64
      %length = arith.constant 40 : i64
      %static = memref.get_global @__ly_memoryview_msg_value : memref<40xi8>
      %message = memref.cast %static : memref<40xi8> to memref<?xi8>
      func.call @__ly_memoryview_raise(%value_error, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    }
    func.call @__ly_memoryview_store(%address, %value) : (i64, i64) -> ()
    func.return
  }

  // The `count` items of a slice of `self` written from `source`, `count`
  // bytes read first so a source in the same memory reads what was there.
  func.func private @__ly_memoryview_write(%self: memref<8xi64>, %start_raw: i64, %stop_raw: i64, %step_raw: i64, %mask: i64, %source: memref<?xi8>, %given: i64) attributes {ly.runtime.contract = "builtins.memoryview"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %s2 = arith.constant 2 : index
    %s3 = arith.constant 3 : index
    %s4 = arith.constant 4 : index
    %step_zero = arith.cmpi eq, %step_raw, %zero : i64
    scf.if %step_zero {
      func.call @__ly_slice_raise_zero_step() : () -> ()
    }
    %step = arith.select %step_zero, %one, %step_raw : i64
    %data = memref.load %self[%s2] : memref<8xi64>
    %length = memref.load %self[%s3] : memref<8xi64>
    %stride = memref.load %self[%s4] : memref<8xi64>
    %start, %count = func.call @__ly_slice_adjust(%length, %start_raw, %stop_raw, %step, %mask) : (i64, i64, i64, i64, i64) -> (i64, i64)
    %mismatch = arith.cmpi ne, %count, %given : i64
    scf.if %mismatch {
      %value_error = arith.constant {ly.class_id_of = "builtins.ValueError"} 53 : i64
      %message_length = arith.constant 66 : i64
      %static = memref.get_global @__ly_memoryview_msg_structure : memref<66xi8>
      %message = memref.cast %static : memref<66xi8> to memref<?xi8>
      func.call @__ly_memoryview_raise(%value_error, %message, %message_length) : (i64, memref<?xi8>, i64) -> ()
    }
    %copy = func.call @__ly_bytes_alloc(%count) : (i64) -> memref<4xi64>
    %staged = func.call @__ly_bytes_payload(%copy) : (memref<4xi64>) -> memref<?xi8>
    %n = arith.index_cast %count : i64 to index
    scf.for %k = %c0 to %n step %c1 {
      %byte = memref.load %source[%k] : memref<?xi8>
      memref.store %byte, %staged[%k] : memref<?xi8>
    }
    %item_stride = arith.muli %stride, %step : i64
    %start_offset = arith.muli %start, %stride : i64
    %first = arith.addi %data, %start_offset : i64
    scf.for %k = %c0 to %n step %c1 {
      %k64 = arith.index_cast %k : index to i64
      %offset = arith.muli %k64, %item_stride : i64
      %address = arith.addi %first, %offset : i64
      %byte = memref.load %staged[%k] : memref<?xi8>
      %wide = arith.extui %byte : i8 to i64
      func.call @__ly_memoryview_store(%address, %wide) : (i64, i64) -> ()
    }
    func.call @LyBytes_DecRef(%copy) : (memref<4xi64>) -> ()
    func.return
  }

  // m[i:j:k] = b for bytes or a bytearray: "memoryview assignment: lvalue and
  // rvalue have different structures" unless the lengths agree.
  func.func @LyMemoryView_SetSlice(%self: memref<8xi64> {ly.ownership.object_header}, %start_raw: i64 {ly.runtime.clip_i64}, %stop_raw: i64 {ly.runtime.clip_i64}, %step_raw: i64 {ly.runtime.clip_i64}, %mask: i64, %value: memref<4xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "builtins.memoryview", ly.runtime.method = "__setslice__"} {
    %length_slot = arith.constant 3 : index
    func.call @__ly_memoryview_check(%self) : (memref<8xi64>) -> ()
    func.call @__ly_memoryview_check_writable(%self) : (memref<8xi64>) -> ()
    %given = memref.load %value[%length_slot] : memref<4xi64>
    %source = func.call @__ly_bytes_payload(%value) : (memref<4xi64>) -> memref<?xi8>
    func.call @__ly_memoryview_write(%self, %start_raw, %stop_raw, %step_raw, %mask, %source, %given) : (memref<8xi64>, i64, i64, i64, i64, memref<?xi8>, i64) -> ()
    func.return
  }

  // m[i:j:k] = other_view: the other view's items, read in order first.
  func.func @LyMemoryView_SetSliceView(%self: memref<8xi64> {ly.ownership.object_header}, %start_raw: i64 {ly.runtime.clip_i64}, %stop_raw: i64 {ly.runtime.clip_i64}, %step_raw: i64 {ly.runtime.clip_i64}, %mask: i64, %value: memref<8xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "builtins.memoryview", ly.runtime.method = "__setslice__"} {
    func.call @__ly_memoryview_check(%self) : (memref<8xi64>) -> ()
    func.call @__ly_memoryview_check_writable(%self) : (memref<8xi64>) -> ()
    %bytes = func.call @LyMemoryView_ToBytes(%value) : (memref<8xi64>) -> memref<4xi64>
    %length_slot = arith.constant 3 : index
    %given = memref.load %bytes[%length_slot] : memref<4xi64>
    %source = func.call @__ly_bytes_payload(%bytes) : (memref<4xi64>) -> memref<?xi8>
    func.call @__ly_memoryview_write(%self, %start_raw, %stop_raw, %step_raw, %mask, %source, %given) : (memref<8xi64>, i64, i64, i64, i64, memref<?xi8>, i64) -> ()
    func.call @LyBytes_DecRef(%bytes) : (memref<4xi64>) -> ()
    func.return
  }

  func.func @LyMemoryView_SliceAssign(%self: memref<8xi64> {ly.ownership.object_header}, %slice: memref<5xi64> {ly.ownership.object_header}, %value: memref<4xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "builtins.memoryview", ly.runtime.method = "__setslice__"} {
    func.call @__ly_memoryview_check(%self) : (memref<8xi64>) -> ()
    %start, %stop, %step, %mask = func.call @__ly_slice_unpack(%slice) : (memref<5xi64>) -> (i64, i64, i64, i64)
    func.call @LyMemoryView_SetSlice(%self, %start, %stop, %step, %mask, %value) : (memref<8xi64>, i64, i64, i64, i64, memref<4xi64>) -> ()
    func.return
  }

  func.func @LyMemoryView_SliceAssignView(%self: memref<8xi64> {ly.ownership.object_header}, %slice: memref<5xi64> {ly.ownership.object_header}, %value: memref<8xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "builtins.memoryview", ly.runtime.method = "__setslice__"} {
    func.call @__ly_memoryview_check(%self) : (memref<8xi64>) -> ()
    %start, %stop, %step, %mask = func.call @__ly_slice_unpack(%slice) : (memref<5xi64>) -> (i64, i64, i64, i64)
    func.call @LyMemoryView_SetSliceView(%self, %start, %stop, %step, %mask, %value) : (memref<8xi64>, i64, i64, i64, i64, memref<8xi64>) -> ()
    func.return
  }

  // v in m: the items compared in order, as a sequence without __contains__
  // is searched -- a value no byte can equal is simply not found.
  func.func @LyMemoryView_ContainsInt(%self: memref<8xi64> {ly.ownership.object_header}, %value: i64 {ly.runtime.clip_i64}) -> i1 attributes {ly.runtime.contract = "builtins.memoryview", ly.runtime.method = "__contains__"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %false = arith.constant false
    %s2 = arith.constant 2 : index
    %s3 = arith.constant 3 : index
    %s4 = arith.constant 4 : index
    func.call @__ly_memoryview_check(%self) : (memref<8xi64>) -> ()
    %data = memref.load %self[%s2] : memref<8xi64>
    %length = memref.load %self[%s3] : memref<8xi64>
    %stride = memref.load %self[%s4] : memref<8xi64>
    %n = arith.index_cast %length : i64 to index
    %found = scf.for %k = %c0 to %n step %c1 iter_args(%seen = %false) -> (i1) {
      %k64 = arith.index_cast %k : index to i64
      %offset = arith.muli %k64, %stride : i64
      %address = arith.addi %data, %offset : i64
      %item = func.call @__ly_memoryview_load(%address) : (i64) -> i64
      %match = arith.cmpi eq, %item, %value : i64
      %either = arith.ori %seen, %match : i1
      scf.yield %either : i1
    }
    func.return %found : i1
  }

  // ===== conversions (memory_tobytes, memory_tolist, memory_hex) =====

  func.func @LyMemoryView_ToBytes(%self: memref<8xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.bytes"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.memoryview", ly.runtime.method = "tobytes"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %s2 = arith.constant 2 : index
    %s3 = arith.constant 3 : index
    %s4 = arith.constant 4 : index
    func.call @__ly_memoryview_check(%self) : (memref<8xi64>) -> ()
    %data = memref.load %self[%s2] : memref<8xi64>
    %length = memref.load %self[%s3] : memref<8xi64>
    %stride = memref.load %self[%s4] : memref<8xi64>
    %result = func.call @__ly_bytes_alloc(%length) : (i64) -> memref<4xi64>
    %payload = func.call @__ly_bytes_payload(%result) : (memref<4xi64>) -> memref<?xi8>
    %n = arith.index_cast %length : i64 to index
    scf.for %k = %c0 to %n step %c1 {
      %k64 = arith.index_cast %k : index to i64
      %offset = arith.muli %k64, %stride : i64
      %address = arith.addi %data, %offset : i64
      %item = func.call @__ly_memoryview_load(%address) : (i64) -> i64
      %byte = arith.trunci %item : i64 to i8
      memref.store %byte, %payload[%k] : memref<?xi8>
    }
    func.return %result : memref<4xi64>
  }

  func.func @LyMemoryView_ToList(%self: memref<8xi64> {ly.ownership.object_header}) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.memoryview", ly.runtime.method = "tolist", ly.runtime.result_contract = "builtins.list", ly.runtime.element_contract = "builtins.int"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %int_class = arith.constant {ly.class_id_of = "builtins.int"} 1 : i64
    %s2 = arith.constant 2 : index
    %s3 = arith.constant 3 : index
    %s4 = arith.constant 4 : index
    func.call @__ly_memoryview_check(%self) : (memref<8xi64>) -> ()
    %data = memref.load %self[%s2] : memref<8xi64>
    %length = memref.load %self[%s3] : memref<8xi64>
    %stride = memref.load %self[%s4] : memref<8xi64>
    %list = func.call @LyList_FromLength(%length) : (i64) -> memref<5xi64>
    %items = func.call @__ly_list_items(%list) : (memref<5xi64>) -> memref<?xi64>
    %n = arith.index_cast %length : i64 to index
    scf.for %k = %c0 to %n step %c1 {
      %k64 = arith.index_cast %k : index to i64
      %offset = arith.muli %k64, %stride : i64
      %address = arith.addi %data, %offset : i64
      %item = func.call @__ly_memoryview_load(%address) : (i64) -> i64
      %word = func.call @LyLong_SlotWordFromI64(%item) : (i64) -> i64
      func.call @__ly_box_store_entity(%items, %k64, %int_class, %word) : (memref<?xi64>, i64, i64, i64) -> ()
    }
    func.return %list : memref<5xi64>
  }

  func.func @LyMemoryView_Hex(%self: memref<8xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.memoryview", ly.runtime.method = "hex", ly.runtime.result_contract = "builtins.str"} {
    %bytes = func.call @LyMemoryView_ToBytes(%self) : (memref<8xi64>) -> memref<4xi64>
    %h, %b = func.call @LyBytes_Hex(%bytes) : (memref<4xi64>) -> (memref<2xi64>, memref<?xi8>)
    func.call @LyBytes_DecRef(%bytes) : (memref<4xi64>) -> ()
    func.return %h, %b : memref<2xi64>, memref<?xi8>
  }

  func.func @LyMemoryView_ToReadOnly(%self: memref<8xi64> {ly.ownership.object_header}) -> memref<8xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.memoryview"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.memoryview", ly.runtime.method = "toreadonly"} {
    %zero = arith.constant 0 : i64
    %true = arith.constant true
    %s2 = arith.constant 2 : index
    %s3 = arith.constant 3 : index
    %s4 = arith.constant 4 : index
    %s5 = arith.constant 5 : index
    %s7 = arith.constant 7 : index
    func.call @__ly_memoryview_check(%self) : (memref<8xi64>) -> ()
    %data = memref.load %self[%s2] : memref<8xi64>
    %length = memref.load %self[%s3] : memref<8xi64>
    %stride = memref.load %self[%s4] : memref<8xi64>
    %object = memref.load %self[%s5] : memref<8xi64>
    %export = memref.load %self[%s7] : memref<8xi64>
    %exported = arith.cmpi ne, %export, %zero : i64
    %view = func.call @__ly_memoryview_alloc(%data, %length, %stride, %object, %true, %exported) : (i64, i64, i64, i64, i1, i1) -> memref<8xi64>
    func.return %view : memref<8xi64>
  }

  // ===== comparison and hashing (memory_richcompare, memory_hash) =====

  // The items of two views, or a view and a bytes-shaped handle, compared
  // through their bytes. A released view equals only itself.
  func.func private @__ly_memoryview_equal_bytes(%self: memref<8xi64>, %other: memref<4xi64>) -> i1 attributes {ly.runtime.contract = "builtins.memoryview"} {
    %zero = arith.constant 0 : i64
    %released_bit = arith.constant 2 : i64
    %false = arith.constant false
    %flags_slot = arith.constant 6 : index
    %flags = memref.load %self[%flags_slot] : memref<8xi64>
    %bit = arith.andi %flags, %released_bit : i64
    %released = arith.cmpi ne, %bit, %zero : i64
    %equal = scf.if %released -> (i1) {
      scf.yield %false : i1
    } else {
      %bytes = func.call @LyMemoryView_ToBytes(%self) : (memref<8xi64>) -> memref<4xi64>
      %same = func.call @LyBytes_EqBool(%bytes, %other) : (memref<4xi64>, memref<4xi64>) -> i1
      func.call @LyBytes_DecRef(%bytes) : (memref<4xi64>) -> ()
      scf.yield %same : i1
    }
    func.return %equal : i1
  }

  func.func private @__ly_memoryview_equal_view(%self: memref<8xi64>, %other: memref<8xi64>) -> i1 attributes {ly.runtime.contract = "builtins.memoryview"} {
    %zero = arith.constant 0 : i64
    %released_bit = arith.constant 2 : i64
    %flags_slot = arith.constant 6 : index
    %self_index = memref.extract_aligned_pointer_as_index %self : memref<8xi64> -> index
    %other_index = memref.extract_aligned_pointer_as_index %other : memref<8xi64> -> index
    %identical = arith.cmpi eq, %self_index, %other_index : index
    %flags = memref.load %other[%flags_slot] : memref<8xi64>
    %bit = arith.andi %flags, %released_bit : i64
    %other_released = arith.cmpi ne, %bit, %zero : i64
    %equal = scf.if %identical -> (i1) {
      %true = arith.constant true
      scf.yield %true : i1
    } else {
      %result = scf.if %other_released -> (i1) {
        %false = arith.constant false
        scf.yield %false : i1
      } else {
        %bytes = func.call @LyMemoryView_ToBytes(%other) : (memref<8xi64>) -> memref<4xi64>
        %same = func.call @__ly_memoryview_equal_bytes(%self, %bytes) : (memref<8xi64>, memref<4xi64>) -> i1
        func.call @LyBytes_DecRef(%bytes) : (memref<4xi64>) -> ()
        scf.yield %same : i1
      }
      scf.yield %result : i1
    }
    func.return %equal : i1
  }

  func.func @LyMemoryView_EqBytes(%self: memref<8xi64> {ly.ownership.object_header}, %other: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.memoryview", ly.runtime.method = "__eq__"} {
    %equal = func.call @__ly_memoryview_equal_bytes(%self, %other) : (memref<8xi64>, memref<4xi64>) -> i1
    func.return %equal : i1
  }

  func.func @LyMemoryView_EqView(%self: memref<8xi64> {ly.ownership.object_header}, %other: memref<8xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.memoryview", ly.runtime.method = "__eq__"} {
    %equal = func.call @__ly_memoryview_equal_view(%self, %other) : (memref<8xi64>, memref<8xi64>) -> i1
    func.return %equal : i1
  }

  func.func @LyMemoryView_NeBytes(%self: memref<8xi64> {ly.ownership.object_header}, %other: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.memoryview", ly.runtime.method = "__ne__"} {
    %true = arith.constant true
    %equal = func.call @__ly_memoryview_equal_bytes(%self, %other) : (memref<8xi64>, memref<4xi64>) -> i1
    %differ = arith.xori %equal, %true : i1
    func.return %differ : i1
  }

  func.func @LyMemoryView_NeView(%self: memref<8xi64> {ly.ownership.object_header}, %other: memref<8xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.memoryview", ly.runtime.method = "__ne__"} {
    %true = arith.constant true
    %equal = func.call @__ly_memoryview_equal_view(%self, %other) : (memref<8xi64>, memref<8xi64>) -> i1
    %differ = arith.xori %equal, %true : i1
    func.return %differ : i1
  }

  // b == m and ba == m, where bytes' and bytearray's own comparisons answer
  // NotImplemented and CPython asks the view.
  func.func @LyBytes_EqMemoryView(%self: memref<4xi64> {ly.ownership.object_header}, %other: memref<8xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__eq__"} {
    %equal = func.call @__ly_memoryview_equal_bytes(%other, %self) : (memref<8xi64>, memref<4xi64>) -> i1
    func.return %equal : i1
  }

  func.func @LyBytes_NeMemoryView(%self: memref<4xi64> {ly.ownership.object_header}, %other: memref<8xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytes", ly.runtime.method = "__ne__"} {
    %true = arith.constant true
    %equal = func.call @__ly_memoryview_equal_bytes(%other, %self) : (memref<8xi64>, memref<4xi64>) -> i1
    %differ = arith.xori %equal, %true : i1
    func.return %differ : i1
  }

  func.func @LyByteArray_EqMemoryView(%self: memref<4xi64> {ly.ownership.object_header}, %other: memref<8xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "__eq__"} {
    %equal = func.call @__ly_memoryview_equal_bytes(%other, %self) : (memref<8xi64>, memref<4xi64>) -> i1
    func.return %equal : i1
  }

  func.func @LyByteArray_NeMemoryView(%self: memref<4xi64> {ly.ownership.object_header}, %other: memref<8xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.bytearray", ly.runtime.method = "__ne__"} {
    %true = arith.constant true
    %equal = func.call @__ly_memoryview_equal_bytes(%other, %self) : (memref<8xi64>, memref<4xi64>) -> i1
    %differ = arith.xori %equal, %true : i1
    func.return %differ : i1
  }

  // hash(m): a read-only view hashes as the bytes it shows; a writable one is
  // "cannot hash writable memoryview object".
  func.func @LyMemoryView_Hash(%self: memref<8xi64> {ly.ownership.object_header}) -> i64 attributes {ly.runtime.contract = "builtins.memoryview", ly.runtime.method = "__hash__"} {
    %true = arith.constant true
    func.call @__ly_memoryview_check(%self) : (memref<8xi64>) -> ()
    %readonly = func.call @__ly_memoryview_readonly(%self) : (memref<8xi64>) -> i1
    %writable = arith.xori %readonly, %true : i1
    scf.if %writable {
      %value_error = arith.constant {ly.class_id_of = "builtins.ValueError"} 53 : i64
      %length = arith.constant 38 : i64
      %static = memref.get_global @__ly_memoryview_msg_hash : memref<38xi8>
      %message = memref.cast %static : memref<38xi8> to memref<?xi8>
      func.call @__ly_memoryview_raise(%value_error, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    }
    %bytes = func.call @LyMemoryView_ToBytes(%self) : (memref<8xi64>) -> memref<4xi64>
    %hash = func.call @LyBytes_Hash(%bytes) : (memref<4xi64>) -> i64
    func.call @LyBytes_DecRef(%bytes) : (memref<4xi64>) -> ()
    func.return %hash : i64
  }

  // repr(m): "<memory at 0x...>", "<released memory at 0x...>" once released.
  func.func @LyMemoryView_Repr(%self: memref<8xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.memoryview", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %four = arith.constant 4 : i64
    %fifteen = arith.constant 15 : i64
    %ten = arith.constant 10 : i64
    %digit_zero = arith.constant 48 : i64
    %letter_a = arith.constant 87 : i64
    %close_byte = arith.constant 62 : i8
    %released_bit = arith.constant 2 : i64
    %flags_slot = arith.constant 6 : index
    %flags = memref.load %self[%flags_slot] : memref<8xi64>
    %bit = arith.andi %flags, %released_bit : i64
    %released = arith.cmpi ne, %bit, %zero : i64
    %buffer = memref.alloca() : memref<48xi8>
    %open_len = scf.if %released -> (index) {
      %static = memref.get_global @__ly_memoryview_repr_released : memref<22xi8>
      %c22 = arith.constant 22 : index
      scf.for %i = %c0 to %c22 step %c1 {
        %byte = memref.load %static[%i] : memref<22xi8>
        memref.store %byte, %buffer[%i] : memref<48xi8>
      }
      scf.yield %c22 : index
    } else {
      %static = memref.get_global @__ly_memoryview_repr_open : memref<13xi8>
      %c13 = arith.constant 13 : index
      scf.for %i = %c0 to %c13 step %c1 {
        %byte = memref.load %static[%i] : memref<13xi8>
        memref.store %byte, %buffer[%i] : memref<48xi8>
      }
      scf.yield %c13 : index
    }
    %self_index = memref.extract_aligned_pointer_as_index %self : memref<8xi64> -> index
    %address = arith.index_cast %self_index : index to i64
    // The address's hex digits, most significant first, without leading zeros.
    %c16 = arith.constant 16 : index
    %written = scf.for %i = %c0 to %c16 step %c1 iter_args(%at = %open_len) -> (index) {
      %i64 = arith.index_cast %i : index to i64
      %c15 = arith.constant 15 : i64
      %from_top = arith.subi %c15, %i64 : i64
      %shift = arith.muli %from_top, %four : i64
      %shifted = arith.shrui %address, %shift : i64
      %nibble = arith.andi %shifted, %fifteen : i64
      %rest = arith.shrui %address, %shift : i64
      %significant = arith.cmpi ne, %rest, %zero : i64
      %last = arith.cmpi eq, %from_top, %zero : i64
      %emit = arith.ori %significant, %last : i1
      %next = scf.if %emit -> (index) {
        %is_digit = arith.cmpi slt, %nibble, %ten : i64
        %base = arith.select %is_digit, %digit_zero, %letter_a : i64
        %code = arith.addi %base, %nibble : i64
        %byte = arith.trunci %code : i64 to i8
        memref.store %byte, %buffer[%at] : memref<48xi8>
        %moved = arith.addi %at, %c1 : index
        scf.yield %moved : index
      } else {
        scf.yield %at : index
      }
      scf.yield %next : index
    }
    memref.store %close_byte, %buffer[%written] : memref<48xi8>
    %total = arith.addi %written, %c1 : index
    %total64 = arith.index_cast %total : index to i64
    %text = memref.cast %buffer : memref<48xi8> to memref<?xi8>
    %h, %b = func.call @__ly_unicode_from_valid_utf8(%text, %c0, %total64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %h, %b : memref<2xi64>, memref<?xi8>
  }

  // ===== the attributes (memory_getsets) =====

  func.func @LyMemoryView_FieldReadonly(%self: memref<8xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.memoryview", ly.runtime.primitive = "field.readonly"} {
    func.call @__ly_memoryview_check(%self) : (memref<8xi64>) -> ()
    %readonly = func.call @__ly_memoryview_readonly(%self) : (memref<8xi64>) -> i1
    func.return %readonly : i1
  }

  func.func @LyMemoryView_FieldFormat(%self: memref<8xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.memoryview", ly.runtime.primitive = "field.format", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1 : i64
    func.call @__ly_memoryview_check(%self) : (memref<8xi64>) -> ()
    %static = memref.get_global @__ly_memoryview_format : memref<1xi8>
    %text = memref.cast %static : memref<1xi8> to memref<?xi8>
    %h, %b = func.call @__ly_unicode_from_valid_utf8(%text, %c0, %one) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %h, %b : memref<2xi64>, memref<?xi8>
  }

  func.func @LyMemoryView_FieldItemsize(%self: memref<8xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.memoryview", ly.runtime.primitive = "field.itemsize", ly.runtime.result_contract = "builtins.int"} {
    %one = arith.constant 1 : i64
    func.call @__ly_memoryview_check(%self) : (memref<8xi64>) -> ()
    %result = func.call @LyLong_FromI64(%one) : (i64) -> memref<2xi64>
    func.return %result : memref<2xi64>
  }

  func.func @LyMemoryView_FieldNbytes(%self: memref<8xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.memoryview", ly.runtime.primitive = "field.nbytes", ly.runtime.result_contract = "builtins.int"} {
    %s3 = arith.constant 3 : index
    func.call @__ly_memoryview_check(%self) : (memref<8xi64>) -> ()
    %length = memref.load %self[%s3] : memref<8xi64>
    %result = func.call @LyLong_FromI64(%length) : (i64) -> memref<2xi64>
    func.return %result : memref<2xi64>
  }

  func.func @LyMemoryView_FieldNdim(%self: memref<8xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.memoryview", ly.runtime.primitive = "field.ndim", ly.runtime.result_contract = "builtins.int"} {
    %one = arith.constant 1 : i64
    func.call @__ly_memoryview_check(%self) : (memref<8xi64>) -> ()
    %result = func.call @LyLong_FromI64(%one) : (i64) -> memref<2xi64>
    func.return %result : memref<2xi64>
  }

  // A 1-tuple of the word at `slot`: shape (the length) or strides (the
  // stride).
  func.func private @__ly_memoryview_one_tuple(%self: memref<8xi64>, %slot: index) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.memoryview"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %int_class = arith.constant {ly.class_id_of = "builtins.int"} 1 : i64
    func.call @__ly_memoryview_check(%self) : (memref<8xi64>) -> ()
    %value = memref.load %self[%slot] : memref<8xi64>
    %tuple = func.call @__ly_tuple_alloc(%one) : (i64) -> memref<5xi64>
    %items = func.call @__ly_tuple_items(%tuple) : (memref<5xi64>) -> memref<?xi64>
    %word = func.call @LyLong_SlotWordFromI64(%value) : (i64) -> i64
    func.call @__ly_box_store_entity(%items, %zero, %int_class, %word) : (memref<?xi64>, i64, i64, i64) -> ()
    func.return %tuple : memref<5xi64>
  }

  func.func @LyMemoryView_FieldShape(%self: memref<8xi64> {ly.ownership.object_header}) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.memoryview", ly.runtime.primitive = "field.shape", ly.runtime.result_contract = "builtins.tuple"} {
    %s3 = arith.constant 3 : index
    %tuple = func.call @__ly_memoryview_one_tuple(%self, %s3) : (memref<8xi64>, index) -> memref<5xi64>
    func.return %tuple : memref<5xi64>
  }

  func.func @LyMemoryView_FieldStrides(%self: memref<8xi64> {ly.ownership.object_header}) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.memoryview", ly.runtime.primitive = "field.strides", ly.runtime.result_contract = "builtins.tuple"} {
    %s4 = arith.constant 4 : index
    %tuple = func.call @__ly_memoryview_one_tuple(%self, %s4) : (memref<8xi64>, index) -> memref<5xi64>
    func.return %tuple : memref<5xi64>
  }

  // One dimension is C- and Fortran-contiguous alike: a stride of one item,
  // or at most one item to step over.
  func.func private @__ly_memoryview_contiguous(%self: memref<8xi64>) -> i1 attributes {ly.runtime.contract = "builtins.memoryview"} {
    %one = arith.constant 1 : i64
    %s3 = arith.constant 3 : index
    %s4 = arith.constant 4 : index
    func.call @__ly_memoryview_check(%self) : (memref<8xi64>) -> ()
    %length = memref.load %self[%s3] : memref<8xi64>
    %stride = memref.load %self[%s4] : memref<8xi64>
    %unit = arith.cmpi eq, %stride, %one : i64
    %short = arith.cmpi sle, %length, %one : i64
    %contiguous = arith.ori %unit, %short : i1
    func.return %contiguous : i1
  }

  func.func @LyMemoryView_FieldContiguous(%self: memref<8xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.memoryview", ly.runtime.primitive = "field.contiguous"} {
    %contiguous = func.call @__ly_memoryview_contiguous(%self) : (memref<8xi64>) -> i1
    func.return %contiguous : i1
  }

  func.func @LyMemoryView_FieldCContiguous(%self: memref<8xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.memoryview", ly.runtime.primitive = "field.c_contiguous"} {
    %contiguous = func.call @__ly_memoryview_contiguous(%self) : (memref<8xi64>) -> i1
    func.return %contiguous : i1
  }

  func.func @LyMemoryView_FieldFContiguous(%self: memref<8xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.memoryview", ly.runtime.primitive = "field.f_contiguous"} {
    %contiguous = func.call @__ly_memoryview_contiguous(%self) : (memref<8xi64>) -> i1
    func.return %contiguous : i1
  }

  // ===== iteration (memoryiterobject) =====
  //
  // [refcount, class 29, index, the view's address -- 0 once exhausted].
  func.func @LyMemoryView_Iter(%self: memref<8xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.memoryview", ly.runtime.method = "__iter__", ly.runtime.result_contract = "builtins.memory_iterator"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %class_iterator = arith.constant {ly.class_id_of = "builtins.memory_iterator"} 29 : i64
    %header_bytes = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %s0 = arith.constant 0 : index
    %s1 = arith.constant 1 : index
    %s2 = arith.constant 2 : index
    %s3 = arith.constant 3 : index
    func.call @__ly_memoryview_check(%self) : (memref<8xi64>) -> ()
    %raw = memref.alloc(%header_bytes) {alignment = 16 : i64} : memref<?xi8>
    %iterator = memref.view %raw[%c0][] {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<?xi8> to memref<4xi64>
    %self_index = memref.extract_aligned_pointer_as_index %self : memref<8xi64> -> index
    %self_word = arith.index_cast %self_index : index to i64
    memref.store %one, %iterator[%s0] : memref<4xi64>
    memref.store %class_iterator, %iterator[%s1] : memref<4xi64>
    memref.store %zero, %iterator[%s2] : memref<4xi64>
    memref.store %self_word, %iterator[%s3] : memref<4xi64>
    %refcount_view = memref.subview %self[0] [2] [1] : memref<8xi64> to memref<2xi64, strided<[1]>>
    %retained = memref.cast %refcount_view : memref<2xi64, strided<[1]>> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%retained) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %iterator : memref<4xi64>
  }

  func.func @LyMemoryIterator_Iter(%self: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.memory_iterator", ly.runtime.method = "__iter__", ly.runtime.result_contract = "builtins.memory_iterator"} {
    %refcount_view = memref.subview %self[0] [2] [1] : memref<4xi64> to memref<2xi64, strided<[1]>>
    %retained = memref.cast %refcount_view : memref<2xi64, strided<[1]>> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%retained) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %self : memref<4xi64>
  }

  // The next item; a view released while items remain is the ValueError
  // CHECK_RELEASED raises.
  func.func @LyMemoryIterator_Next(%self: memref<4xi64> {ly.ownership.object_header}) -> (memref<2xi64>, i1, memref<4xi64>) attributes {ly.ownership.owned_result_contracts = ["builtins.int", "builtins.memory_iterator"], ly.ownership.owned_results = [0, 2], ly.runtime.contract = "builtins.memory_iterator", ly.runtime.method = "__next__", ly.runtime.element_contract = "builtins.int", ly.runtime.next_contract = "builtins.memory_iterator", ly.runtime.valid_result_index = 1 : i64} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %eight = arith.constant 8 : i64
    %false = arith.constant false
    %index_slot = arith.constant 2 : index
    %view_slot = arith.constant 3 : index
    %length_slot = arith.constant 3 : index
    %stride_slot = arith.constant 4 : index
    %data_slot = arith.constant 2 : index
    %index = memref.load %self[%index_slot] : memref<4xi64>
    %view_word = memref.load %self[%view_slot] : memref<4xi64>
    %attached = arith.cmpi ne, %view_word, %zero : i64
    %valid, %value = scf.if %attached -> (i1, i64) {
      %words = func.call @__ly_global_view_i64(%view_word, %eight) : (i64, i64) -> memref<?xi64>
      %view = memref.cast %words : memref<?xi64> to memref<8xi64>
      %length = memref.load %view[%length_slot] : memref<8xi64>
      %more = arith.cmpi slt, %index, %length : i64
      %item = scf.if %more -> (i64) {
        func.call @__ly_memoryview_check(%view) : (memref<8xi64>) -> ()
        %data = memref.load %view[%data_slot] : memref<8xi64>
        %stride = memref.load %view[%stride_slot] : memref<8xi64>
        %offset = arith.muli %index, %stride : i64
        %address = arith.addi %data, %offset : i64
        %byte = func.call @__ly_memoryview_load(%address) : (i64) -> i64
        scf.yield %byte : i64
      } else {
        memref.store %zero, %self[%view_slot] : memref<4xi64>
        func.call @LyMemoryView_DecRef(%view) : (memref<8xi64>) -> ()
        scf.yield %zero : i64
      }
      scf.yield %more, %item : i1, i64
    } else {
      scf.yield %false, %zero : i1, i64
    }
    %advanced = arith.addi %index, %one : i64
    %next_index = arith.select %valid, %advanced, %index : i64
    memref.store %next_index, %self[%index_slot] : memref<4xi64>
    %refcount_view = memref.subview %self[0] [2] [1] : memref<4xi64> to memref<2xi64, strided<[1]>>
    %retained = memref.cast %refcount_view : memref<2xi64, strided<[1]>> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%retained) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    %element = func.call @LyLong_FromI64(%value) : (i64) -> memref<2xi64>
    func.return %element, %valid, %self : memref<2xi64>, i1, memref<4xi64>
  }

  func.func @LyMemoryIterator_DecRef(%self: memref<4xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.memory_iterator", ly.runtime.deallocator} {
    %storage = memref.cast %self : memref<4xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    %zero = arith.constant 0 : i64
    %eight = arith.constant 8 : i64
    %view_slot = arith.constant 3 : index
    %view_word = memref.load %self[%view_slot] : memref<4xi64>
    %attached = arith.cmpi ne, %view_word, %zero : i64
    scf.if %attached {
      %words = func.call @__ly_global_view_i64(%view_word, %eight) : (i64, i64) -> memref<?xi64>
      %view = memref.cast %words : memref<?xi64> to memref<8xi64>
      func.call @LyMemoryView_DecRef(%view) : (memref<8xi64>) -> ()
    }
    memref.dealloc %self : memref<4xi64>
    cf.br ^done

  ^done:
    func.return
  }
}
