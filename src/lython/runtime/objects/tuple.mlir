// `tuple` -- CPython's Objects/tupleobject.c.
//
// Signature source (1:1 correspondence target):
//   https://github.com/python/typeshed/blob/main/stdlib/builtins.pyi

module attributes {
  ly.typing.manifest,
  ly.runtime.contracts = ["builtins.tuple"]
} {
  // ===== declared here, defined in another runtime file or built by the lowering =====
  func.func private @__ly_unicode_from_valid_utf8(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]}
  func.func private @LyErr_NoMemory() attributes {ly.runtime.contract = "builtins.MemoryError"}
  func.func private @LyLong_FromI64(%value: i64 {ly.runtime.default_i64 = 0 : i64}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 1 : i64, ly.runtime.contract = "builtins.int", ly.runtime.initializer = "__new__"}
  func.func private @LyObject_ReleaseBoxedPayloadArraySlotRaw(%payload: memref<?xi64>, %logical_index: i64)
  func.func private @LyObject_ReleaseStorageToZero(%storage: memref<?xi64>) -> i1 attributes {ly.runtime.contract = "builtins.object", ly.runtime.primitive = "release_to_zero"}
  func.func private @LyUnicode_Concat(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_bytes: memref<?xi8>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__add__"}
  func.func private @LyUnicode_DecRef(%header: memref<2xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.str", ly.runtime.deallocator}
  func.func private @LyUnicode_FromBytes(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 4 : i64, ly.runtime.contract = "builtins.str", ly.runtime.initializer = "__new__"}
  func.func private @__ly_alloc_count(%count: i64, %element_bytes: i64, %extra: i64) -> index
  func.func private @__ly_box_hash(%box: !llvm.ptr) -> i64
  func.func private @__ly_box_store_entity(%items: memref<?xi64>, %slot: i64, %class_id: i64, %entity: i64)
  func.func private @__ly_box_word_count() -> i64
  func.func private @__ly_check_alloc_count(%count: i64, %element_bytes: i64, %extra: i64)
  func.func private @__ly_global_view_i64(%pointer: i64, %size: i64) -> memref<?xi64>
  func.func private @__ly_global_view_i8(%pointer: i64, %size: i64) -> memref<?xi8>
  func.func private @__ly_long_parts(%header: memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
  func.func private @__ly_raise_static_message(%class_id: i64, %message: memref<?xi8>, %length: i64)
  func.func private @__ly_repeat_overflows(%len: i64, %n: i64) -> i1
  func.func private @__ly_repr_boxed_or_default(%box_ptr: !llvm.ptr, %class_id: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.object", ly.runtime.primitive = "repr_boxed_or_default", ly.runtime.result_contract = "builtins.str"}
  memref.global "private" constant @__ly_repr_comma : memref<2xi8>
  memref.global "private" constant @__ly_repr_lparen : memref<1xi8>
  memref.global "private" constant @__ly_repr_rparen : memref<1xi8>
  func.func private @__ly_seq_fill_concat(%dst_items: memref<?xi64>, %llen: i64, %li: memref<?xi64>, %rlen: i64, %ri: memref<?xi64>)
  func.func private @__ly_seq_fill_copy(%dst_items: memref<?xi64>, %src_len: i64, %src_items: memref<?xi64>)
  func.func private @__ly_seq_fill_repeat(%dst_items: memref<?xi64>, %len: i64, %li: memref<?xi64>, %n: i64)
  func.func private @__ly_seq_fill_slice(%dst_items: memref<?xi64>, %count: i64, %start: i64, %step: i64, %src_items: memref<?xi64>)
  func.func private @__ly_seq_repeat_count(%nm: memref<2xi64>, %nd: memref<?xi32>) -> i64
  func.func private @__ly_seq_slice_bounds(%len: i64, %start_raw: i64, %stop_raw: i64, %step_raw: i64, %mask: i64) -> (i64, i64, i64)
  func.func private @__ly_sequence_compare_lens(%lhs_len: i64, %lhs_items: memref<?xi64>, %rhs_len: i64, %rhs_items: memref<?xi64>) -> i64
  func.func private @__ly_sequence_count_lens(%len: i64, %items: memref<?xi64>, %probe: !llvm.ptr) -> i64
  func.func private @__ly_sequence_equal_lens(%lhs_len: i64, %lhs_items: memref<?xi64>, %rhs_len: i64, %rhs_items: memref<?xi64>) -> i1
  func.func private @__ly_sequence_find_lens(%len: i64, %items: memref<?xi64>, %probe: !llvm.ptr) -> i64
  func.func private @__ly_slot_class(%word: i64) -> i64

  py.class @tuple attributes {
    base_names = ["Sequence"], ly.typing.params = ["T"],
    ly.runtime.contract = "builtins.tuple", ly.runtime.required,
    ly.runtime.required_deallocator,
    ly.runtime.required_initializers = ["__new__"],
    ly.runtime.required_methods = ["__len__"],
    ly.typing.param_variance = ["covariant"],
    ly.typing.base_args = [[!py.contract<"$T">]],
    method_names = ["__len__", "__contains__", "__getitem__", "__getslice__",
                    "__iter__",
                    "__add__", "__mul__", "count", "index", "__repr__",
                    "__hash__", "__eq__", "__ne__", "__lt__", "__le__",
                    "__gt__", "__ge__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"builtins.tuple">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.tuple">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.tuple">, !py.contract<"typing.SupportsIndex">] -> [!py.contract<"$T">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.tuple">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.tuple", [!py.contract<"$T">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.tuple">] -> [!py.protocol<"Iterator", [!py.contract<"$T">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.tuple">, !py.contract<"builtins.tuple">] -> [!py.contract<"builtins.tuple">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.tuple">, !py.contract<"typing.SupportsIndex">] -> [!py.contract<"builtins.tuple">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.tuple">, !py.contract<"typing.Any">] -> [!py.contract<"builtins.int">]>,
      // `index` is the one-argument form only, matching builtins.list.index:
      // the start/stop window was declared here but never implemented, so the
      // wider spelling was an empty promise that only failed at lowering.
      !py.protocol<"Callable", [!py.contract<"builtins.tuple">, !py.contract<"typing.Any">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.tuple">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.tuple">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.tuple">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.tuple">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.tuple">, !py.contract<"builtins.tuple">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.tuple">, !py.contract<"builtins.tuple">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.tuple">, !py.contract<"builtins.tuple">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.tuple">, !py.contract<"builtins.tuple">] -> [!py.contract<"builtins.bool">]>
    ],
    method_kinds = ["instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance"]
  } {}

  // tuple.__hash__: CPython's xxHash-based combiner (tuples of equal elements
  // hash equal; unhashable elements raise through __ly_box_hash).
  func.func @LyTuple_Hash(%self: memref<5xi64> {ly.ownership.object_header}) -> i64 attributes {ly.runtime.contract = "builtins.tuple", ly.runtime.method = "__hash__"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16_i64 = func.call @__ly_box_word_count() : () -> i64
    %length_slot = arith.constant 2 : index
    %len = memref.load %self[%length_slot] : memref<5xi64>
    %items = func.call @__ly_tuple_items(%self) : (memref<5xi64>) -> memref<?xi64>
    %len_index = arith.index_cast %len : i64 to index
    %items_idx = memref.extract_aligned_pointer_as_index %items : memref<?xi64> -> index
    %items_i64 = arith.index_cast %items_idx : index to i64
    %items_ptr = llvm.inttoptr %items_i64 : i64 to !llvm.ptr
    %prime1 = arith.constant -7046029288634856825 : i64
    %prime2 = arith.constant -4417276706812531889 : i64
    %prime5 = arith.constant 2870177450012600261 : i64
    %acc_init = arith.constant 2870177450012600261 : i64
    %c31 = arith.constant 31 : i64
    %c33 = arith.constant 33 : i64
    %acc = scf.for %i = %c0 to %len_index step %c1 iter_args(%a = %acc_init) -> (i64) {
      %i_i64 = arith.index_cast %i : index to i64
      %off = arith.muli %i_i64, %c16_i64 : i64
      %box_ptr = llvm.getelementptr %items_ptr[%off] : (!llvm.ptr, i64) -> !llvm.ptr, i64
      %lane = func.call @__ly_box_hash(%box_ptr) : (!llvm.ptr) -> i64
      %scaled = arith.muli %lane, %prime2 : i64
      %added = arith.addi %a, %scaled : i64
      %rot_hi = arith.shli %added, %c31 : i64
      %rot_lo = arith.shrui %added, %c33 : i64
      %rotated = arith.ori %rot_hi, %rot_lo : i64
      %next = arith.muli %rotated, %prime1 : i64
      scf.yield %next : i64
    }
    // acc += len ^ (XXPRIME_5 ^ 3527539)
    %salt = arith.constant 3527539 : i64
    %mix0 = arith.xori %prime5, %salt : i64
    %mix1 = arith.xori %len, %mix0 : i64
    %final = arith.addi %acc, %mix1 : i64
    %sentinel = arith.constant -1 : i64
    %replacement = arith.constant 1546275796 : i64
    %is_sentinel = arith.cmpi eq, %final, %sentinel : i64
    %result = arith.select %is_sentinel, %replacement, %final : i1, i64
    func.return %result : i64
  }

  func.func @LyTuple_EqBool(%lhs: memref<5xi64> {ly.ownership.object_header}, %rhs: memref<5xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.tuple", ly.runtime.method = "__eq__"} {
    %length_slot = arith.constant 2 : index
    %lhs_len = memref.load %lhs[%length_slot] : memref<5xi64>
    %rhs_len = memref.load %rhs[%length_slot] : memref<5xi64>
    %lhs_items = func.call @__ly_tuple_items(%lhs) : (memref<5xi64>) -> memref<?xi64>
    %rhs_items = func.call @__ly_tuple_items(%rhs) : (memref<5xi64>) -> memref<?xi64>
    %eq = func.call @__ly_sequence_equal_lens(%lhs_len, %lhs_items, %rhs_len, %rhs_items) : (i64, memref<?xi64>, i64, memref<?xi64>) -> i1
    func.return %eq : i1
  }

  func.func @LyTuple_NeBool(%lhs: memref<5xi64> {ly.ownership.object_header}, %rhs: memref<5xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.tuple", ly.runtime.method = "__ne__"} {
    %eq = func.call @LyTuple_EqBool(%lhs, %rhs) : (memref<5xi64>, memref<5xi64>) -> i1
    %true = arith.constant true
    %ne = arith.xori %eq, %true : i1
    func.return %ne : i1
  }

  // Lane-shaped wrapper for the contracts whose length is still a lane.

  // Lexicographic compare of two tuple handles. Length by value and a plain
  // slot array, so the shared `__ly_sequence_compare_lens` is reached without
  // a tuple-shaped variant of it.
  func.func private @__ly_tuple_compare(%lhs: memref<5xi64>, %rhs: memref<5xi64>) -> i64 {
    %length_slot = arith.constant 2 : index
    %lhs_len = memref.load %lhs[%length_slot] : memref<5xi64>
    %rhs_len = memref.load %rhs[%length_slot] : memref<5xi64>
    %li = func.call @__ly_tuple_items(%lhs) : (memref<5xi64>) -> memref<?xi64>
    %ri = func.call @__ly_tuple_items(%rhs) : (memref<5xi64>) -> memref<?xi64>
    %cmp = func.call @__ly_sequence_compare_lens(%lhs_len, %li, %rhs_len, %ri) : (i64, memref<?xi64>, i64, memref<?xi64>) -> i64
    func.return %cmp : i64
  }

  func.func @LyTuple_LtBool(%lhs: memref<5xi64> {ly.ownership.object_header}, %rhs: memref<5xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.tuple", ly.runtime.method = "__lt__"} {
    %zero = arith.constant 0 : i64
    %cmp = func.call @__ly_tuple_compare(%lhs, %rhs) : (memref<5xi64>, memref<5xi64>) -> i64
    %lt = arith.cmpi slt, %cmp, %zero : i64
    func.return %lt : i1
  }

  func.func @LyTuple_LeBool(%lhs: memref<5xi64> {ly.ownership.object_header}, %rhs: memref<5xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.tuple", ly.runtime.method = "__le__"} {
    %one = arith.constant 1 : i64
    %cmp = func.call @__ly_tuple_compare(%lhs, %rhs) : (memref<5xi64>, memref<5xi64>) -> i64
    %le = arith.cmpi slt, %cmp, %one : i64
    func.return %le : i1
  }

  func.func @LyTuple_GtBool(%lhs: memref<5xi64> {ly.ownership.object_header}, %rhs: memref<5xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.tuple", ly.runtime.method = "__gt__"} {
    %zero = arith.constant 0 : i64
    %cmp = func.call @__ly_tuple_compare(%lhs, %rhs) : (memref<5xi64>, memref<5xi64>) -> i64
    %gt = arith.cmpi sgt, %cmp, %zero : i64
    func.return %gt : i1
  }

  func.func @LyTuple_GeBool(%lhs: memref<5xi64> {ly.ownership.object_header}, %rhs: memref<5xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.tuple", ly.runtime.method = "__ge__"} {
    %minus_one = arith.constant -1 : i64
    %cmp = func.call @__ly_tuple_compare(%lhs, %rhs) : (memref<5xi64>, memref<5xi64>) -> i64
    %ge = arith.cmpi sgt, %cmp, %minus_one : i64
    func.return %ge : i1
  }

  // Store an owned int (h, m, d) into a tuple slot as its canonical handle.
  // ⛔ ONE LANE. This wrote three -- header, meta and digits -- which is what
  // `builtins.int` expanded to before its handle narrowed to the header alone.
  // The extra pointers were dead words no reader looked at once
  // `__ly_boxed_long_view` started going through the entity, but word 3 said
  // THREE and that is the count a box describes itself by.
  func.func private @__ly_tuple_store_long(%items: memref<?xi64>, %slot: index, %h: memref<2xi64>) {
    %one = arith.constant 1 : i64
    %int_class = arith.constant 1 : i64
    %slot_i64 = arith.index_cast %slot : index to i64
    %h_idx = memref.extract_aligned_pointer_as_index %h : memref<2xi64> -> index
    %h_ptr = arith.index_cast %h_idx : index to i64
    func.call @__ly_box_store_entity(%items, %slot_i64, %int_class, %h_ptr) : (memref<?xi64>, i64, i64, i64) -> ()
    func.return
  }

  func.func @LyTuple_Concat(%lhs: memref<5xi64> {ly.ownership.object_header}, %rhs: memref<5xi64> {ly.ownership.object_header}) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.tuple", ly.runtime.method = "__add__", ly.runtime.result_contract = "builtins.tuple"} {
    %length_slot = arith.constant 2 : index
    %llen = memref.load %lhs[%length_slot] : memref<5xi64>
    %rlen = memref.load %rhs[%length_slot] : memref<5xi64>
    %li = func.call @__ly_tuple_items(%lhs) : (memref<5xi64>) -> memref<?xi64>
    %ri = func.call @__ly_tuple_items(%rhs) : (memref<5xi64>) -> memref<?xi64>
    %result = func.call @__ly_tuple_concat_alloc(%llen, %li, %rlen, %ri) : (i64, memref<?xi64>, i64, memref<?xi64>) -> memref<5xi64>
    func.return %result : memref<5xi64>
  }

  func.func @LyTuple_Repeat(%self: memref<5xi64> {ly.ownership.object_header}, %nh: memref<2xi64> {ly.ownership.object_header}) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.tuple", ly.runtime.method = "__mul__", ly.runtime.result_contract = "builtins.tuple"} {
    %nm, %nd = func.call @__ly_long_parts(%nh) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %length_slot = arith.constant 2 : index
    %len = memref.load %self[%length_slot] : memref<5xi64>
    %items = func.call @__ly_tuple_items(%self) : (memref<5xi64>) -> memref<?xi64>
    %n = func.call @__ly_seq_repeat_count(%nm, %nd) : (memref<2xi64>, memref<?xi32>) -> i64
    %result = func.call @__ly_tuple_repeat_alloc(%len, %items, %n) : (i64, memref<?xi64>, i64) -> memref<5xi64>
    func.return %result : memref<5xi64>
  }

  func.func @LyTuple_ContainsBox(%self: memref<5xi64> {ly.ownership.object_header}, %elem_box: memref<5xi64>) -> i1 attributes {ly.runtime.contract = "builtins.tuple", ly.runtime.primitive = "contains_box"} {
    %minus_one = arith.constant -1 : i64
    %length_slot = arith.constant 2 : index
    %len = memref.load %self[%length_slot] : memref<5xi64>
    %items = func.call @__ly_tuple_items(%self) : (memref<5xi64>) -> memref<?xi64>
    %box_idx = memref.extract_aligned_pointer_as_index %elem_box : memref<5xi64> -> index
    %box_i64 = arith.index_cast %box_idx : index to i64
    %box_ptr = llvm.inttoptr %box_i64 : i64 to !llvm.ptr
    %found = func.call @__ly_sequence_find_lens(%len, %items, %box_ptr) : (i64, memref<?xi64>, !llvm.ptr) -> i64
    %result = arith.cmpi ne, %found, %minus_one : i64
    func.return %result : i1
  }

  func.func @LyTuple_CountBox(%self: memref<5xi64> {ly.ownership.object_header}, %elem_box: memref<5xi64>) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.tuple", ly.runtime.primitive = "count_box", ly.runtime.result_contract = "builtins.int"} {
    %length_slot = arith.constant 2 : index
    %len = memref.load %self[%length_slot] : memref<5xi64>
    %items = func.call @__ly_tuple_items(%self) : (memref<5xi64>) -> memref<?xi64>
    %box_idx = memref.extract_aligned_pointer_as_index %elem_box : memref<5xi64> -> index
    %box_i64 = arith.index_cast %box_idx : index to i64
    %box_ptr = llvm.inttoptr %box_i64 : i64 to !llvm.ptr
    %count = func.call @__ly_sequence_count_lens(%len, %items, %box_ptr) : (i64, memref<?xi64>, !llvm.ptr) -> i64
    %h = func.call @LyLong_FromI64(%count) : (i64) -> memref<2xi64>
    func.return %h : memref<2xi64>
  }
  // "tuple.index(x): x not in tuple"
  memref.global "private" constant @__ly_tuple_msg_index_missing : memref<30xi8> = dense<[116, 117, 112, 108, 101, 46, 105, 110, 100, 101, 120, 40, 120, 41, 58, 32, 120, 32, 110, 111, 116, 32, 105, 110, 32, 116, 117, 112, 108, 101]>

  func.func @LyTuple_IndexBox(%self: memref<5xi64> {ly.ownership.object_header}, %elem_box: memref<5xi64>) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.tuple", ly.runtime.primitive = "index_box", ly.runtime.result_contract = "builtins.int"} {
    %minus_one = arith.constant -1 : i64
    %length_slot = arith.constant 2 : index
    %len = memref.load %self[%length_slot] : memref<5xi64>
    %items = func.call @__ly_tuple_items(%self) : (memref<5xi64>) -> memref<?xi64>
    %box_idx = memref.extract_aligned_pointer_as_index %elem_box : memref<5xi64> -> index
    %box_i64 = arith.index_cast %box_idx : index to i64
    %box_ptr = llvm.inttoptr %box_i64 : i64 to !llvm.ptr
    %found = func.call @__ly_sequence_find_lens(%len, %items, %box_ptr) : (i64, memref<?xi64>, !llvm.ptr) -> i64
    %missing = arith.cmpi eq, %found, %minus_one : i64
    scf.if %missing {
      %value_error = arith.constant 53 : i64
      %msg_static = memref.get_global @__ly_tuple_msg_index_missing : memref<30xi8>
      %msg = memref.cast %msg_static : memref<30xi8> to memref<?xi8>
      %msg_len = arith.constant 30 : i64
      func.call @__ly_raise_static_message(%value_error, %msg, %msg_len) : (i64, memref<?xi8>, i64) -> ()
    }
    %h = func.call @LyLong_FromI64(%found) : (i64) -> memref<2xi64>
    func.return %h : memref<2xi64>
  }

  func.func private @LyTuple_Shape() -> memref<5xi64> attributes {ly.runtime.contract = "builtins.tuple", ly.runtime.shape}

  // ===== builtins.tuple: one entity, one root =====
  //
  // The handle is `memref<5xi64>`:
  //
  //   word 0  refcount            word 3  capacity
  //   word 1  class id (11)       word 4  items base address
  //   word 2  length              (items follow in the same block)
  //
  // Words 0..4 are ContainerLayout.h's sequence prefix, the same assignment
  // `list` uses.
  //
  // Why the items array is an ADDRESS in the handle and not a lane beside it:
  // a lane travels beside the root, so a holder can keep one across a
  // reallocation. A tuple is immutable and never grows, so nothing reallocates
  // it -- but the property that matters is not growth, it is that the entity
  // has ONE root. Three lanes made `tuple` indistinguishable from `set` and
  // `frozenset` on both release interface and canonical shape, which is the
  // configuration that hid a double free for `str`/`bytes`
  // (rfc/memory-safety-proof.md, `NonInstantiationIsNotConformance`).
  //
  // Why this is a FORK of the once-shared `__ly_sequence_alloc` rather than a
  // conversion of it: `set` and `frozenset` were being converted on another
  // track at the time, and a fork kept both sides buildable with disjoint
  // diffs (`list` set the precedent with `__ly_list_alloc`). All five forks
  // now exist and the shared original is deleted -- so this comment records
  // why the duplication was accepted, not a shared function still to unify.
  // PyTuple_New(n) takes exactly n slots, and a tuple never grows, so the
  // minimum this used to round up to was pure waste -- 8 KB for a pair.
  // A tuple laid out at compile time in read-only data with the immortal
  // refcount, its items address pointing into itself (a relocation): a
  // constant literal. Nothing allocates and nothing is ever written.
  func.func @LyTuple_FromStatic(%address: i64) -> memref<5xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.tuple"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.tuple", ly.runtime.primitive = "from_static"} {
    %handle_bytes = arith.constant 40 : i64
    %raw = func.call @__ly_global_view_i8(%address, %handle_bytes) : (i64, i64) -> memref<?xi8>
    %c0 = arith.constant 0 : index
    %self = memref.view %raw[%c0][] {ly.ownership.object_header} : memref<?xi8> to memref<5xi64>
    func.return %self : memref<5xi64>
  }

  // The empty tuple every empty result is, as CPython's is a singleton:
  // immortal, never written, and holding no items (its items address is 0
  // and nothing reads past a length of 0).
  memref.global "private" constant @__ly_tuple_empty : memref<5xi64> = dense<[9223372036854775807, 11, 0, 0, 0]> {alignment = 16 : i64}

  func.func private @__ly_tuple_alloc(%length: i64) -> memref<5xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.tuple"], ly.ownership.owned_results = [0]} {
    %zero_length = arith.constant 0 : i64
    %is_empty = arith.cmpi sle, %length, %zero_length : i64
    %r = scf.if %is_empty -> memref<5xi64> {
      %empty = memref.get_global @__ly_tuple_empty : memref<5xi64>
      scf.yield %empty : memref<5xi64>
    } else {
      %fresh = func.call @__ly_tuple_alloc_fresh(%length) : (i64) -> memref<5xi64>
      scf.yield %fresh : memref<5xi64>
    }
    func.return %r : memref<5xi64>
  }

  func.func private @__ly_tuple_alloc_fresh(%length: i64) -> memref<5xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.tuple"], ly.ownership.owned_results = [0]} {
    %one = arith.constant 1 : i64
    %handle_words = func.call @__ly_box_word_count() : () -> i64
    %class_id = arith.constant 11 : i64
    %zero = arith.constant 0 : i64
    %eight = arith.constant 8 : i64
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %length_slot = arith.constant 2 : index
    %capacity_slot = arith.constant 3 : index
    %items_slot = arith.constant 4 : index
    // ⭐ ONE ALLOCATION, as CPython's PyTupleObject is one: the five handle
    // words a tuple uses (refcount, class, length, capacity, items address)
    // and then the items, starting where word 5 would.
    // ⛔ Not the separate items malloc this was: a pair took a 112-byte
    // handle and a 16-byte array, two allocations, where it now takes 56 bytes.
    %used_words = arith.constant 5 : i64
    %capacity = arith.maxsi %length, %zero : i64
    %slot_bytes = arith.muli %handle_words, %eight : i64
    %used_bytes = arith.muli %used_words, %eight : i64
    %capacity_index = func.call @__ly_alloc_count(%capacity, %slot_bytes, %used_bytes) : (i64, i64, i64) -> index
    %slot_bytes_index = arith.index_cast %slot_bytes : i64 to index
    %used_bytes_index = arith.index_cast %used_bytes : i64 to index
    %payload_bytes = arith.muli %capacity_index, %slot_bytes_index : index
    %block_bytes = arith.addi %payload_bytes, %used_bytes_index : index
    %block = memref.alloc(%block_bytes) {alignment = 16 : i64} : memref<?xi8>
    %c0 = arith.constant 0 : index
    %self = memref.view %block[%c0][] {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<?xi8> to memref<5xi64>
    %self_index = memref.extract_aligned_pointer_as_index %self : memref<5xi64> -> index
    %self_word = arith.index_cast %self_index : index to i64
    %items_offset = arith.muli %used_words, %eight : i64
    %items_word = arith.addi %self_word, %items_offset : i64

    memref.store %one, %self[%refcount_slot] : memref<5xi64>
    memref.store %class_id, %self[%layout_slot] : memref<5xi64>
    memref.store %length, %self[%length_slot] : memref<5xi64>
    memref.store %capacity, %self[%capacity_slot] : memref<5xi64>
    memref.store %items_word, %self[%items_slot] : memref<5xi64>
    func.return %self : memref<5xi64>
  }

  // Borrowed view of the items array, derived at the point of use. The view's
  // SSA name is not an identity: identity is the handle, so two views of the
  // same handle name the same slots of the same entity.
  // Marked ly.runtime.interior_word so release placement pins the handle
  // across the view's uses; a plain private helper would leave the ownership
  // walk nothing to follow from the call.
  func.func private @__ly_tuple_items(%self: memref<5xi64>) -> memref<?xi64> attributes {ly.runtime.contract = "builtins.tuple", ly.runtime.interior_word, ly.runtime.primitive = "items_view"} {
    %capacity_slot = arith.constant 3 : index
    %items_slot = arith.constant 4 : index
    %handle_words = func.call @__ly_box_word_count() : () -> i64
    %capacity = memref.load %self[%capacity_slot] : memref<5xi64>
    %words = arith.muli %capacity, %handle_words : i64
    %base = memref.load %self[%items_slot] : memref<5xi64>
    %view = func.call @__ly_global_view_i64(%base, %words) : (i64, i64) -> memref<?xi64>
    func.return %view : memref<?xi64>
  }

  // The four allocating tuple producers, each the tuple-shaped half of a
  // `__ly_seq_fill_*` pairing: identical fill loop, different destination.
  // The fill helpers take a plain slot array and a length by value, so they
  // are shared with `list` and `set` untouched.
  func.func private @__ly_tuple_copy_alloc(%src_len: i64, %src_items: memref<?xi64>) -> memref<5xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.tuple"], ly.ownership.owned_results = [0]} {
    %self = func.call @__ly_tuple_alloc(%src_len) : (i64) -> memref<5xi64>
    %items = func.call @__ly_tuple_items(%self) : (memref<5xi64>) -> memref<?xi64>
    func.call @__ly_seq_fill_copy(%items, %src_len, %src_items) : (memref<?xi64>, i64, memref<?xi64>) -> ()
    func.return %self : memref<5xi64>
  }

  func.func private @__ly_tuple_concat_alloc(%llen: i64, %li: memref<?xi64>, %rlen: i64, %ri: memref<?xi64>) -> memref<5xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.tuple"], ly.ownership.owned_results = [0]} {
    %total = arith.addi %llen, %rlen : i64
    %self = func.call @__ly_tuple_alloc(%total) : (i64) -> memref<5xi64>
    %items = func.call @__ly_tuple_items(%self) : (memref<5xi64>) -> memref<?xi64>
    func.call @__ly_seq_fill_concat(%items, %llen, %li, %rlen, %ri) : (memref<?xi64>, i64, memref<?xi64>, i64, memref<?xi64>) -> ()
    func.return %self : memref<5xi64>
  }

  func.func private @__ly_tuple_repeat_alloc(%len: i64, %li: memref<?xi64>, %n: i64) -> memref<5xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.tuple"], ly.ownership.owned_results = [0]} {
    // CPython's tuple_repeat: a length past the word, or past the allocator's
    // reach, is MemoryError.
    %overflows = func.call @__ly_repeat_overflows(%len, %n) : (i64, i64) -> i1
    scf.if %overflows {
      func.call @LyErr_NoMemory() : () -> ()
    }
    %total = arith.muli %len, %n : i64
    %handle_words = func.call @__ly_box_word_count() : () -> i64
    %word_bytes = arith.constant 8 : i64
    %slot_bytes = arith.muli %handle_words, %word_bytes : i64
    %handle_prefix = arith.constant 40 : i64
    func.call @__ly_check_alloc_count(%total, %slot_bytes, %handle_prefix) : (i64, i64, i64) -> ()
    %self = func.call @__ly_tuple_alloc(%total) : (i64) -> memref<5xi64>
    %items = func.call @__ly_tuple_items(%self) : (memref<5xi64>) -> memref<?xi64>
    func.call @__ly_seq_fill_repeat(%items, %len, %li, %n) : (memref<?xi64>, i64, memref<?xi64>, i64) -> ()
    func.return %self : memref<5xi64>
  }

  func.func private @__ly_tuple_slice_alloc(%count: i64, %start: i64, %step: i64, %src_items: memref<?xi64>) -> memref<5xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.tuple"], ly.ownership.owned_results = [0]} {
    %self = func.call @__ly_tuple_alloc(%count) : (i64) -> memref<5xi64>
    %items = func.call @__ly_tuple_items(%self) : (memref<5xi64>) -> memref<?xi64>
    func.call @__ly_seq_fill_slice(%items, %count, %start, %step, %src_items) : (memref<?xi64>, i64, i64, i64, memref<?xi64>) -> ()
    func.return %self : memref<5xi64>
  }

  func.func @LyTuple_FromLength(%length: i64 {ly.runtime.default_i64 = 0 : i64}) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 11 : i64, ly.runtime.contract = "builtins.tuple", ly.runtime.initializer = "__new__", ly.runtime.result_contract = "builtins.tuple"} {
    %self = func.call @__ly_tuple_alloc(%length) : (i64) -> memref<5xi64>
    func.return %self : memref<5xi64>
  }

  func.func @LyTuple_Len(%self: memref<5xi64> {ly.ownership.object_header}) -> i64 attributes {ly.runtime.contract = "builtins.tuple", ly.runtime.method = "__len__"} {
    %length_slot = arith.constant 2 : index
    %length = memref.load %self[%length_slot] : memref<5xi64>
    func.return %length : i64
  }

  func.func @LyTuple_GetSlice(%self: memref<5xi64> {ly.ownership.object_header}, %start_raw: i64 {ly.runtime.clip_i64}, %stop_raw: i64 {ly.runtime.clip_i64}, %step_raw: i64 {ly.runtime.clip_i64}, %mask: i64) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.tuple", ly.runtime.method = "__getslice__", ly.runtime.result_contract = "builtins.tuple"} {
    %length_slot = arith.constant 2 : index
    %len = memref.load %self[%length_slot] : memref<5xi64>
    %b:3 = func.call @__ly_seq_slice_bounds(%len, %start_raw, %stop_raw, %step_raw, %mask) : (i64, i64, i64, i64, i64) -> (i64, i64, i64)
    %items = func.call @__ly_tuple_items(%self) : (memref<5xi64>) -> memref<?xi64>
    %result = func.call @__ly_tuple_slice_alloc(%b#1, %b#0, %b#2, %items) : (i64, i64, i64, memref<?xi64>) -> memref<5xi64>
    func.return %result : memref<5xi64>
  }

  // tuple.__repr__: `(e0, e1, ...)`; a single element gets a trailing comma
  // (`(1,)`), matching CPython. Same uniform element dispatch as LyList_Repr.
  func.func @LyTuple_Repr(%self: memref<5xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.tuple", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c16_i64 = func.call @__ly_box_word_count() : () -> i64
    %length_slot = arith.constant 2 : index
    %len = memref.load %self[%length_slot] : memref<5xi64>
    %items = func.call @__ly_tuple_items(%self) : (memref<5xi64>) -> memref<?xi64>
    %len_idx = arith.index_cast %len : i64 to index
    %items_idx = memref.extract_aligned_pointer_as_index %items : memref<?xi64> -> index
    %items_i64 = arith.index_cast %items_idx : index to i64
    %items_ptr = llvm.inttoptr %items_i64 : i64 to !llvm.ptr

    %open_ref = memref.get_global @__ly_repr_lparen : memref<1xi8>
    %open_dyn = memref.cast %open_ref : memref<1xi8> to memref<?xi8>
    %r0_h, %r0_b = func.call @__ly_unicode_from_valid_utf8(%open_dyn, %c0, %c1_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)

    %loop:2 = scf.for %i = %c0 to %len_idx step %c1 iter_args(%rh = %r0_h, %rb = %r0_b) -> (memref<2xi64>, memref<?xi8>) {
      %i_i64 = arith.index_cast %i : index to i64
      %is_pos = arith.cmpi sgt, %i_i64, %c0_i64 : i64
      %sep:2 = scf.if %is_pos -> (memref<2xi64>, memref<?xi8>) {
        %sep_ref = memref.get_global @__ly_repr_comma : memref<2xi8>
        %sep_dyn = memref.cast %sep_ref : memref<2xi8> to memref<?xi8>
        %sh, %sb = func.call @__ly_unicode_from_valid_utf8(%sep_dyn, %c0, %c2_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
        %ch, %cb = func.call @LyUnicode_Concat(%rh, %rb, %sh, %sb) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
        func.call @LyUnicode_DecRef(%rh) : (memref<2xi64>) -> ()
        func.call @LyUnicode_DecRef(%sh) : (memref<2xi64>) -> ()
        scf.yield %ch, %cb : memref<2xi64>, memref<?xi8>
      } else {
        scf.yield %rh, %rb : memref<2xi64>, memref<?xi8>
      }
      %off = arith.muli %i_i64, %c16_i64 : i64
      %box_ptr = llvm.getelementptr %items_ptr[%off] : (!llvm.ptr, i64) -> !llvm.ptr, i64
      %class_word = llvm.load %box_ptr : !llvm.ptr -> i64
      %class_id = func.call @__ly_slot_class(%class_word) : (i64) -> i64
      %erh, %erb = func.call @__ly_repr_boxed_or_default(%box_ptr, %class_id) : (!llvm.ptr, i64) -> (memref<2xi64>, memref<?xi8>)
      %nh, %nb = func.call @LyUnicode_Concat(%sep#0, %sep#1, %erh, %erb) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      func.call @LyUnicode_DecRef(%sep#0) : (memref<2xi64>) -> ()
      func.call @LyUnicode_DecRef(%erh) : (memref<2xi64>) -> ()
      scf.yield %nh, %nb : memref<2xi64>, memref<?xi8>
    }

    // Single-element tuple: trailing comma.
    %is_single = arith.cmpi eq, %len, %c1_i64 : i64
    %comma:2 = scf.if %is_single -> (memref<2xi64>, memref<?xi8>) {
      %comma_ref = memref.get_global @__ly_repr_comma : memref<2xi8>
      %comma_dyn = memref.cast %comma_ref : memref<2xi8> to memref<?xi8>
      %tc_h, %tc_b = func.call @__ly_unicode_from_valid_utf8(%comma_dyn, %c0, %c1_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
      %wh, %wb = func.call @LyUnicode_Concat(%loop#0, %loop#1, %tc_h, %tc_b) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      func.call @LyUnicode_DecRef(%loop#0) : (memref<2xi64>) -> ()
      func.call @LyUnicode_DecRef(%tc_h) : (memref<2xi64>) -> ()
      scf.yield %wh, %wb : memref<2xi64>, memref<?xi8>
    } else {
      scf.yield %loop#0, %loop#1 : memref<2xi64>, memref<?xi8>
    }

    %close_ref = memref.get_global @__ly_repr_rparen : memref<1xi8>
    %close_dyn = memref.cast %close_ref : memref<1xi8> to memref<?xi8>
    %clh, %clb = func.call @__ly_unicode_from_valid_utf8(%close_dyn, %c0, %c1_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %out_h, %out_b = func.call @LyUnicode_Concat(%comma#0, %comma#1, %clh, %clb) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.call @LyUnicode_DecRef(%comma#0) : (memref<2xi64>) -> ()
    func.call @LyUnicode_DecRef(%clh) : (memref<2xi64>) -> ()
    func.return %out_h, %out_b : memref<2xi64>, memref<?xi8>
  }

  func.func @LyTuple_DecRef(%self: memref<5xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.tuple", ly.runtime.deallocator} {
    %storage = memref.cast %self : memref<5xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    %length_slot = arith.constant 2 : index
    %items_slot = arith.constant 4 : index
    %lower = arith.constant 0 : index
    %step = arith.constant 1 : index
    %items = func.call @__ly_tuple_items(%self) : (memref<5xi64>) -> memref<?xi64>
    %length = memref.load %self[%length_slot] : memref<5xi64>
    %length_index = arith.index_cast %length : i64 to index
    scf.for %i = %lower to %length_index step %step {
      %logical_index = arith.index_cast %i : index to i64
      func.call @LyObject_ReleaseBoxedPayloadArraySlotRaw(%items, %logical_index) : (memref<?xi64>, i64) -> ()
    }
    // The items live in the handle's own block (`__ly_tuple_alloc`), so
    // freeing the handle frees them.
    memref.dealloc %self : memref<5xi64>
    cf.br ^done

  ^done:
    func.return
  }
}
