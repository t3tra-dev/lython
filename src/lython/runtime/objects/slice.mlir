// `slice` -- CPython's Objects/sliceobject.c: the slice object (its bounds,
// repr, comparisons, hash and indices()) and the index arithmetic every
// sequence's `__getslice__` shares (PySlice_Unpack + PySlice_AdjustIndices).
//
// Signature source (1:1 correspondence target):
//   https://github.com/python/typeshed/blob/main/stdlib/builtins.pyi
//
// Deviations from CPython:
// - The bounds are `int | None`, where CPython keeps any objects (typeshed's
//   `slice[_StartT, _StopT, _StepT]`): a sequence reads nothing else, and a
//   bound read back from the object needs one static type. A bound of another
//   type is refused where the slice is made, and `slice` takes no type
//   arguments.
// - A bool bound is refused there too (`ly.typing.keeps_arguments`): the slice
//   keeps what it is given and reads it back as an int, and a bool is not one
//   in this compiler's representation -- `slice(True).stop` would print 1.

module attributes {
  ly.typing.manifest,
  ly.runtime.contracts = ["builtins.slice"]
} {
  // ===== declared here, defined in another runtime file or built by the lowering =====
  func.func private @LyLong_Add(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_header: memref<2xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__add__"}
  func.func private @LyLong_Compare(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_header: memref<2xi64> {ly.ownership.object_header}) -> i64 attributes {ly.runtime.contract = "builtins.int", ly.runtime.method = "__richcompare__"}
  func.func private @LyLong_DecRef(%header: memref<2xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.int", ly.runtime.deallocator}
  func.func private @LyLong_SlotWordTakingRef(%header: memref<2xi64> {ly.ownership.object_header}) -> i64 attributes {ly.runtime.contract = "builtins.int", ly.runtime.primitive = "slot_word_taking_ref"}
  func.func private @LyLong_Sub(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_header: memref<2xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__sub__"}
  func.func private @LyLong_TryAsI64(%header: memref<2xi64> {ly.ownership.object_header}) -> (i64, i1) attributes {ly.runtime.contract = "builtins.int", ly.runtime.primitive = "try_unbox.i64"}
  func.func private @__ly_sequence_compare_op(%lhs_len: i64, %lhs_items: memref<?xi64>, %rhs_len: i64, %rhs_items: memref<?xi64>, %op: i64) -> i1
  func.func private @LyLong_AsI64Clipped(%header: memref<2xi64> {ly.ownership.object_header}) -> i64 attributes {ly.runtime.contract = "builtins.int", ly.runtime.primitive = "unbox.i64.clip"}
  func.func private @LyLong_DeferredStandIn() -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.primitive = "deferred_stand_in"}
  func.func private @LyLong_FromSlotWord(%slot_view: memref<2xi64>) -> memref<2xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.primitive = "from_slot_word"}
  func.func private @LyLong_SlotWordFromI64(%value: i64) -> i64 attributes {ly.runtime.contract = "builtins.int", ly.runtime.primitive = "slot_word_from_i64"}
  func.func private @LyObject_ReleaseBoxedPayloadArraySlotRaw(%payload: memref<?xi64>, %logical_index: i64)
  func.func private @LyObject_ReleaseStorageToZero(%storage: memref<?xi64>) -> i1 attributes {ly.runtime.contract = "builtins.object", ly.runtime.primitive = "release_to_zero"}
  func.func private @LyUnicode_DecRef(%header: memref<2xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.str", ly.runtime.deallocator}
  func.func private @__ly_box_equal(%lhs: !llvm.ptr, %rhs: !llvm.ptr) -> i1
  func.func private @__ly_box_store_entity(%items: memref<?xi64>, %slot: i64, %class_id: i64, %entity: i64)
  func.func private @__ly_global_view_i64(%pointer: i64, %size: i64) -> memref<?xi64>
  func.func private @__ly_handle_retain_raw(%entity: i64)
  func.func private @__ly_int_from_immediate(%word: i64) -> i64
  func.func private @__ly_repr_boxed_or_default(%box_ptr: !llvm.ptr, %class_id: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.object", ly.runtime.primitive = "repr_boxed_or_default", ly.runtime.result_contract = "builtins.str"}
  memref.global "private" constant @__ly_repr_comma : memref<2xi8>
  memref.global "private" constant @__ly_repr_rparen : memref<1xi8>
  func.func private @__ly_slot_class(%word: i64) -> i64
  func.func private @__ly_slot_word_is_immediate(%word: i64) -> i1
  func.func private @__ly_tuple_alloc(%length: i64) -> memref<5xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.tuple"], ly.ownership.owned_results = [0]}
  func.func private @__ly_tuple_items(%self: memref<5xi64>) -> memref<?xi64> attributes {ly.runtime.contract = "builtins.tuple", ly.runtime.interior_word, ly.runtime.primitive = "items_view"}
  func.func private @__ly_xxhash_slot_lanes(%items: !llvm.ptr, %count: i64) -> i64
  func.func private @__ly_unicode_from_valid_utf8(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]}
  func.func private @LyBaseException_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.BaseException", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"}
  func.func private @LyBaseException_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 5 : i64, ly.runtime.contract = "builtins.BaseException", ly.runtime.initializer = "__new__"}
  func.func private @LyEH_ThrowException(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "raise"}
  func.func private @LyLong_FromI64(%value: i64 {ly.runtime.default_i64 = 0 : i64}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 1 : i64, ly.runtime.contract = "builtins.int", ly.runtime.initializer = "__new__"}
  func.func private @LyLong_Str(%header: memref<2xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.int", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"}
  func.func private @LyUnicode_Concat(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_bytes: memref<?xi8>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__add__"}
  func.func private @__ly_raise_static_message(%class_id: i64, %message: memref<?xi8>, %length: i64)

  // The bounds are `int | None`: typeshed's `slice[_StartT, _StopT, _StepT]`
  // is generic over any objects, which a slice of a sequence never holds.
  // ⛔ The members are written in the order a union normalizes to (int before
  // None, by spelling): `LySlice_Field` hands back a tag that is an index into
  // this list.
  py.class @slice attributes {
    base_names = ["object"], ly.typing.final, ly.typing.keeps_arguments,
    field_names = ["start", "stop", "step"],
    field_contract_types = [!py.union<!py.contract<"builtins.int">, !py.literal<None>>, !py.union<!py.contract<"builtins.int">, !py.literal<None>>, !py.union<!py.contract<"builtins.int">, !py.literal<None>>],
    method_names = ["__new__", "__new__", "__new__", "__init__", "__init__",
                    "__init__", "indices", "__eq__", "__ne__", "__hash__",
                    "__repr__", "__lt__", "__le__", "__gt__", "__ge__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.slice">>, !py.union<!py.contract<"builtins.int">, !py.literal<None>>] -> [!py.self]>,
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.slice">>, !py.union<!py.contract<"builtins.int">, !py.literal<None>>, !py.union<!py.contract<"builtins.int">, !py.literal<None>>] -> [!py.self]>,
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.slice">>, !py.union<!py.contract<"builtins.int">, !py.literal<None>>, !py.union<!py.contract<"builtins.int">, !py.literal<None>>, !py.union<!py.contract<"builtins.int">, !py.literal<None>>] -> [!py.self]>,
      !py.protocol<"Callable", [!py.contract<"builtins.slice">, !py.union<!py.contract<"builtins.int">, !py.literal<None>>] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.slice">, !py.union<!py.contract<"builtins.int">, !py.literal<None>>, !py.union<!py.contract<"builtins.int">, !py.literal<None>>] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.slice">, !py.union<!py.contract<"builtins.int">, !py.literal<None>>, !py.union<!py.contract<"builtins.int">, !py.literal<None>>, !py.union<!py.contract<"builtins.int">, !py.literal<None>>] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.slice">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.tuple", [!py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.slice">, !py.contract<"builtins.slice">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.slice">, !py.contract<"builtins.slice">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.slice">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.slice">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.slice">, !py.contract<"builtins.slice">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.slice">, !py.contract<"builtins.slice">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.slice">, !py.contract<"builtins.slice">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.slice">, !py.contract<"builtins.slice">] -> [!py.contract<"builtins.bool">]>
    ],
    method_kinds = ["classmethod", "classmethod", "classmethod", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance", "instance"]
  } {}

  memref.global "private" constant @__ly_slice_msg_zero_step : memref<25xi8> = dense<[115, 108, 105, 99, 101, 32, 115, 116, 101, 112, 32, 99, 97, 110, 110, 111, 116, 32, 98, 101, 32, 122, 101, 114, 111]>

  // ValueError("slice step cannot be zero") -- shared by every __getslice__.
  func.func private @__ly_slice_raise_zero_step() {
    %class_id = arith.constant 53 : i64
    %length = arith.constant 25 : i64
    %message_static = memref.get_global @__ly_slice_msg_zero_step : memref<25xi8>
    %message = memref.cast %message_static : memref<25xi8> to memref<?xi8>
    func.call @__ly_raise_static_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  // CPython PySlice_Unpack + the clamping half of PySlice_AdjustIndices over
  // a length: absent bounds (mask bit0 = start present, bit1 = stop present)
  // default by the step's sign, explicit bounds normalize (+len) and clamp
  // into the window the sign allows. Returns (start, stop) -- the pair
  // _PySlice_GetLongIndices answers with exact ints, which `slice.indices`
  // and a range's slice need, where a sequence copy needs the count instead.
  func.func private @__ly_slice_indices(%len: i64, %start_in: i64, %stop_in: i64, %step: i64, %mask: i64) -> (i64, i64) {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %minus_one = arith.constant -1 : i64
    %neg_step = arith.cmpi slt, %step, %zero : i64
    %len_minus_one = arith.subi %len, %one : i64
    %neg_floor = arith.select %neg_step, %minus_one, %zero : i1, i64
    %hi_clamp = arith.select %neg_step, %len_minus_one, %len : i1, i64

    %has_start_bit = arith.andi %mask, %one : i64
    %has_start = arith.cmpi ne, %has_start_bit, %zero : i64
    %s_plus = arith.addi %start_in, %len : i64
    %s_isneg = arith.cmpi slt, %start_in, %zero : i64
    %s_plus_isneg = arith.cmpi slt, %s_plus, %zero : i64
    %s_neg_val = arith.select %s_plus_isneg, %neg_floor, %s_plus : i1, i64
    %s_big = arith.cmpi sge, %start_in, %len : i64
    %s_pos_val = arith.select %s_big, %hi_clamp, %start_in : i1, i64
    %s_adj = arith.select %s_isneg, %s_neg_val, %s_pos_val : i1, i64
    %s_default = arith.select %neg_step, %len_minus_one, %zero : i1, i64
    %start = arith.select %has_start, %s_adj, %s_default : i1, i64

    %has_stop_bit = arith.andi %mask, %two : i64
    %has_stop = arith.cmpi ne, %has_stop_bit, %zero : i64
    %e_plus = arith.addi %stop_in, %len : i64
    %e_isneg = arith.cmpi slt, %stop_in, %zero : i64
    %e_plus_isneg = arith.cmpi slt, %e_plus, %zero : i64
    %e_neg_val = arith.select %e_plus_isneg, %neg_floor, %e_plus : i1, i64
    %e_big = arith.cmpi sge, %stop_in, %len : i64
    %e_pos_val = arith.select %e_big, %hi_clamp, %stop_in : i1, i64
    %e_adj = arith.select %e_isneg, %e_neg_val, %e_pos_val : i1, i64
    %e_default = arith.select %neg_step, %minus_one, %len : i1, i64
    %stop = arith.select %has_stop, %e_adj, %e_default : i1, i64
    func.return %start, %stop : i64, i64
  }

  // CPython PySlice_Unpack + PySlice_AdjustIndices over a length. Returns
  // (start, slicelength); the caller iterates start, start+step, ...
  // slicelength times.
  func.func private @__ly_slice_adjust(%len: i64, %start_in: i64, %stop_in: i64, %step: i64, %mask: i64) -> (i64, i64) {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %neg_step = arith.cmpi slt, %step, %zero : i64
    %start, %stop = func.call @__ly_slice_indices(%len, %start_in, %stop_in, %step, %mask) : (i64, i64, i64, i64, i64) -> (i64, i64)

    %pos_diff = arith.subi %stop, %start : i64
    %neg_diff = arith.subi %start, %stop : i64
    %pos_nonempty = arith.cmpi slt, %start, %stop : i64
    %neg_nonempty = arith.cmpi slt, %stop, %start : i64
    %neg_step_abs = arith.subi %zero, %step : i64
    %step_abs = arith.select %neg_step, %neg_step_abs, %step : i1, i64
    %diff = arith.select %neg_step, %neg_diff, %pos_diff : i1, i64
    %diff_m1 = arith.subi %diff, %one : i64
    %quot = arith.divsi %diff_m1, %step_abs : i64
    %count_raw = arith.addi %quot, %one : i64
    %nonempty = arith.select %neg_step, %neg_nonempty, %pos_nonempty : i1, i1
    %count = arith.select %nonempty, %count_raw, %zero : i1, i64
    func.return %start, %count : i64, i64
  }

  // " to extended slice of size "
  memref.global "private" constant @__ly_slice_msg_extended_middle : memref<27xi8> = dense<[32, 116, 111, 32, 101, 120, 116, 101, 110, 100, 101, 100, 32, 115, 108, 105, 99, 101, 32, 111, 102, 32, 115, 105, 122, 101, 32]>

  // ValueError for `a[i:j:k] = xs` with k != 1 and len(xs) != slicelength:
  // `prefix` + the given size + " to extended slice of size " + the slice's,
  // the prefix naming what was assigned -- "attempt to assign sequence of
  // size " for list_ass_subscript, "... bytes of size " for bytearray's.
  func.func private @__ly_slice_raise_extended_mismatch(%prefix: memref<?xi8>, %prefix_len: i64, %src_len: i64, %slice_len: i64) {
    %class_id = arith.constant 53 : i64
    %start = arith.constant 0 : index
    %middle_static = memref.get_global @__ly_slice_msg_extended_middle : memref<27xi8>
    %middle = memref.cast %middle_static : memref<27xi8> to memref<?xi8>
    %middle_len = arith.constant 27 : i64
    %ph, %pb = func.call @__ly_unicode_from_valid_utf8(%prefix, %start, %prefix_len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %src_int = func.call @LyLong_FromI64(%src_len) : (i64) -> memref<2xi64>
    %src_str:2 = func.call @LyLong_Str(%src_int) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi8>)
    %m1:2 = func.call @LyUnicode_Concat(%ph, %pb, %src_str#0, %src_str#1) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    %mh, %mb = func.call @__ly_unicode_from_valid_utf8(%middle, %start, %middle_len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %m2:2 = func.call @LyUnicode_Concat(%m1#0, %m1#1, %mh, %mb) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    %cnt_int = func.call @LyLong_FromI64(%slice_len) : (i64) -> memref<2xi64>
    %cnt_str:2 = func.call @LyLong_Str(%cnt_int) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi8>)
    %m3:2 = func.call @LyUnicode_Concat(%m2#0, %m2#1, %cnt_str#0, %cnt_str#1) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    %exception:3 = func.call @LyBaseException_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    %initialized:3 = func.call @LyBaseException_Init(%exception#0, %exception#1, %exception#2, %m3#0, %m3#1) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.call @LyEH_ThrowException(%initialized#0, %initialized#1, %initialized#2) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }
  // ===== the slice object (sliceobject.c's PySliceObject) =====
  //
  // One entity, five words: the refcount, class 25, and start/stop/step as
  // the slot words a container element keeps -- 0 for None, an int's
  // immediate, or the address of an int object the slice holds a reference
  // to.
  func.func private @LySlice_Shape() -> memref<5xi64> attributes {ly.runtime.contract = "builtins.slice", ly.runtime.shape}

  // The three words as an items array, the shape the slot helpers take.
  func.func private @__ly_slice_items(%self: memref<5xi64>) -> memref<?xi64> {
    %three = arith.constant 3 : i64
    %first_word = arith.constant 16 : i64
    %self_index = memref.extract_aligned_pointer_as_index %self : memref<5xi64> -> index
    %self_word = arith.index_cast %self_index : index to i64
    %items_word = arith.addi %self_word, %first_word : i64
    %items = func.call @__ly_global_view_i64(%items_word, %three) : (i64, i64) -> memref<?xi64>
    func.return %items : memref<?xi64>
  }

  // A pointer to bound `which`'s word, the shape the box helpers take.
  func.func private @__ly_slice_word_ptr(%self: memref<5xi64>, %which: i64) -> !llvm.ptr {
    %two = arith.constant 2 : i64
    %self_index = memref.extract_aligned_pointer_as_index %self : memref<5xi64> -> index
    %self_word = arith.index_cast %self_index : index to i64
    %base = llvm.inttoptr %self_word : i64 to !llvm.ptr
    %slot = arith.addi %which, %two : i64
    %ptr = llvm.getelementptr %base[%slot] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    func.return %ptr : !llvm.ptr
  }

  // The word a slice keeps for an argument handed over in a box: the box's
  // entity -- 0 for None, an int's immediate or its object -- with a
  // reference of its own, as slice_new keeps each object it is given.
  func.func private @__ly_slice_word_of(%box: memref<?xi64>) -> i64 {
    %entity_slot = arith.constant 2 : index
    %entity = memref.load %box[%entity_slot] : memref<?xi64>
    func.call @__ly_handle_retain_raw(%entity) : (i64) -> ()
    func.return %entity : i64
  }

  func.func private @__ly_slice_alloc(%start: i64, %stop: i64, %step: i64) -> memref<5xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.slice"], ly.ownership.owned_results = [0]} {
    %one = arith.constant 1 : i64
    %block_bytes = arith.constant 40 : index
    %block = memref.alloc(%block_bytes) {alignment = 16 : i64} : memref<?xi8>
    %self_offset = arith.constant 0 : index
    %self = memref.view %block[%self_offset][] {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<?xi8> to memref<5xi64>
    %class_slice = arith.constant 25 : i64
    %refcount_slot = arith.constant 0 : index
    %class_slot = arith.constant 1 : index
    %start_slot = arith.constant 2 : index
    %stop_slot = arith.constant 3 : index
    %step_slot = arith.constant 4 : index
    memref.store %one, %self[%refcount_slot] : memref<5xi64>
    memref.store %class_slice, %self[%class_slot] : memref<5xi64>
    memref.store %start, %self[%start_slot] : memref<5xi64>
    memref.store %stop, %self[%stop_slot] : memref<5xi64>
    memref.store %step, %self[%step_slot] : memref<5xi64>
    func.return %self : memref<5xi64>
  }

  // slice(stop), slice(start, stop), slice(start, stop, step): one
  // initializer per arity, as slice_new reads one to three arguments; an
  // absent start or step is None.
  func.func @LySlice_New(%stop: memref<?xi64>) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 25 : i64, ly.runtime.contract = "builtins.slice", ly.runtime.initializer = "__new__"} {
    %none = arith.constant 0 : i64
    %stop_word = func.call @__ly_slice_word_of(%stop) : (memref<?xi64>) -> i64
    %self = func.call @__ly_slice_alloc(%none, %stop_word, %none) : (i64, i64, i64) -> memref<5xi64>
    func.return %self : memref<5xi64>
  }

  func.func @LySlice_NewStart(%start: memref<?xi64>, %stop: memref<?xi64>) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 25 : i64, ly.runtime.contract = "builtins.slice", ly.runtime.initializer = "__new__"} {
    %none = arith.constant 0 : i64
    %start_word = func.call @__ly_slice_word_of(%start) : (memref<?xi64>) -> i64
    %stop_word = func.call @__ly_slice_word_of(%stop) : (memref<?xi64>) -> i64
    %self = func.call @__ly_slice_alloc(%start_word, %stop_word, %none) : (i64, i64, i64) -> memref<5xi64>
    func.return %self : memref<5xi64>
  }

  func.func @LySlice_NewStep(%start: memref<?xi64>, %stop: memref<?xi64>, %step: memref<?xi64>) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 25 : i64, ly.runtime.contract = "builtins.slice", ly.runtime.initializer = "__new__"} {
    %start_word = func.call @__ly_slice_word_of(%start) : (memref<?xi64>) -> i64
    %stop_word = func.call @__ly_slice_word_of(%stop) : (memref<?xi64>) -> i64
    %step_word = func.call @__ly_slice_word_of(%step) : (memref<?xi64>) -> i64
    %self = func.call @__ly_slice_alloc(%start_word, %stop_word, %step_word) : (i64, i64, i64) -> memref<5xi64>
    func.return %self : memref<5xi64>
  }

  func.func @LySlice_Init(%self: memref<5xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "builtins.slice", ly.runtime.method = "__init__"} {
    func.return
  }

  // slice.start / .stop / .step (`which` 0/1/2), as the `int | None` they
  // are: the union's tag -- 0 an int, 1 None -- and its int lane, which for
  // None is the immortal stand-in every inactive int lane holds.
  func.func @LySlice_Field(%self: memref<5xi64> {ly.ownership.object_header}, %which: i64) -> (i64, memref<2xi64>) attributes {ly.ownership.owned_result_contracts = ["builtins.int"], ly.ownership.owned_results = [1], ly.runtime.contract = "builtins.slice", ly.runtime.primitive = "field"} {
    %zero = arith.constant 0 : i64
    %int_tag = arith.constant 0 : i64
    %none_tag = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %ptr = func.call @__ly_slice_word_ptr(%self, %which) : (memref<5xi64>, i64) -> !llvm.ptr
    %word = llvm.load %ptr : !llvm.ptr -> i64
    %is_none = arith.cmpi eq, %word, %zero : i64
    %tag = arith.select %is_none, %none_tag, %int_tag : i64
    %value = scf.if %is_none -> (memref<2xi64>) {
      %stand_in = func.call @LyLong_DeferredStandIn() : () -> memref<2xi64>
      scf.yield %stand_in : memref<2xi64>
    } else {
      // A slot view's aligned pointer IS the slot's word (what
      // LyLong_FromSlotWord reads it as), not the word's address.
      %view = func.call @__ly_global_view_i64(%word, %two) : (i64, i64) -> memref<?xi64>
      %slot_view = memref.cast %view : memref<?xi64> to memref<2xi64>
      %header = func.call @LyLong_FromSlotWord(%slot_view) : (memref<2xi64>) -> memref<2xi64>
      scf.yield %header : memref<2xi64>
    }
    func.return %tag, %value : i64, memref<2xi64>
  }

  // The word index a bound names: its value, or the nearest end of the word
  // for an int past it (_PyEval_SliceIndex).
  func.func private @__ly_slice_word_index(%word: i64) -> i64 {
    %two = arith.constant 2 : i64
    %immediate = func.call @__ly_slot_word_is_immediate(%word) : (i64) -> i1
    %index = scf.if %immediate -> (i64) {
      %v = func.call @__ly_int_from_immediate(%word) : (i64) -> i64
      scf.yield %v : i64
    } else {
      %view = func.call @__ly_global_view_i64(%word, %two) : (i64, i64) -> memref<?xi64>
      %header = memref.cast %view : memref<?xi64> to memref<2xi64>
      %v = func.call @LyLong_AsI64Clipped(%header) : (memref<2xi64>) -> i64
      scf.yield %v : i64
    }
    func.return %index : i64
  }

  // CPython PySlice_Unpack: (start, stop, step, mask) as a written slice
  // spells them -- an absent bound cleared from the mask (bit0 start, bit1
  // stop), a missing step 1 -- with each int clipped to the word, a zero step
  // refused, and the most negative step raised to -INT64_MAX so that `-step`
  // stays a word.
  func.func private @__ly_slice_unpack(%self: memref<5xi64>) -> (i64, i64, i64, i64) {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %min = arith.constant -9223372036854775808 : i64
    %min_plus_one = arith.constant -9223372036854775807 : i64
    %start_slot = arith.constant 2 : index
    %stop_slot = arith.constant 3 : index
    %step_slot = arith.constant 4 : index
    %start_word = memref.load %self[%start_slot] : memref<5xi64>
    %stop_word = memref.load %self[%stop_slot] : memref<5xi64>
    %step_word = memref.load %self[%step_slot] : memref<5xi64>
    %has_start = arith.cmpi ne, %start_word, %zero : i64
    %has_stop = arith.cmpi ne, %stop_word, %zero : i64
    %has_step = arith.cmpi ne, %step_word, %zero : i64
    %start = scf.if %has_start -> (i64) {
      %v = func.call @__ly_slice_word_index(%start_word) : (i64) -> i64
      scf.yield %v : i64
    } else {
      scf.yield %zero : i64
    }
    %stop = scf.if %has_stop -> (i64) {
      %v = func.call @__ly_slice_word_index(%stop_word) : (i64) -> i64
      scf.yield %v : i64
    } else {
      scf.yield %zero : i64
    }
    %step_raw = scf.if %has_step -> (i64) {
      %v = func.call @__ly_slice_word_index(%step_word) : (i64) -> i64
      scf.yield %v : i64
    } else {
      scf.yield %one : i64
    }
    %step_zero = arith.cmpi eq, %step_raw, %zero : i64
    scf.if %step_zero {
      func.call @__ly_slice_raise_zero_step() : () -> ()
    }
    %step_min = arith.cmpi eq, %step_raw, %min : i64
    %step = arith.select %step_min, %min_plus_one, %step_raw : i64
    %start_bit = arith.select %has_start, %one, %zero : i64
    %stop_bit = arith.select %has_stop, %two, %zero : i64
    %mask = arith.ori %start_bit, %stop_bit : i64
    func.return %start, %stop, %step, %mask : i64, i64, i64, i64
  }

  // "length should not be negative"
  memref.global "private" constant @__ly_slice_msg_negative_length : memref<29xi8> = dense<[108, 101, 110, 103, 116, 104, 32, 115, 104, 111, 117, 108, 100, 32, 110, 111, 116, 32, 98, 101, 32, 110, 101, 103, 97, 116, 105, 118, 101]>

  // slice.indices(length): CPython's slice_indices (_PySlice_GetLongIndices)
  // -- (start, stop, step) the way a sequence of that length reads the slice.
  // The step comes back as the slice holds it, not clipped to the word, and a
  // length past the word is computed in Python ints, as CPython computes
  // every length.
  func.func @LySlice_Indices(%self: memref<5xi64> {ly.ownership.object_header}, %length: memref<2xi64> {ly.ownership.object_header}) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.slice", ly.runtime.method = "indices", ly.runtime.result_contract = "builtins.tuple"} {
    %zero = arith.constant 0 : i64
    %three = arith.constant 3 : i64
    %int_class = arith.constant 1 : i64
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c2 = arith.constant 2 : i64
    %step_slot = arith.constant 4 : index
    %zero_h = func.call @LyLong_FromI64(%zero) : (i64) -> memref<2xi64>
    %sign = func.call @LyLong_Compare(%length, %zero_h) : (memref<2xi64>, memref<2xi64>) -> i64
    func.call @LyLong_DecRef(%zero_h) : (memref<2xi64>) -> ()
    %negative = arith.cmpi slt, %sign, %zero : i64
    scf.if %negative {
      %class_id = arith.constant 53 : i64
      %message_len = arith.constant 29 : i64
      %message_static = memref.get_global @__ly_slice_msg_negative_length : memref<29xi8>
      %message = memref.cast %message_static : memref<29xi8> to memref<?xi8>
      func.call @__ly_raise_static_message(%class_id, %message, %message_len) : (i64, memref<?xi8>, i64) -> ()
    }
    %start_raw, %stop_raw, %step, %mask = func.call @__ly_slice_unpack(%self) : (memref<5xi64>) -> (i64, i64, i64, i64)
    %step_held = memref.load %self[%step_slot] : memref<5xi64>
    %has_step = arith.cmpi ne, %step_held, %zero : i64
    %step_word = scf.if %has_step -> (i64) {
      func.call @__ly_handle_retain_raw(%step_held) : (i64) -> ()
      scf.yield %step_held : i64
    } else {
      %w = func.call @LyLong_SlotWordFromI64(%c1) : (i64) -> i64
      scf.yield %w : i64
    }
    %small, %fits = func.call @LyLong_TryAsI64(%length) : (memref<2xi64>) -> (i64, i1)
    %start_word, %stop_word = scf.if %fits -> (i64, i64) {
      %start, %stop = func.call @__ly_slice_indices(%small, %start_raw, %stop_raw, %step, %mask) : (i64, i64, i64, i64, i64) -> (i64, i64)
      %sw = func.call @LyLong_SlotWordFromI64(%start) : (i64) -> i64
      %tw = func.call @LyLong_SlotWordFromI64(%stop) : (i64) -> i64
      scf.yield %sw, %tw : i64, i64
    } else {
      %sw, %tw = func.call @__ly_slice_long_indices(%self, %length, %step) : (memref<5xi64>, memref<2xi64>, i64) -> (i64, i64)
      scf.yield %sw, %tw : i64, i64
    }
    %result = func.call @__ly_tuple_alloc(%three) : (i64) -> memref<5xi64>
    %items = func.call @__ly_tuple_items(%result) : (memref<5xi64>) -> memref<?xi64>
    func.call @__ly_box_store_entity(%items, %c0, %int_class, %start_word) : (memref<?xi64>, i64, i64, i64) -> ()
    func.call @__ly_box_store_entity(%items, %c1, %int_class, %stop_word) : (memref<?xi64>, i64, i64, i64) -> ()
    func.call @__ly_box_store_entity(%items, %c2, %int_class, %step_word) : (memref<?xi64>, i64, i64, i64) -> ()
    func.return %result : memref<5xi64>
  }

  // _PySlice_GetLongIndices for a length past the word: the bounds in
  // [lower, upper] -- [0, length], or [-1, length - 1] for a negative step --
  // as owned slot words.
  // ⛔ A contract on a private helper: it moves references by hand, and the
  // release insertion would add its own to a function it treats as untrusted.
  func.func private @__ly_slice_long_indices(%self: memref<5xi64>, %length: memref<2xi64>, %step: i64) -> (i64, i64) attributes {ly.runtime.contract = "builtins.slice"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %minus_one = arith.constant -1 : i64
    %start_slot = arith.constant 2 : index
    %stop_slot = arith.constant 3 : index
    %negative_step = arith.cmpi slt, %step, %zero : i64
    %lower_value = arith.select %negative_step, %minus_one, %zero : i64
    %delta = arith.select %negative_step, %one, %zero : i64
    %lower_h = func.call @LyLong_FromI64(%lower_value) : (i64) -> memref<2xi64>
    %delta_h = func.call @LyLong_FromI64(%delta) : (i64) -> memref<2xi64>
    %upper_h = func.call @LyLong_Sub(%length, %delta_h) : (memref<2xi64>, memref<2xi64>) -> memref<2xi64>
    func.call @LyLong_DecRef(%delta_h) : (memref<2xi64>) -> ()
    %start_word = memref.load %self[%start_slot] : memref<5xi64>
    %stop_word = memref.load %self[%stop_slot] : memref<5xi64>
    %start = func.call @__ly_slice_long_bound(%start_word, %length, %lower_h, %upper_h, %negative_step) : (i64, memref<2xi64>, memref<2xi64>, memref<2xi64>, i1) -> i64
    %true = arith.constant true
    %positive_step = arith.xori %negative_step, %true : i1
    %stop = func.call @__ly_slice_long_bound(%stop_word, %length, %lower_h, %upper_h, %positive_step) : (i64, memref<2xi64>, memref<2xi64>, memref<2xi64>, i1) -> i64
    func.call @LyLong_DecRef(%lower_h) : (memref<2xi64>) -> ()
    func.call @LyLong_DecRef(%upper_h) : (memref<2xi64>) -> ()
    func.return %start, %stop : i64, i64
  }

  // One bound of _PySlice_GetLongIndices as an owned slot word: None is
  // `upper` when `none_is_upper`, else `lower`; a negative bound counts from
  // the end and stops at `lower`; any other stops at `upper`.
  func.func private @__ly_slice_long_bound(%word: i64, %length: memref<2xi64>, %lower: memref<2xi64>, %upper: memref<2xi64>, %none_is_upper: i1) -> i64 attributes {ly.runtime.contract = "builtins.slice"} {
    %zero = arith.constant 0 : i64
    %two = arith.constant 2 : i64
    %zero_h = func.call @LyLong_FromI64(%zero) : (i64) -> memref<2xi64>
    %is_none = arith.cmpi eq, %word, %zero : i64
    %bound = scf.if %is_none -> (memref<2xi64>) {
      %chosen = arith.select %none_is_upper, %upper, %lower : memref<2xi64>
      %copy = func.call @LyLong_Add(%chosen, %zero_h) : (memref<2xi64>, memref<2xi64>) -> memref<2xi64>
      scf.yield %copy : memref<2xi64>
    } else {
      %view = func.call @__ly_global_view_i64(%word, %two) : (i64, i64) -> memref<?xi64>
      %slot_view = memref.cast %view : memref<?xi64> to memref<2xi64>
      %given = func.call @LyLong_FromSlotWord(%slot_view) : (memref<2xi64>) -> memref<2xi64>
      %sign = func.call @LyLong_Compare(%given, %zero_h) : (memref<2xi64>, memref<2xi64>) -> i64
      %negative = arith.cmpi slt, %sign, %zero : i64
      %clamped = scf.if %negative -> (memref<2xi64>) {
        %from_end = func.call @LyLong_Add(%given, %length) : (memref<2xi64>, memref<2xi64>) -> memref<2xi64>
        func.call @LyLong_DecRef(%given) : (memref<2xi64>) -> ()
        %below = func.call @LyLong_Compare(%from_end, %lower) : (memref<2xi64>, memref<2xi64>) -> i64
        %is_below = arith.cmpi slt, %below, %zero : i64
        %r = scf.if %is_below -> (memref<2xi64>) {
          func.call @LyLong_DecRef(%from_end) : (memref<2xi64>) -> ()
          %copy = func.call @LyLong_Add(%lower, %zero_h) : (memref<2xi64>, memref<2xi64>) -> memref<2xi64>
          scf.yield %copy : memref<2xi64>
        } else {
          scf.yield %from_end : memref<2xi64>
        }
        scf.yield %r : memref<2xi64>
      } else {
        %above = func.call @LyLong_Compare(%given, %upper) : (memref<2xi64>, memref<2xi64>) -> i64
        %is_above = arith.cmpi sgt, %above, %zero : i64
        %r = scf.if %is_above -> (memref<2xi64>) {
          func.call @LyLong_DecRef(%given) : (memref<2xi64>) -> ()
          %copy = func.call @LyLong_Add(%upper, %zero_h) : (memref<2xi64>, memref<2xi64>) -> memref<2xi64>
          scf.yield %copy : memref<2xi64>
        } else {
          scf.yield %given : memref<2xi64>
        }
        scf.yield %r : memref<2xi64>
      }
      scf.yield %clamped : memref<2xi64>
    }
    func.call @LyLong_DecRef(%zero_h) : (memref<2xi64>) -> ()
    %result = func.call @LyLong_SlotWordTakingRef(%bound) : (memref<2xi64>) -> i64
    func.return %result : i64
  }

  // "slice(" / ", " / ")"
  memref.global "private" constant @__ly_slice_repr_open : memref<6xi8> = dense<[115, 108, 105, 99, 101, 40]>

  // repr(slice): "slice(start, stop, step)" with each bound's repr.
  func.func @LySlice_Repr(%self: memref<5xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.slice", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %six = arith.constant 6 : i64
    %open_ref = memref.get_global @__ly_slice_repr_open : memref<6xi8>
    %open = memref.cast %open_ref : memref<6xi8> to memref<?xi8>
    %comma_ref = memref.get_global @__ly_repr_comma : memref<2xi8>
    %comma = memref.cast %comma_ref : memref<2xi8> to memref<?xi8>
    %close_ref = memref.get_global @__ly_repr_rparen : memref<1xi8>
    %close = memref.cast %close_ref : memref<1xi8> to memref<?xi8>
    %head_h, %head_b = func.call @__ly_unicode_from_valid_utf8(%open, %c0, %six) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %a_h, %a_b = func.call @__ly_slice_append_bound(%head_h, %head_b, %self, %zero) : (memref<2xi64>, memref<?xi8>, memref<5xi64>, i64) -> (memref<2xi64>, memref<?xi8>)
    %b_h, %b_b = func.call @__ly_slice_append_text(%a_h, %a_b, %comma, %two) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    %c_h, %c_b = func.call @__ly_slice_append_bound(%b_h, %b_b, %self, %one) : (memref<2xi64>, memref<?xi8>, memref<5xi64>, i64) -> (memref<2xi64>, memref<?xi8>)
    %d_h, %d_b = func.call @__ly_slice_append_text(%c_h, %c_b, %comma, %two) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    %e_h, %e_b = func.call @__ly_slice_append_bound(%d_h, %d_b, %self, %two) : (memref<2xi64>, memref<?xi8>, memref<5xi64>, i64) -> (memref<2xi64>, memref<?xi8>)
    %f_h, %f_b = func.call @__ly_slice_append_text(%e_h, %e_b, %close, %one) : (memref<2xi64>, memref<?xi8>, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %f_h, %f_b : memref<2xi64>, memref<?xi8>
  }

  // acc + repr(bound `which`); acc is released.
  func.func private @__ly_slice_append_bound(%acc_h: memref<2xi64> {ly.ownership.object_header}, %acc_b: memref<?xi8>, %self: memref<5xi64>, %which: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [0]} {
    %ptr = func.call @__ly_slice_word_ptr(%self, %which) : (memref<5xi64>, i64) -> !llvm.ptr
    %word = llvm.load %ptr : !llvm.ptr -> i64
    %class = func.call @__ly_slot_class(%word) : (i64) -> i64
    %r_h, %r_b = func.call @__ly_repr_boxed_or_default(%ptr, %class) : (!llvm.ptr, i64) -> (memref<2xi64>, memref<?xi8>)
    %h, %b = func.call @LyUnicode_Concat(%acc_h, %acc_b, %r_h, %r_b) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.call @LyUnicode_DecRef(%acc_h) : (memref<2xi64>) -> ()
    func.call @LyUnicode_DecRef(%r_h) : (memref<2xi64>) -> ()
    func.return %h, %b : memref<2xi64>, memref<?xi8>
  }

  // acc + `length` bytes of the runtime's own text; acc is released.
  func.func private @__ly_slice_append_text(%acc_h: memref<2xi64> {ly.ownership.object_header}, %acc_b: memref<?xi8>, %text: memref<?xi8>, %length: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [0]} {
    %c0 = arith.constant 0 : index
    %t_h, %t_b = func.call @__ly_unicode_from_valid_utf8(%text, %c0, %length) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %h, %b = func.call @LyUnicode_Concat(%acc_h, %acc_b, %t_h, %t_b) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.call @LyUnicode_DecRef(%acc_h) : (memref<2xi64>) -> ()
    func.call @LyUnicode_DecRef(%t_h) : (memref<2xi64>) -> ()
    func.return %h, %b : memref<2xi64>, memref<?xi8>
  }

  // slice == slice: the three bounds pairwise, as CPython compares the
  // (start, stop, step) tuples.
  func.func @LySlice_EqBool(%lhs: memref<5xi64> {ly.ownership.object_header}, %rhs: memref<5xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.slice", ly.runtime.method = "__eq__"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %l0 = func.call @__ly_slice_word_ptr(%lhs, %zero) : (memref<5xi64>, i64) -> !llvm.ptr
    %r0 = func.call @__ly_slice_word_ptr(%rhs, %zero) : (memref<5xi64>, i64) -> !llvm.ptr
    %l1 = func.call @__ly_slice_word_ptr(%lhs, %one) : (memref<5xi64>, i64) -> !llvm.ptr
    %r1 = func.call @__ly_slice_word_ptr(%rhs, %one) : (memref<5xi64>, i64) -> !llvm.ptr
    %l2 = func.call @__ly_slice_word_ptr(%lhs, %two) : (memref<5xi64>, i64) -> !llvm.ptr
    %r2 = func.call @__ly_slice_word_ptr(%rhs, %two) : (memref<5xi64>, i64) -> !llvm.ptr
    %e0 = func.call @__ly_box_equal(%l0, %r0) : (!llvm.ptr, !llvm.ptr) -> i1
    %e1 = func.call @__ly_box_equal(%l1, %r1) : (!llvm.ptr, !llvm.ptr) -> i1
    %e2 = func.call @__ly_box_equal(%l2, %r2) : (!llvm.ptr, !llvm.ptr) -> i1
    %e01 = arith.andi %e0, %e1 : i1
    %all = arith.andi %e01, %e2 : i1
    func.return %all : i1
  }

  func.func @LySlice_NeBool(%lhs: memref<5xi64> {ly.ownership.object_header}, %rhs: memref<5xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.slice", ly.runtime.method = "__ne__"} {
    %true = arith.constant true
    %eq = func.call @LySlice_EqBool(%lhs, %rhs) : (memref<5xi64>, memref<5xi64>) -> i1
    %ne = arith.xori %eq, %true : i1
    func.return %ne : i1
  }

  // slice < slice and the other orderings: slice_richcompare compares the
  // (start, stop, step) tuples, so the slice's three words are compared as
  // a tuple's items are.
  func.func private @__ly_slice_compare(%lhs: memref<5xi64>, %rhs: memref<5xi64>, %op: i64) -> i1 {
    %three = arith.constant 3 : i64
    %lhs_items = func.call @__ly_slice_items(%lhs) : (memref<5xi64>) -> memref<?xi64>
    %rhs_items = func.call @__ly_slice_items(%rhs) : (memref<5xi64>) -> memref<?xi64>
    %cmp = func.call @__ly_sequence_compare_op(%three, %lhs_items, %three, %rhs_items, %op) : (i64, memref<?xi64>, i64, memref<?xi64>, i64) -> i1
    func.return %cmp : i1
  }

  func.func @LySlice_LtBool(%lhs: memref<5xi64> {ly.ownership.object_header}, %rhs: memref<5xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.slice", ly.runtime.method = "__lt__"} {
    %op = arith.constant 0 : i64
    %result = func.call @__ly_slice_compare(%lhs, %rhs, %op) : (memref<5xi64>, memref<5xi64>, i64) -> i1
    func.return %result : i1
  }

  func.func @LySlice_LeBool(%lhs: memref<5xi64> {ly.ownership.object_header}, %rhs: memref<5xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.slice", ly.runtime.method = "__le__"} {
    %op = arith.constant 1 : i64
    %result = func.call @__ly_slice_compare(%lhs, %rhs, %op) : (memref<5xi64>, memref<5xi64>, i64) -> i1
    func.return %result : i1
  }

  func.func @LySlice_GtBool(%lhs: memref<5xi64> {ly.ownership.object_header}, %rhs: memref<5xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.slice", ly.runtime.method = "__gt__"} {
    %op = arith.constant 4 : i64
    %result = func.call @__ly_slice_compare(%lhs, %rhs, %op) : (memref<5xi64>, memref<5xi64>, i64) -> i1
    func.return %result : i1
  }

  func.func @LySlice_GeBool(%lhs: memref<5xi64> {ly.ownership.object_header}, %rhs: memref<5xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.slice", ly.runtime.method = "__ge__"} {
    %op = arith.constant 5 : i64
    %result = func.call @__ly_slice_compare(%lhs, %rhs, %op) : (memref<5xi64>, memref<5xi64>, i64) -> i1
    func.return %result : i1
  }

  // hash(slice): CPython 3.12+'s slice_hash -- tuplehash's xxHash lanes over
  // (start, stop, step) without the length tuplehash adds at the end.
  func.func @LySlice_Hash(%self: memref<5xi64> {ly.ownership.object_header}) -> i64 attributes {ly.runtime.contract = "builtins.slice", ly.runtime.method = "__hash__"} {
    %zero = arith.constant 0 : i64
    %three = arith.constant 3 : i64
    %first = func.call @__ly_slice_word_ptr(%self, %zero) : (memref<5xi64>, i64) -> !llvm.ptr
    %acc = func.call @__ly_xxhash_slot_lanes(%first, %three) : (!llvm.ptr, i64) -> i64
    %sentinel = arith.constant -1 : i64
    %replacement = arith.constant 1546275796 : i64
    %is_sentinel = arith.cmpi eq, %acc, %sentinel : i64
    %result = arith.select %is_sentinel, %replacement, %acc : i1, i64
    func.return %result : i64
  }

  func.func @LySlice_DecRef(%self: memref<5xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.slice", ly.runtime.deallocator} {
    %storage = memref.cast %self : memref<5xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c2 = arith.constant 2 : i64
    %items = func.call @__ly_slice_items(%self) : (memref<5xi64>) -> memref<?xi64>
    func.call @LyObject_ReleaseBoxedPayloadArraySlotRaw(%items, %c0) : (memref<?xi64>, i64) -> ()
    func.call @LyObject_ReleaseBoxedPayloadArraySlotRaw(%items, %c1) : (memref<?xi64>, i64) -> ()
    func.call @LyObject_ReleaseBoxedPayloadArraySlotRaw(%items, %c2) : (memref<?xi64>, i64) -> ()
    memref.dealloc %self : memref<5xi64>
    cf.br ^done

  ^done:
    func.return
  }
}
