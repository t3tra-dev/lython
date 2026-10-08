// `dict` and its views -- CPython's Objects/dictobject.c: insertion-ordered
// entries beside a table of indices, as the compact dict keeps them.
//
// Signature source (1:1 correspondence target):
//   https://github.com/python/typeshed/blob/main/stdlib/builtins.pyi

module attributes {
  ly.typing.manifest,
  ly.runtime.contracts = ["builtins.dict"]
} {
  func.func private @__ly_pending_push(%mark: i64, %kind: i64, %value: i64) -> i64
  func.func private @__ly_pending_pop(%index: i64)
  // ===== declared here, defined in another runtime file or built by the lowering =====
  func.func private @__ly_unicode_from_valid_utf8(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]}
  func.func private @LyBaseException_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.BaseException", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"}
  func.func private @LyBaseException_New(%class_word: i64 {ly.runtime.class_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class, ly.runtime.contract = "builtins.BaseException", ly.runtime.initializer = "__new__"}
  func.func private @LyEH_ThrowException(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "raise"}
  func.func private @LyObject_ReleaseBoxedPayloadArraySlotRaw(%payload: memref<?xi64>, %logical_index: i64)
  func.func private @LyObject_ReleaseBoxedPayloadRaw(%box: memref<5xi64>)
  func.func private @LyObject_ReleaseStorageToZero(%storage: memref<?xi64>) -> i1 attributes {ly.runtime.contract = "builtins.object", ly.runtime.primitive = "release_to_zero"}
  func.func private @LyUnicode_Concat(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_bytes: memref<?xi8>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__add__"}
  func.func private @LyUnicode_DecRef(%header: memref<2xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.str", ly.runtime.deallocator}
  func.func private @LyUnicode_FromBytes(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class, ly.runtime.contract = "builtins.str", ly.runtime.initializer = "__new__"}
  func.func private @__ly_box_equal(%lhs: !llvm.ptr, %rhs: !llvm.ptr) -> i1
  func.func private @__ly_box_hash_key(%box: !llvm.ptr, %role: i64) -> i64
  func.func private @__ly_box_word_count() -> i64
  func.func private @__ly_default_repr_from_addr(%ptr: i64, %prefix: memref<?xi8>, %prefix_len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.object", ly.runtime.primitive = "default_repr_addr", ly.runtime.result_contract = "builtins.str"}
  func.func private @__ly_exc_ext_set(%header: memref<3xi64>, %slot: i64, %value: i64) attributes {ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "ext_set"}
  func.func private @__ly_exc_payload_alloc(%count: i64) -> i64 attributes {ly.runtime.contract = "builtins.BaseException", ly.runtime.primitive = "payload_alloc"}
  func.func private @__ly_exc_payload_store_box(%block: i64, %slot: i64, %box: !llvm.ptr)
  func.func private @__ly_global_view_i64(%pointer: i64, %size: i64) -> memref<?xi64>
  func.func private @__ly_global_view_i8(%pointer: i64, %size: i64) -> memref<?xi8>
  func.func private @__ly_handle_retain_raw(%entity: i64)
  memref.global "private" constant @__ly_object_repr_prefix : memref<20xi8>
  func.func private @__ly_repr_boxed_by_contract(%box: !llvm.ptr, %class_word: i64) -> (memref<2xi64>, memref<?xi8>, i1) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]}
  func.func private @__ly_repr_boxed_or_default(%box_ptr: !llvm.ptr, %class_word: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.object", ly.runtime.primitive = "repr_boxed_or_default", ly.runtime.result_contract = "builtins.str"}
  memref.global "private" constant @__ly_repr_colon : memref<2xi8>
  memref.global "private" constant @__ly_repr_comma : memref<2xi8>
  memref.global "private" constant @__ly_repr_lbrace : memref<1xi8>
  memref.global "private" constant @__ly_repr_rbrace : memref<1xi8>
  func.func private @__ly_slot_class(%word: i64) -> i64
  func.func private @free_raw_i64_ptr(%address: i64)

  py.class @dict attributes {
    base_names = ["MutableMapping"], ly.typing.params = ["K", "V"],
    ly.runtime.contract = "builtins.dict", ly.runtime.required,
    ly.runtime.required_deallocator,
    ly.runtime.required_initializers = ["__new__"],
    ly.runtime.required_methods = ["__len__"],
    ly.runtime.required_primitives = ["ensure_capacity"],
    ly.typing.structural_mutators = ["__setitem__", "update"],
    ly.typing.base_args = [[!py.contract<"$K">, !py.contract<"$V">]],
    method_names = ["__init__", "__init__", "__len__", "__iter__",
                    "__getitem__", "get", "get", "get", "__setitem__",
                    "__delitem__", "__contains__", "keys", "values", "items",
                    "__repr__", "clear", "copy", "update", "update", "__or__",
                    "__eq__", "__ne__", "pop", "pop"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"builtins.dict">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.dict">, !py.protocol<"Iterable", [!py.contract<"builtins.tuple", [!py.contract<"$K">, !py.contract<"$V">]>]>] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.dict">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.dict">] -> [!py.protocol<"Iterator", [!py.contract<"$K">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.dict">, !py.contract<"$K">] -> [!py.contract<"$V">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.dict">, !py.contract<"$K">] -> [!py.union<!py.contract<"$V">, !py.literal<None>>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.dict">, !py.contract<"$K">, !py.contract<"$V">] -> [!py.contract<"$V">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.dict">, !py.contract<"$K">, !py.typevar<"D">] -> [!py.union<!py.contract<"$V">, !py.typevar<"D">>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.dict">, !py.contract<"$K">, !py.contract<"$V">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.dict">, !py.contract<"$K">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.dict">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.dict">] -> [!py.contract<"builtins.dict_keys", [!py.contract<"$K">, !py.contract<"$V">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.dict">] -> [!py.contract<"builtins.dict_values", [!py.contract<"$K">, !py.contract<"$V">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.dict">] -> [!py.contract<"builtins.dict_items", [!py.contract<"$K">, !py.contract<"$V">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.dict">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.dict">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.dict">] -> [!py.contract<"builtins.dict", [!py.contract<"$K">, !py.contract<"$V">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.dict">, !py.contract<"builtins.dict", [!py.contract<"$K">, !py.contract<"$V">]>] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.dict">, !py.protocol<"Iterable", [!py.contract<"builtins.tuple", [!py.contract<"$K">, !py.contract<"$V">]>]>] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.dict">, !py.contract<"builtins.dict", [!py.contract<"$K">, !py.contract<"$V">]>] -> [!py.contract<"builtins.dict", [!py.contract<"$K">, !py.contract<"$V">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.dict">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.dict">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.dict">, !py.contract<"$K">] -> [!py.contract<"$V">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.dict">, !py.contract<"$K">, !py.contract<"$V">] -> [!py.contract<"$V">]>
    ],
    method_kinds = ["instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance"]
  } {}

  py.class @MappingView attributes {
    base_names = ["Sized"], ly.typing.abstract,
    method_names = ["__len__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"typing.MappingView">] -> [!py.contract<"builtins.int">]>
    ],
    method_kinds = ["instance"]
  } {}
  py.class @KeysView attributes {
    base_names = ["MappingView", "AbstractSet"], ly.typing.params = ["K"],
    ly.typing.param_variance = ["covariant"],
    ly.typing.base_args = [[], [!py.contract<"$K">]],
    method_names = ["__contains__", "__iter__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"typing.KeysView">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"typing.KeysView">] -> [!py.protocol<"Iterator", [!py.contract<"$K">]>]>
    ],
    method_kinds = ["instance", "instance"]
  } {}
  py.class @ValuesView attributes {
    base_names = ["MappingView", "Collection"], ly.typing.params = ["V"],
    ly.typing.param_variance = ["covariant"],
    ly.typing.base_args = [[], [!py.contract<"$V">]],
    method_names = ["__contains__", "__iter__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"typing.ValuesView">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"typing.ValuesView">] -> [!py.protocol<"Iterator", [!py.contract<"$V">]>]>
    ],
    method_kinds = ["instance", "instance"]
  } {}
  py.class @ItemsView attributes {
    base_names = ["MappingView", "AbstractSet"], ly.typing.params = ["K", "V"],
    ly.typing.param_variance = ["covariant", "covariant"],
    ly.typing.base_args = [[], [!py.contract<"builtins.tuple">]],
    method_names = ["__contains__", "__iter__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"typing.ItemsView">, !py.contract<"builtins.tuple">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"typing.ItemsView">] -> [!py.protocol<"Iterator", [!py.contract<"builtins.tuple", [!py.contract<"$K">, !py.contract<"$V">]>]>]>
    ],
    method_kinds = ["instance", "instance"]
  } {}
  py.class @dict_keys attributes {
    base_names = ["KeysView"], ly.typing.final,
    ly.typing.params = ["K", "V"],
    ly.typing.param_variance = ["covariant", "covariant"],
    ly.typing.base_args = [[!py.contract<"$K">]],
    method_names = ["__reversed__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"builtins.dict_keys", [!py.contract<"$K">, !py.contract<"$V">]>] -> [!py.protocol<"Iterator", [!py.contract<"$K">]>]>
    ],
    method_kinds = ["instance"]
  } {}
  py.class @dict_values attributes {
    base_names = ["ValuesView"], ly.typing.final,
    ly.typing.params = ["K", "V"],
    ly.typing.param_variance = ["covariant", "covariant"],
    ly.typing.base_args = [[!py.contract<"$V">]],
    method_names = ["__reversed__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"builtins.dict_values", [!py.contract<"$K">, !py.contract<"$V">]>] -> [!py.protocol<"Iterator", [!py.contract<"$V">]>]>
    ],
    method_kinds = ["instance"]
  } {}
  py.class @dict_items attributes {
    base_names = ["ItemsView"], ly.typing.final,
    ly.typing.params = ["K", "V"],
    ly.typing.param_variance = ["covariant", "covariant"],
    ly.typing.base_args = [[!py.contract<"$K">, !py.contract<"$V">]],
    method_names = ["__reversed__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"builtins.dict_items", [!py.contract<"$K">, !py.contract<"$V">]>] -> [!py.protocol<"Iterator", [!py.contract<"builtins.tuple", [!py.contract<"$K">, !py.contract<"$V">]>]>]>
    ],
    method_kinds = ["instance"]
  } {}

  func.func private @LyDict_Shape() -> memref<8xi64> attributes {ly.runtime.contract = "builtins.dict", ly.runtime.shape}

  // ===== builtins.dict: one entity, one root =====
  //
  // The handle is `memref<8xi64>` (the same width _io's wrappers use):
  //
  //   word 0  refcount            word 4  keys base address
  //   word 1  class word (12)       word 5  values base address
  //   word 2  length              word 6  present base address
  //   word 3  capacity            word 7  reserved
  //
  // Why the arrays are ADDRESSES in the handle and not values beside it: a
  // growth then writes the new address THROUGH the handle, so every holder of
  // the dict observes it with no further action, and a mutation has nothing to
  // rename. That is what lets ensure_capacity / setitem_box / update be void
  // and in-place instead of "consume the entity and hand back a new tuple"
  // (rfc/memory-safety-proof.md, `Interior`). The word offsets are mirrored in
  // Passes/Runtime/ABI/ContainerLayout.h.
  // The words of a table block: the stamp, then `__ly_dict_table_slots` slots
  // of `__ly_dict_index_width` bytes, rounded up to whole words.
  func.func private @__ly_dict_table_words(%capacity: i64) -> i64 {
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %seven = arith.constant 7 : i64
    %three = arith.constant 3 : i64
    %width = func.call @__ly_dict_index_width(%capacity) : (i64) -> i64
    %slots = func.call @__ly_dict_table_slots(%capacity) : (i64) -> i64
    %bytes = arith.muli %slots, %width : i64
    %padded = arith.addi %bytes, %seven : i64
    %slot_words = arith.shrui %padded, %three : i64
    %words = arith.addi %slot_words, %one : i64
    func.return %words : i64
  }

  // The stamp every empty dict's table word points at: an empty dict owns no
  // block until its first insert, as CPython's shares `empty_keys_struct`.
  // ⛔ Not a constant: the lowering re-stores the stamp it read whenever it
  // writes a dict's length (`invalidateMappingTableOnShrink`), and an empty
  // dict's length cannot shrink, so what it stores here is always the 0 it read.
  memref.global "private" @__ly_dict_empty_stamp : memref<1xi64> = dense<0> {alignment = 8 : i64}
  // ...and its arrays: one empty slot and one absent hash, read-only. A reader
  // that loads slot 0 before deciding a miss (the lowering's d.get) reads an
  // empty slot rather than address 0; a writer grows the dict first, so
  // nothing writes here -- and one that did would fault, not corrupt.
  memref.global "private" constant @__ly_dict_empty_slot : memref<1xi64> = dense<0> {alignment = 8 : i64}
  memref.global "private" constant @__ly_dict_empty_hash : memref<1xi64> = dense<-1> {alignment = 8 : i64}

  func.func private @__ly_dict_alloc(%length: i64) -> memref<8xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.dict"], ly.ownership.owned_results = [0]} {
    %zero = arith.constant 0 : i64
    %is_empty = arith.cmpi sle, %length, %zero : i64
    %r = scf.if %is_empty -> memref<8xi64> {
      %self = memref.alloc() {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<8xi64>
      %one = arith.constant 1 : i64
      %class_word = arith.constant {ly.class_of = "builtins.dict"} 12 : i64
      %stamp = memref.get_global @__ly_dict_empty_stamp : memref<1xi64>
      %stamp_index = memref.extract_aligned_pointer_as_index %stamp : memref<1xi64> -> index
      %stamp_word = arith.index_cast %stamp_index : index to i64
      %empty_slot = memref.get_global @__ly_dict_empty_slot : memref<1xi64>
      %empty_slot_index = memref.extract_aligned_pointer_as_index %empty_slot : memref<1xi64> -> index
      %slot_word = arith.index_cast %empty_slot_index : index to i64
      %empty_hash = memref.get_global @__ly_dict_empty_hash : memref<1xi64>
      %empty_hash_index = memref.extract_aligned_pointer_as_index %empty_hash : memref<1xi64> -> index
      %hash_word = arith.index_cast %empty_hash_index : index to i64
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      memref.store %one, %self[%c0] : memref<8xi64>
      memref.store %class_word, %self[%c1] : memref<8xi64>
      memref.store %zero, %self[%c2] : memref<8xi64>
      memref.store %zero, %self[%c3] : memref<8xi64>
      memref.store %slot_word, %self[%c4] : memref<8xi64>
      memref.store %slot_word, %self[%c5] : memref<8xi64>
      memref.store %hash_word, %self[%c6] : memref<8xi64>
      memref.store %stamp_word, %self[%c7] : memref<8xi64>
      scf.yield %self : memref<8xi64>
    } else {
      %self = func.call @__ly_dict_alloc_block(%length) : (i64) -> memref<8xi64>
      scf.yield %self : memref<8xi64>
    }
    func.return %r : memref<8xi64>
  }

  func.func private @__ly_dict_alloc_block(%length: i64) -> memref<8xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.dict"], ly.ownership.owned_results = [0]} {
    %one = arith.constant 1 : i64
    %handle_words = func.call @__ly_box_word_count() : () -> i64
    %class_word = arith.constant {ly.class_of = "builtins.dict"} 12 : i64
    %zero = arith.constant 0 : i64
    %lower = arith.constant 0 : index
    %step = arith.constant 1 : index
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %length_slot = arith.constant 2 : index
    %capacity_slot = arith.constant 3 : index
    %keys_slot = arith.constant 4 : index
    %values_slot = arith.constant 5 : index
    %present_slot = arith.constant 6 : index
    %reserved_slot = arith.constant 7 : index

    %self = memref.alloc() {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<8xi64>
    // A dict made at a known size (a literal, a copy) holds exactly that many
    // entries; one that grows takes the floor (`__ly_dict_round_capacity`).
    %one_entry = arith.constant 1 : i64
    %capacity = arith.maxsi %length, %one_entry : i64
    // ⭐ ONE BLOCK FOR ALL FOUR ARRAYS. Keys, values, the present flags and the
    // index table are all sized from the capacity and all live and die
    // together, and they used to be four allocations -- five with the handle,
    // where CPython's dict is two. Building `{j: j for j in range(20)}` a
    // hundred thousand times spent more of its time in the allocator than in
    // the hashing.
    //
    // The handle still holds four addresses, so every reader and every growth
    // is unchanged; what changed is where they come from. Keys sit at offset
    // zero, which is what lets `free_raw_i64_ptr(keys)` release the block.
    //
    //   [0, capacity*W)                  keys
    //   [capacity*W, 2*capacity*W)       values
    //   [2*capacity*W, +capacity)        key hashes: -1 = no entry (CPython
    //                                    never hashes to -1), 0 = an entry
    //                                    whose hash is not computed yet
    //   [.., +2*capacity + 1)            index table, word 0 = built-for length;
    //                                    one state word per slot (the entry's
    //                                    hash is in the hashes array)
    //
    // ⛔ The hashes are not in the key's box: a box is the value's handle,
    // the same in a list as in a dict, and a list never reads a hash. CPython
    // keeps the hash in the dict entry beside the key, as this does.
    %four = arith.constant 4 : i64
    %eight = arith.constant 8 : i64
    %two = arith.constant 2 : i64
    %payload_words = arith.muli %capacity, %handle_words : i64
    %pair_words = arith.muli %payload_words, %two : i64
    %flag_words = arith.addi %capacity, %zero : i64
    %through_present = arith.addi %pair_words, %flag_words : i64
    %table_alloc_i64 = func.call @__ly_dict_table_words(%capacity) : (i64) -> i64
    %block_words = arith.addi %through_present, %table_alloc_i64 : i64
    %block_words_index = arith.index_cast %block_words : i64 to index
    // Plain memref.alloc with no alignment attribute is a bare malloc, so the
    // aligned pointer IS the allocated pointer and free_raw_i64_ptr can
    // release it later (same convention as __ly_exc_payload_alloc).
    %block = memref.alloc(%block_words_index) : memref<?xi64>
    %block_index = memref.extract_aligned_pointer_as_index %block : memref<?xi64> -> index
    %keys_word = arith.index_cast %block_index : index to i64
    %values_off = arith.muli %payload_words, %eight : i64
    %values_word = arith.addi %keys_word, %values_off : i64
    %present_off = arith.muli %pair_words, %eight : i64
    %present_word = arith.addi %keys_word, %present_off : i64
    %table_off = arith.muli %through_present, %eight : i64
    %table_word = arith.addi %keys_word, %table_off : i64

    memref.store %one, %self[%refcount_slot] : memref<8xi64>
    memref.store %class_word, %self[%layout_slot] : memref<8xi64>
    memref.store %length, %self[%length_slot] : memref<8xi64>
    memref.store %capacity, %self[%capacity_slot] : memref<8xi64>
    memref.store %keys_word, %self[%keys_slot] : memref<8xi64>
    memref.store %values_word, %self[%values_slot] : memref<8xi64>
    memref.store %present_word, %self[%present_slot] : memref<8xi64>
    memref.store %table_word, %self[%reserved_slot] : memref<8xi64>
    // Zero the flags and the table in one walk. Zero is the right table stamp
    // for a fresh handle whose entries the caller has not written yet -- a
    // nonzero %length means an evidence-written dict, and the mismatch is what
    // makes the first probe build the table.
    %pair_words_index = arith.index_cast %pair_words : i64 to index
    scf.for %w = %pair_words_index to %block_words_index step %step {
      memref.store %zero, %block[%w] : memref<?xi64>
    }
    // Every entry starts absent: hash -1, which no hash is (`__ly_hash_fixup`).
    %absent = arith.constant -1 : i64
    %through_index = arith.index_cast %through_present : i64 to index
    scf.for %w = %pair_words_index to %through_index step %step {
      memref.store %absent, %block[%w] : memref<?xi64>
    }
    func.return %self : memref<8xi64>
  }

  // Borrowed views of the interior arrays, derived at the point of use. The
  // view's SSA name is not an identity: identity is the handle, so a fresh
  // view after a growth and a view from before it name the same slot of the
  // same entity. Marked ly.runtime.interior_word so release placement pins the
  // handle across the view's uses (same role as __ly_exc_fields_block); a
  // plain private helper would leave the ownership walk nothing to follow.
  func.func private @__ly_dict_keys(%self: memref<8xi64>) -> memref<?xi64> attributes {ly.runtime.contract = "builtins.dict", ly.runtime.interior_word, ly.runtime.primitive = "keys_view"} {
    %capacity_slot = arith.constant 3 : index
    %keys_slot = arith.constant 4 : index
    %handle_words = func.call @__ly_box_word_count() : () -> i64
    %capacity = memref.load %self[%capacity_slot] : memref<8xi64>
    %words = arith.muli %capacity, %handle_words : i64
    %base = memref.load %self[%keys_slot] : memref<8xi64>
    %view = func.call @__ly_global_view_i64(%base, %words) : (i64, i64) -> memref<?xi64>
    func.return %view : memref<?xi64>
  }

  func.func private @__ly_dict_values(%self: memref<8xi64>) -> memref<?xi64> attributes {ly.runtime.contract = "builtins.dict", ly.runtime.interior_word, ly.runtime.primitive = "values_view"} {
    %capacity_slot = arith.constant 3 : index
    %values_slot = arith.constant 5 : index
    %handle_words = func.call @__ly_box_word_count() : () -> i64
    %capacity = memref.load %self[%capacity_slot] : memref<8xi64>
    %words = arith.muli %capacity, %handle_words : i64
    %base = memref.load %self[%values_slot] : memref<8xi64>
    %view = func.call @__ly_global_view_i64(%base, %words) : (i64, i64) -> memref<?xi64>
    func.return %view : memref<?xi64>
  }

  func.func private @__ly_dict_present(%self: memref<8xi64>) -> memref<?xi64> attributes {ly.runtime.contract = "builtins.dict", ly.runtime.interior_word, ly.runtime.primitive = "present_view"} {
    %capacity_slot = arith.constant 3 : index
    %present_slot = arith.constant 6 : index
    %capacity = memref.load %self[%capacity_slot] : memref<8xi64>
    %base = memref.load %self[%present_slot] : memref<8xi64>
    %view = func.call @__ly_global_view_i64(%base, %capacity) : (i64, i64) -> memref<?xi64>
    func.return %view : memref<?xi64>
  }

  // The key hashes, one word per entry: -1 means there is no entry there,
  // 0 that its hash is not computed yet (a hash that IS 0 is recomputed,
  // which is cheap). ⛔ No separate present flags: "is there an entry" is the
  // hash word's -1, the one value no hash takes, as the hash array is what
  // every reader of an entry loads anyway.
  func.func private @__ly_dict_hashes(%self: memref<8xi64>) -> memref<?xi64> attributes {ly.runtime.contract = "builtins.dict", ly.runtime.interior_word, ly.runtime.primitive = "hashes_view"} {
    %capacity_slot = arith.constant 3 : index
    %present_slot = arith.constant 6 : index
    %zero_off = arith.constant 0 : i64
    %capacity = memref.load %self[%capacity_slot] : memref<8xi64>
    %present = memref.load %self[%present_slot] : memref<8xi64>
    %offset = arith.muli %capacity, %zero_off : i64
    %base = arith.addi %present, %offset : i64
    %view = func.call @__ly_global_view_i64(%base, %capacity) : (i64, i64) -> memref<?xi64>
    func.return %view : memref<?xi64>
  }

  // ===== builtins.dict: the index table =====
  //
  // ⭐ CPython's dict is dk_indices (a hash table of INDICES) beside dk_entries
  // (dense, insertion-ordered), and the dense half is what makes iteration and
  // repr insertion-ordered without the table having to be. This had the dense
  // half and no table at all: `__ly_dict_probe` walked every live entry, so
  // d[k] was O(n) and building a dict was O(n^2) -- 32,000 keys took 1.21 s
  // against CPython 3.14's 18 ms.
  //
  // The table is the tail of the dict's one block: word 0 is the length it
  // was last built for, then `__ly_dict_table_slots` slots of
  // `__ly_dict_index_width` bytes, each 0 unused, 1 dummy or dense index + 2
  // (the entry's hash is in the hashes array, not the slot).
  //
  // ⛔ Why the size is derived from `capacity` rather than carried: the handle
  // has eight words and every one is spoken for (ABI/ContainerLayout.h).
  // Tying the table to the entries array is what CPython does anyway
  // (dk_size against USABLE_FRACTION).
  func.func private @__ly_dict_round_capacity(%wanted: i64) -> i64 {
    // Five entries over an eight-slot table: CPython's PyDict_MINSIZE table
    // and its USABLE_FRACTION. The dense capacity is any count: the table is
    // what has to be a power of two (`__ly_dict_table_slots`).
    %minimum = arith.constant 5 : i64
    %capacity = arith.maxsi %wanted, %minimum : i64
    func.return %capacity : i64
  }

  // ⭐ The table's slot count: the smallest power of two that keeps the dense
  // array within two thirds of it -- CPython's USABLE_FRACTION -- so a
  // 5-entry dict holds 5 dense entries over 8 slots instead of 8 over 16.
  func.func private @__ly_dict_table_slots(%capacity: i64) -> i64 {
    %one = arith.constant 1 : i64
    %three = arith.constant 3 : i64
    %eight = arith.constant 8 : i64
    %bits = arith.constant 64 : i64
    %tripled = arith.muli %capacity, %three : i64
    %plus = arith.addi %tripled, %one : i64
    %needed = arith.shrui %plus, %one : i64
    // The next power of two at or above `needed`, by its leading zeros; the
    // eight-slot floor makes `needed - 1` positive wherever it matters.
    %below = arith.subi %needed, %one : i64
    %zeros = math.ctlz %below : i64
    %exponent = arith.subi %bits, %zeros : i64
    %power = arith.shli %one, %exponent : i64
    %slots = arith.maxsi %power, %eight : i64
    func.return %slots : i64
  }

  // The table's slots as BYTES: `__ly_dict_table_slots` slots of
  // `__ly_dict_index_width` bytes each, after the stamp word. Read and written through
  // `__ly_dict_slot_load` / `__ly_dict_slot_store`.
  func.func private @__ly_dict_table(%self: memref<8xi64>) -> memref<?xi8> attributes {ly.runtime.contract = "builtins.dict", ly.runtime.interior_word, ly.runtime.primitive = "table_view"} {
    %capacity_slot = arith.constant 3 : index
    %table_slot = arith.constant 7 : index
    %eight = arith.constant 8 : i64
    %two = arith.constant 2 : i64
    %capacity = memref.load %self[%capacity_slot] : memref<8xi64>
    %width = func.call @__ly_dict_index_width(%capacity) : (i64) -> i64
    %slots = func.call @__ly_dict_table_slots(%capacity) : (i64) -> i64
    %bytes = arith.muli %slots, %width : i64
    %base = memref.load %self[%table_slot] : memref<8xi64>
    %first = arith.addi %base, %eight : i64
    %view = func.call @__ly_global_view_i8(%first, %bytes) : (i64, i64) -> memref<?xi8>
    func.return %view : memref<?xi8>
  }

  // ⭐ CPython's dk_indices width: a slot holds a dense index + 2 (0 unused, 1
  // dummy), so the narrowest integer that holds capacity + 1 is enough -- one
  // byte up to 128 entries, where every slot used to be eight.
  func.func private @__ly_dict_index_width(%capacity: i64) -> i64 {
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %four = arith.constant 4 : i64
    %eight = arith.constant 8 : i64
    %byte_limit = arith.constant 128 : i64
    %short_limit = arith.constant 32766 : i64
    %int_limit = arith.constant 2147483645 : i64
    %fits1 = arith.cmpi sle, %capacity, %byte_limit : i64
    %fits2 = arith.cmpi sle, %capacity, %short_limit : i64
    %fits4 = arith.cmpi sle, %capacity, %int_limit : i64
    %w4 = arith.select %fits4, %four, %eight : i64
    %w2 = arith.select %fits2, %two, %w4 : i64
    %w = arith.select %fits1, %one, %w2 : i64
    func.return %w : i64
  }

  func.func private @__ly_dict_slot_load(%table: memref<?xi8>, %width: i64, %i: i64) -> i64 {
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %four = arith.constant 4 : i64
    %c0 = arith.constant 0 : index
    %offset = arith.muli %i, %width : i64
    %at = arith.index_cast %offset : i64 to index
    %is1 = arith.cmpi eq, %width, %one : i64
    %v = scf.if %is1 -> (i64) {
      %b = memref.load %table[%at] : memref<?xi8>
      %z = arith.extui %b : i8 to i64
      scf.yield %z : i64
    } else {
      %is2 = arith.cmpi eq, %width, %two : i64
      %w = scf.if %is2 -> (i64) {
        %view = memref.view %table[%at][] : memref<?xi8> to memref<1xi16>
        %h = memref.load %view[%c0] : memref<1xi16>
        %z = arith.extui %h : i16 to i64
        scf.yield %z : i64
      } else {
        %is4 = arith.cmpi eq, %width, %four : i64
        %x = scf.if %is4 -> (i64) {
          %view = memref.view %table[%at][] : memref<?xi8> to memref<1xi32>
          %h = memref.load %view[%c0] : memref<1xi32>
          %z = arith.extui %h : i32 to i64
          scf.yield %z : i64
        } else {
          %view = memref.view %table[%at][] : memref<?xi8> to memref<1xi64>
          %h = memref.load %view[%c0] : memref<1xi64>
          scf.yield %h : i64
        }
        scf.yield %x : i64
      }
      scf.yield %w : i64
    }
    func.return %v : i64
  }

  func.func private @__ly_dict_slot_store(%table: memref<?xi8>, %width: i64, %i: i64, %value: i64) {
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %four = arith.constant 4 : i64
    %c0 = arith.constant 0 : index
    %offset = arith.muli %i, %width : i64
    %at = arith.index_cast %offset : i64 to index
    %is1 = arith.cmpi eq, %width, %one : i64
    scf.if %is1 {
      %b = arith.trunci %value : i64 to i8
      memref.store %b, %table[%at] : memref<?xi8>
    } else {
      %is2 = arith.cmpi eq, %width, %two : i64
      scf.if %is2 {
        %view = memref.view %table[%at][] : memref<?xi8> to memref<1xi16>
        %h = arith.trunci %value : i64 to i16
        memref.store %h, %view[%c0] : memref<1xi16>
      } else {
        %is4 = arith.cmpi eq, %width, %four : i64
        scf.if %is4 {
          %view = memref.view %table[%at][] : memref<?xi8> to memref<1xi32>
          %h = arith.trunci %value : i64 to i32
          memref.store %h, %view[%c0] : memref<1xi32>
        } else {
          %view = memref.view %table[%at][] : memref<?xi8> to memref<1xi64>
          memref.store %value, %view[%c0] : memref<1xi64>
        }
      }
    }
    func.return
  }

  // Word 0 of the table block: the length the table was built for.
  func.func private @__ly_dict_table_stamp(%self: memref<8xi64>) -> memref<?xi64> attributes {ly.runtime.contract = "builtins.dict", ly.runtime.interior_word, ly.runtime.primitive = "table_stamp_view"} {
    %table_slot = arith.constant 7 : index
    %one = arith.constant 1 : i64
    %base = memref.load %self[%table_slot] : memref<8xi64>
    %view = func.call @__ly_global_view_i64(%base, %one) : (i64, i64) -> memref<?xi64>
    func.return %view : memref<?xi64>
  }

  func.func private @__ly_dict_self_index_width(%self: memref<8xi64>) -> i64 {
    %capacity_slot = arith.constant 3 : index
    %capacity = memref.load %self[%capacity_slot] : memref<8xi64>
    %width = func.call @__ly_dict_index_width(%capacity) : (i64) -> i64
    func.return %width : i64
  }

  func.func private @__ly_dict_mask(%self: memref<8xi64>) -> i64 {
    %capacity_slot = arith.constant 3 : index
    %one = arith.constant 1 : i64
    %capacity = memref.load %self[%capacity_slot] : memref<8xi64>
    %slots = func.call @__ly_dict_table_slots(%capacity) : (i64) -> i64
    %mask = arith.subi %slots, %one : i64
    func.return %mask : i64
  }

  // Zero the table and re-insert every present entry, then stamp the length it
  // now describes. O(capacity), paid where CPython pays dictresize.
  // Insert the entries at dense [%from, length) and stamp the length. The
  // whole table is `from` = 0 after a zeroing; a growth of the dense array is
  // the tail alone, which is what keeps an append O(1).
  func.func private @__ly_dict_table_fill(%self: memref<8xi64>, %from: i64) {
    %absent_hash = arith.constant -1 : i64
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = func.call @__ly_box_word_count() : () -> i64
    %length_slot = arith.constant 2 : index
    %hashes = func.call @__ly_dict_hashes(%self) : (memref<8xi64>) -> memref<?xi64>
    %table = func.call @__ly_dict_table(%self) : (memref<8xi64>) -> memref<?xi8>
    %width = func.call @__ly_dict_self_index_width(%self) : (memref<8xi64>) -> i64
    %mask = func.call @__ly_dict_mask(%self) : (memref<8xi64>) -> i64
    %from_index = arith.index_cast %from : i64 to index
    %keys = func.call @__ly_dict_keys(%self) : (memref<8xi64>) -> memref<?xi64>
    %present = func.call @__ly_dict_present(%self) : (memref<8xi64>) -> memref<?xi64>
    %keys_idx = memref.extract_aligned_pointer_as_index %keys : memref<?xi64> -> index
    %keys_i64 = arith.index_cast %keys_idx : index to i64
    %keys_ptr = llvm.inttoptr %keys_i64 : i64 to !llvm.ptr
    %len = memref.load %self[%length_slot] : memref<8xi64>
    %len_index = arith.index_cast %len : i64 to index
    scf.for %i = %from_index to %len_index step %c1 {
      %flag = memref.load %present[%i] : memref<?xi64>
      %is_present = arith.cmpi ne, %flag, %absent_hash : i64
      scf.if %is_present {
        %ii = arith.index_cast %i : index to i64
        %base = arith.muli %ii, %c16 : i64
        %cached = memref.load %hashes[%i] : memref<?xi64>
        %unknown = arith.cmpi eq, %cached, %zero : i64
        %hash = scf.if %unknown -> (i64) {
          %entry = llvm.getelementptr %keys_ptr[%base] : (!llvm.ptr, i64) -> !llvm.ptr, i64
          %hash_role_1 = arith.constant 1 : i64
          %computed = func.call @__ly_box_hash_key(%entry, %hash_role_1) : (!llvm.ptr, i64) -> i64
          memref.store %computed, %hashes[%i] : memref<?xi64>
          scf.yield %computed : i64
        } else {
          scf.yield %cached : i64
        }
        %slot = func.call @__ly_dict_clean_slot(%table, %width, %mask, %hash) : (memref<?xi8>, i64, i64, i64) -> i64
        %state = arith.addi %ii, %two : i64
        func.call @__ly_dict_slot_store(%table, %width, %slot, %state) : (memref<?xi8>, i64, i64, i64) -> ()
      }
    }
    %stamp = func.call @__ly_dict_table_stamp(%self) : (memref<8xi64>) -> memref<?xi64>
    memref.store %len, %stamp[%c0] : memref<?xi64>
    func.return
  }

  // ⭐ THE TABLE IS VALIDATED AGAINST THE LENGTH, not maintained by everyone who
  // writes an entry. A dict literal is filled by the LOWERING, which stores
  // keys, values and present words straight into the arrays
  // (Runtime/Core/CollectionPayload.cpp) and never calls setitem; a delete
  // shifts the dense tail and renumbers every index the table holds. Both move
  // the length, so both are caught here and pay one rebuild at the next probe
  // instead of needing a call the C++ side would have to remember to emit.
  //
  // ⛔ What this does NOT catch is a writer that REPLACES a key in place
  // without moving the length. Nothing does that today -- the lowering fills
  // slot 0..n-1 of a fresh dict and setitem only ever overwrites a VALUE -- and
  // the failure would be a lookup that misses, never one that answers wrongly,
  // because the probe still compares the key it lands on.
  // ⭐ THE TABLE IS VALIDATED AGAINST THE LENGTH, not maintained by everyone
  // who writes an entry. A dict literal, and `d[k] = v` itself, are filled by
  // the LOWERING, which stores keys, values and present words straight into the
  // arrays (Runtime/Core/CollectionPayload.cpp) and never calls setitem -- the
  // manifest's own `LyDict_SetItemBox` is not even called in a program that
  // only subscripts. Making every one of those writers announce itself would be
  // a call the C++ side has to remember to emit at each site; comparing against
  // the length catches all of them at the point it matters, the next probe.
  //
  // A grown dense array is caught as `built < length` and costs only the tail:
  // the entries below `built` are still where the table says they are, because
  // an insert appends. That is what keeps building a dict linear -- a full
  // rebuild per append would put the quadratic straight back.
  //
  // ⛔ A SHRUNK one cannot be caught here, and this is the half the lowering
  // does have to announce. `del d[k]` shifts the dense tail down, so the table
  // holds the wrong index for every entry after the hole; a delete that follows
  // an insert leaves the length exactly where it started, and the comparison
  // sees nothing. `RuntimeBundleLowerer::storeContainerLength` stamps the table
  // stale whenever the length it writes is smaller than the one already there,
  // which is the one fact the manifest cannot observe after the fact.
  func.func private @__ly_dict_table_sync(%self: memref<8xi64>) {
    %zero = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %length_slot = arith.constant 2 : index
    %stamp = func.call @__ly_dict_table_stamp(%self) : (memref<8xi64>) -> memref<?xi64>
    %built = memref.load %stamp[%c0] : memref<?xi64>
    %len = memref.load %self[%length_slot] : memref<8xi64>
    %grown = arith.cmpi slt, %built, %len : i64
    %shrunk = arith.cmpi sgt, %built, %len : i64
    scf.if %shrunk {
      func.call @__ly_dict_table_rebuild(%self) : (memref<8xi64>) -> ()
    } else {
      scf.if %grown {
        %negative = arith.cmpi slt, %built, %zero : i64
        %from = arith.select %negative, %zero, %built : i1, i64
        %fresh = arith.cmpi eq, %from, %zero : i64
        scf.if %fresh {
          func.call @__ly_dict_table_rebuild(%self) : (memref<8xi64>) -> ()
        } else {
          func.call @__ly_dict_table_fill(%self, %from) : (memref<8xi64>, i64) -> ()
        }
      }
    }
    func.return
  }

  // Zero every slot, then fill from the first entry.
  func.func private @__ly_dict_table_rebuild(%self: memref<8xi64>) {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %table = func.call @__ly_dict_table(%self) : (memref<8xi64>) -> memref<?xi8>
    %bytes = memref.dim %table, %c0 : memref<?xi8>
    %zero_byte = arith.constant 0 : i8
    scf.for %b = %c0 to %bytes step %c1 {
      memref.store %zero_byte, %table[%b] : memref<?xi8>
    }
    func.call @__ly_dict_table_fill(%self, %zero) : (memref<8xi64>, i64) -> ()
    func.return
  }

  func.func @LyDict_FromLength(%length: i64 {ly.runtime.default_i64 = 0 : i64}) -> memref<8xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class, ly.runtime.contract = "builtins.dict", ly.runtime.initializer = "__new__"} {
    %self = func.call @__ly_dict_alloc(%length) : (i64) -> memref<8xi64>
    func.return %self : memref<8xi64>
  }

  func.func @LyDict_Len(%self: memref<8xi64> {ly.ownership.object_header}) -> i64 attributes {ly.runtime.contract = "builtins.dict", ly.runtime.method = "__len__"} {
    %length_slot = arith.constant 2 : index
    %length = memref.load %self[%length_slot] : memref<8xi64>
    func.return %length : i64
  }

  // Grow the interior arrays in place. Void and non-transfer: the new base
  // addresses are written into the handle, which every holder already names,
  // so there is nothing to hand back and no reference for a caller to
  // re-acquire. This is the acceptance condition of the one-lane form -- the
  // five-lane spelling had to declare transfer_args = [0] + owned_results = [0]
  // because the array VALUES were the entity's identity.
  func.func @LyDict_EnsureCapacity(%self: memref<8xi64> {ly.ownership.object_header}, %required: i64) attributes {ly.runtime.contract = "builtins.dict", ly.runtime.primitive = "ensure_capacity"} {
    %handle_words = func.call @__ly_box_word_count() : () -> i64
    %two = arith.constant 2 : i64
    %zero = arith.constant 0 : i64
    %capacity_slot = arith.constant 3 : index
    %keys_slot = arith.constant 4 : index
    %values_slot = arith.constant 5 : index
    %present_slot = arith.constant 6 : index
    %lower = arith.constant 0 : index
    %step = arith.constant 1 : index

    %capacity = memref.load %self[%capacity_slot] : memref<8xi64>
    %needs_grow = arith.cmpi slt, %capacity, %required : i64
    scf.if %needs_grow {
      %keys = func.call @__ly_dict_keys(%self) : (memref<8xi64>) -> memref<?xi64>
      %values = func.call @__ly_dict_values(%self) : (memref<8xi64>) -> memref<?xi64>
      %present = func.call @__ly_dict_present(%self) : (memref<8xi64>) -> memref<?xi64>
      %old_keys_word = memref.load %self[%keys_slot] : memref<8xi64>
      %old_values_word = memref.load %self[%values_slot] : memref<8xi64>
      %old_present_word = memref.load %self[%present_slot] : memref<8xi64>
      // ⭐ Half again what is asked for, and at least double: CPython's
      // GROWTH_RATE (`used * 3`) sizes its TABLE, whose usable two thirds is
      // twice the entries, and this table is already twice the dense array --
      // so the dense headroom that matches is under 2x, not 3x. Three times
      // the entries took a 5-entry dict to 16 dense slots (517 B where CPython
      // spends 235). The doubling floor keeps an append amortised O(1).
      %one_g0 = arith.constant 1 : i64
      %half = arith.shrui %required, %one_g0 : i64
      %grown_wanted = arith.addi %required, %half : i64
      %doubled = arith.muli %capacity, %two : i64
      %below_required = arith.cmpi slt, %grown_wanted, %doubled : i64
      %wanted = arith.select %below_required, %doubled, %grown_wanted : i1, i64
      %rounded = func.call @__ly_dict_round_capacity(%wanted) : (i64) -> i64
      // A dict's FIRST block is exactly what is asked for: a literal reserves
      // its size before it inserts, and `{k: v}` is then one entry, not five.
      %first_block = arith.cmpi eq, %capacity, %zero : i64
      %new_capacity = arith.select %first_block, %required, %rounded : i1, i64
      %old_words = arith.muli %capacity, %handle_words : i64
      %new_words = arith.muli %new_capacity, %handle_words : i64
      %old_words_index = arith.index_cast %old_words : i64 to index
      %new_words_index = arith.index_cast %new_words : i64 to index
      %old_capacity_index = arith.index_cast %capacity : i64 to index
      %new_capacity_index = arith.index_cast %new_capacity : i64 to index
      // One block, laid out as `__ly_dict_alloc` lays it out.
      %four_g = arith.constant 4 : i64
      %eight_g = arith.constant 8 : i64
      %one_g = arith.constant 1 : i64
      %new_pair_words = arith.muli %new_words, %two : i64
      %new_flag_words = arith.addi %new_capacity, %zero : i64
      %new_through_present = arith.addi %new_pair_words, %new_flag_words : i64
      %new_table_alloc_g = func.call @__ly_dict_table_words(%new_capacity) : (i64) -> i64
      %new_block_words = arith.addi %new_through_present, %new_table_alloc_g : i64
      %new_block_words_index = arith.index_cast %new_block_words : i64 to index
      %new_block = memref.alloc(%new_block_words_index) : memref<?xi64>
      %new_block_index = memref.extract_aligned_pointer_as_index %new_block : memref<?xi64> -> index
      %new_base = arith.index_cast %new_block_index : index to i64
      %new_values_off = arith.muli %new_words, %eight_g : i64
      %new_values_base = arith.addi %new_base, %new_values_off : i64
      %new_present_off = arith.muli %new_pair_words, %eight_g : i64
      %new_present_base = arith.addi %new_base, %new_present_off : i64
      %new_table_off = arith.muli %new_through_present, %eight_g : i64
      %new_table_base = arith.addi %new_base, %new_table_off : i64
      %new_keys = func.call @__ly_global_view_i64(%new_base, %new_words) : (i64, i64) -> memref<?xi64>
      %new_values = func.call @__ly_global_view_i64(%new_values_base, %new_words) : (i64, i64) -> memref<?xi64>
      %new_present = func.call @__ly_global_view_i64(%new_present_base, %new_capacity) : (i64, i64) -> memref<?xi64>
      %new_tail_index = arith.index_cast %new_pair_words : i64 to index
      scf.for %w = %new_tail_index to %new_block_words_index step %step {
        memref.store %zero, %new_block[%w] : memref<?xi64>
      }
      scf.for %i = %lower to %old_words_index step %step {
        %key_word = memref.load %keys[%i] : memref<?xi64>
        %value_word = memref.load %values[%i] : memref<?xi64>
        memref.store %key_word, %new_keys[%i] : memref<?xi64>
        memref.store %value_word, %new_values[%i] : memref<?xi64>
      }
      %absent = arith.constant -1 : i64
      scf.for %i = %lower to %new_capacity_index step %step {
        %copied = arith.cmpi ult, %i, %old_capacity_index : index
        %hash_word = scf.if %copied -> (i64) {
          %old = memref.load %present[%i] : memref<?xi64>
          scf.yield %old : i64
        } else {
          scf.yield %absent : i64
        }
        memref.store %hash_word, %new_present[%i] : memref<?xi64>
      }
      // Publish capacity and the new bases together, then free the old
      // blocks: after these stores no view derived from the handle can reach
      // the old blocks, and before them no reader could reach the new ones.
      // The table is sized from the capacity, so a growth replaces it and
      // rebuilds. That is dictresize, and the doubling is what amortises it.
      %table_slot = arith.constant 7 : index
      %four = arith.constant 4 : i64
      %one_i64 = arith.constant 1 : i64
      memref.store %new_capacity, %self[%capacity_slot] : memref<8xi64>
      memref.store %new_base, %self[%keys_slot] : memref<8xi64>
      memref.store %new_values_base, %self[%values_slot] : memref<8xi64>
      memref.store %new_present_base, %self[%present_slot] : memref<8xi64>
      memref.store %new_table_base, %self[%table_slot] : memref<8xi64>
      func.call @__ly_dict_table_rebuild(%self) : (memref<8xi64>) -> ()
      // Keys sit at offset zero of the block, so this frees all four arrays
      // -- of a dict that had a block: an empty one's arrays are shared.
      %had_block = arith.cmpi sgt, %capacity, %zero : i64
      scf.if %had_block {
        func.call @free_raw_i64_ptr(%old_keys_word) : (i64) -> ()
      }
    }
    func.return
  }

  // Hash-first probe among the present entries: an entry matches when its
  // cached hash (key box word 15; 0 = not yet computed, filled lazily so
  // evidence-written entries join the scheme) equals the probe hash and
  // __ly_box_equal accepts the pair. Returns the slot index or -1. Dense
  // slot order is insertion order, so iteration/repr keep the R6 guarantee.
  // The dense index of the entry whose key equals %key_box, or -1. One table
  // lookup, where this used to be a walk of every live entry.
  func.func private @__ly_dict_probe(%self: memref<8xi64>, %key_box: !llvm.ptr, %key_hash: i64) -> i64 {
    // An empty dict has no table to walk (`__ly_dict_alloc` gives it none).
    %capacity_slot = arith.constant 3 : index
    %capacity = memref.load %self[%capacity_slot] : memref<8xi64>
    %zero = arith.constant 0 : i64
    %has_table = arith.cmpi sgt, %capacity, %zero : i64
    %result = scf.if %has_table -> (i64) {
      %found = func.call @__ly_dict_probe_table(%self, %key_box, %key_hash) : (memref<8xi64>, !llvm.ptr, i64) -> i64
      scf.yield %found : i64
    } else {
      %missing = arith.constant -1 : i64
      scf.yield %missing : i64
    }
    func.return %result : i64
  }

  func.func private @__ly_dict_probe_table(%self: memref<8xi64>, %key_box: !llvm.ptr, %key_hash: i64) -> i64 {
    func.call @__ly_dict_table_sync(%self) : (memref<8xi64>) -> ()
    %keys = func.call @__ly_dict_keys(%self) : (memref<8xi64>) -> memref<?xi64>
    %keys_idx = memref.extract_aligned_pointer_as_index %keys : memref<?xi64> -> index
    %keys_i64 = arith.index_cast %keys_idx : index to i64
    %keys_ptr = llvm.inttoptr %keys_i64 : i64 to !llvm.ptr
    %table = func.call @__ly_dict_table(%self) : (memref<8xi64>) -> memref<?xi8>
    %width = func.call @__ly_dict_self_index_width(%self) : (memref<8xi64>) -> i64
    %mask = func.call @__ly_dict_mask(%self) : (memref<8xi64>) -> i64
    %hashes = func.call @__ly_dict_hashes(%self) : (memref<8xi64>) -> memref<?xi64>
    %found = func.call @__ly_dict_lookup(%table, %width, %mask, %keys_ptr, %hashes, %key_box, %key_hash) : (memref<?xi8>, i64, i64, !llvm.ptr, memref<?xi64>, !llvm.ptr, i64) -> i64
    func.return %found : i64
  }

  // Raise KeyError whose message is the missing key's repr (CPython prints
  // the repr of the key for a dict miss).
  func.func private @__ly_dict_raise_missing_key(%key_box: !llvm.ptr) attributes {ly.runtime.contract = "builtins.dict", ly.runtime.primitive = "raise_missing_key_ptr"} {
    %c1 = arith.constant 1 : i64
    %entity_word = llvm.load %key_box : !llvm.ptr -> i64
    %class_word = func.call @__ly_slot_class(%entity_word) : (i64) -> i64
    %rh, %rb, %ok = func.call @__ly_repr_boxed_by_contract(%key_box, %class_word) : (!llvm.ptr, i64) -> (memref<2xi64>, memref<?xi8>, i1)
    // A key class without a conforming __repr__ still gets a catchable
    // KeyError, CPython-style: fall back to the default object repr keyed on
    // the box address (aborting here turned a user-visible KeyError into a
    // process crash).
    cf.cond_br %ok, ^raise(%rh, %rb : memref<2xi64>, memref<?xi8>), ^default
  ^default:
    %prefix_static = memref.get_global @__ly_object_repr_prefix : memref<20xi8>
    %prefix = memref.cast %prefix_static : memref<20xi8> to memref<?xi8>
    %prefix_len = arith.constant 20 : i64
    %addr = llvm.ptrtoint %key_box : !llvm.ptr to i64
    %dh, %db = func.call @__ly_default_repr_from_addr(%addr, %prefix, %prefix_len) : (i64, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    cf.br ^raise(%dh, %db : memref<2xi64>, memref<?xi8>)
  ^raise(%mh: memref<2xi64>, %mb: memref<?xi8>):
    %key_error = arith.constant {ly.class_of = "builtins.KeyError"} 54 : i64
    %exception:3 = func.call @LyBaseException_New(%key_error) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    %initialized:3 = func.call @LyBaseException_Init(%exception#0, %exception#1, %exception#2, %mh, %mb) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    // The key object, not its repr, is what .args[0] must yield -- and here
    // the key is already boxed, so the payload keeps the real object (an int
    // key comes back as an int, the way CPython's KeyError carries it).
    %one_arg = arith.constant 1 : i64
    %slot_zero = arith.constant 0 : i64
    %payload_slot = arith.constant 3 : i64
    %block = func.call @__ly_exc_payload_alloc(%one_arg) : (i64) -> i64
    func.call @__ly_exc_ext_set(%initialized#0, %payload_slot, %block) : (memref<3xi64>, i64, i64) -> ()
    func.call @__ly_exc_payload_store_box(%block, %slot_zero, %key_box) : (i64, i64, !llvm.ptr) -> ()
    func.call @LyEH_ThrowException(%initialized#0, %initialized#1, %initialized#2) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  // Boxed variant callable from the C++ getitem path (borrowed key box).
  func.func @LyDict_RaiseMissingKey(%key_box: memref<5xi64>) attributes {ly.runtime.contract = "builtins.dict", ly.runtime.primitive = "raise_missing_key"} {
    %box_idx = memref.extract_aligned_pointer_as_index %key_box : memref<5xi64> -> index
    %box_i64 = arith.index_cast %box_idx : index to i64
    %box_ptr = llvm.inttoptr %box_i64 : i64 to !llvm.ptr
    func.call @__ly_dict_raise_missing_key(%box_ptr) : (!llvm.ptr) -> ()
    func.return
  }

  // Probe with a BORROWED transient key box (any hashable class; raises
  // TypeError for unhashable keys). Returns the slot index or -1.
  func.func @LyDict_LookupBox(%self: memref<8xi64> {ly.ownership.object_header}, %key_box: memref<5xi64>) -> i64 attributes {ly.runtime.contract = "builtins.dict", ly.runtime.primitive = "lookup_box"} {
    %box_idx = memref.extract_aligned_pointer_as_index %key_box : memref<5xi64> -> index
    %box_i64 = arith.index_cast %box_idx : index to i64
    %box_ptr = llvm.inttoptr %box_i64 : i64 to !llvm.ptr
    %hash_role_2 = arith.constant 1 : i64
    %hash = func.call @__ly_box_hash_key(%box_ptr, %hash_role_2) : (!llvm.ptr, i64) -> i64
    %slot = func.call @__ly_dict_probe(%self, %box_ptr, %hash) : (memref<8xi64>, !llvm.ptr, i64) -> i64
    func.return %slot : i64
  }

  // Probe that RAISES KeyError (repr message) when the key is missing: the
  // C++ paths call this so the transient key box is only ever read while the
  // call (and therefore the key's pin) is still live.
  func.func @LyDict_GetSlotOrRaise(%self: memref<8xi64> {ly.ownership.object_header}, %key_box: memref<5xi64>) -> i64 attributes {ly.runtime.contract = "builtins.dict", ly.runtime.primitive = "lookup_box_checked"} {
    %minus_one = arith.constant -1 : i64
    %slot = func.call @LyDict_LookupBox(%self, %key_box) : (memref<8xi64>, memref<5xi64>) -> i64
    %missing = arith.cmpi eq, %slot, %minus_one : i64
    scf.if %missing {
      %box_idx = memref.extract_aligned_pointer_as_index %key_box : memref<5xi64> -> index
      %box_i64 = arith.index_cast %box_idx : index to i64
      %box_ptr = llvm.inttoptr %box_i64 : i64 to !llvm.ptr
      func.call @__ly_dict_raise_missing_key(%box_ptr) : (!llvm.ptr) -> ()
    }
    func.return %slot : i64
  }

  // pop probe that raises on a miss (see LyDict_GetSlotOrRaise).
  func.func @LyDict_PopSlotChecked(%self: memref<8xi64> {ly.ownership.object_header}, %key_box: memref<5xi64>) -> i64 attributes {ly.runtime.contract = "builtins.dict", ly.runtime.primitive = "pop_slot_checked"} {
    %minus_one = arith.constant -1 : i64
    %slot = func.call @LyDict_PopSlot(%self, %key_box) : (memref<8xi64>, memref<5xi64>) -> i64
    %missing = arith.cmpi eq, %slot, %minus_one : i64
    scf.if %missing {
      %box_idx = memref.extract_aligned_pointer_as_index %key_box : memref<5xi64> -> index
      %box_i64 = arith.index_cast %box_idx : index to i64
      %box_ptr = llvm.inttoptr %box_i64 : i64 to !llvm.ptr
      func.call @__ly_dict_raise_missing_key(%box_ptr) : (!llvm.ptr) -> ()
    }
    func.return %slot : i64
  }

  // `key in d` with a BORROWED transient key box.
  func.func @LyDict_ContainsBox(%self: memref<8xi64> {ly.ownership.object_header}, %key_box: memref<5xi64>) -> i1 attributes {ly.runtime.contract = "builtins.dict", ly.runtime.primitive = "contains_box"} {
    %c0_i64 = arith.constant 0 : i64
    %slot = func.call @LyDict_LookupBox(%self, %key_box) : (memref<8xi64>, memref<5xi64>) -> i64
    %found = arith.cmpi sge, %slot, %c0_i64 : i64
    func.return %found : i1
  }

  // Delete with a BORROWED transient key box: KeyError(repr) on a miss;
  // otherwise release the entry and compact the dense tail so slot order
  // stays the insertion order the iteration paths walk.
  func.func @LyDict_DelItemBox(%self: memref<8xi64> {ly.ownership.object_header}, %key_box: memref<5xi64>) attributes {ly.runtime.contract = "builtins.dict", ly.runtime.primitive = "delitem_box"} {
    %absent_entry = arith.constant -1 : i64
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %minus_one = arith.constant -1 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16_words = func.call @__ly_box_word_count() : () -> i64
    %c16 = arith.index_cast %c16_words : i64 to index
    %length_slot = arith.constant 2 : index
    %keys = func.call @__ly_dict_keys(%self) : (memref<8xi64>) -> memref<?xi64>
    %values = func.call @__ly_dict_values(%self) : (memref<8xi64>) -> memref<?xi64>
    %present = func.call @__ly_dict_present(%self) : (memref<8xi64>) -> memref<?xi64>
    %slot = func.call @LyDict_LookupBox(%self, %key_box) : (memref<8xi64>, memref<5xi64>) -> i64
    %missing = arith.cmpi eq, %slot, %minus_one : i64
    scf.if %missing {
      %box_idx = memref.extract_aligned_pointer_as_index %key_box : memref<5xi64> -> index
      %box_i64 = arith.index_cast %box_idx : index to i64
      %box_ptr = llvm.inttoptr %box_i64 : i64 to !llvm.ptr
      func.call @__ly_dict_raise_missing_key(%box_ptr) : (!llvm.ptr) -> ()
    }
    %len = memref.load %self[%length_slot] : memref<8xi64>
    func.call @LyObject_ReleaseBoxedPayloadArraySlotRaw(%keys, %slot) : (memref<?xi64>, i64) -> ()
    func.call @LyObject_ReleaseBoxedPayloadArraySlotRaw(%values, %slot) : (memref<?xi64>, i64) -> ()
    // Shift the dense tail down one entry (16 words per box).
    %slot_index = arith.index_cast %slot : i64 to index
    %from = arith.addi %slot_index, %c1 : index
    %len_index = arith.index_cast %len : i64 to index
    scf.for %j = %from to %len_index step %c1 {
      %dst_entry = arith.subi %j, %c1 : index
      %src_base = arith.muli %j, %c16 : index
      %dst_base = arith.muli %dst_entry, %c16 : index
      scf.for %w = %c0 to %c16 step %c1 {
        %src = arith.addi %src_base, %w : index
        %dst = arith.addi %dst_base, %w : index
        %kw = memref.load %keys[%src] : memref<?xi64>
        %vw = memref.load %values[%src] : memref<?xi64>
        memref.store %kw, %keys[%dst] : memref<?xi64>
        memref.store %vw, %values[%dst] : memref<?xi64>
      }
    }
    func.call @__ly_dict_shift_hashes(%self, %from, %len_index) : (memref<8xi64>, index, index) -> ()
    %new_len = arith.subi %len, %one : i64
    %last = arith.index_cast %new_len : i64 to index
    %last_base = arith.muli %last, %c16 : index
    scf.for %w = %c0 to %c16 step %c1 {
      %dst = arith.addi %last_base, %w : index
      memref.store %zero, %keys[%dst] : memref<?xi64>
      memref.store %zero, %values[%dst] : memref<?xi64>
    }
    memref.store %absent_entry, %present[%last] : memref<?xi64>
    memref.store %new_len, %self[%length_slot] : memref<8xi64>
    func.return
  }

  // The key hashes follow their entries when the dense tail shifts down over
  // a removed one, or the table rebuilt from them would file every key after
  // it under its neighbour's hash.
  func.func private @__ly_dict_shift_hashes(%self: memref<8xi64>, %from: index, %len: index) {
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %hashes = func.call @__ly_dict_hashes(%self) : (memref<8xi64>) -> memref<?xi64>
    scf.for %j = %from to %len step %c1 {
      %dst = arith.subi %j, %c1 : index
      %hash = memref.load %hashes[%j] : memref<?xi64>
      memref.store %hash, %hashes[%dst] : memref<?xi64>
    }
    %last = arith.subi %len, %c1 : index
    %absent = arith.constant -1 : i64
    memref.store %absent, %hashes[%last] : memref<?xi64>
    func.return
  }

  // Runtime dict insert/replace with boxed key and value (any hashable key
  // class). The caller retained both boxes; on key replacement the duplicate
  // key box is consumed here. The computed hash is kept in the hashes array
  // beside the new entry.
  // Void and non-transfer, for the same reason as ensure_capacity: the growth
  // it may trigger updates the handle, and the entry it writes lands in an
  // array the handle points at. Nothing about the entity is renamed, so the
  // caller's reference is still the caller's after the call.
  // ⭐ The two boxes are this call's to place, so they are registered until
  // they are placed: hashing the key and comparing it against the table can
  // raise (an unhashable key, a user `__hash__` or `__eq__`), and the caller
  // has already handed their references over (errors.mlir, "what a native
  // body owes").
  func.func @LyDict_SetItemBox(%self: memref<8xi64> {ly.ownership.object_header}, %key_box: memref<5xi64>, %value_box: memref<5xi64>) attributes {ly.runtime.contract = "builtins.dict", ly.runtime.primitive = "setitem_box"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %minus_one = arith.constant -1 : i64
    %handle_words = func.call @__ly_box_word_count() : () -> i64
    %length_slot = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %len = memref.load %self[%length_slot] : memref<8xi64>
    %box_idx = memref.extract_aligned_pointer_as_index %key_box : memref<5xi64> -> index
    %box_i64 = arith.index_cast %box_idx : index to i64
    %box_ptr = llvm.inttoptr %box_i64 : i64 to !llvm.ptr
    %mark_slot = memref.alloca() : memref<1xi64>
    %mark_index = memref.extract_aligned_pointer_as_index %mark_slot : memref<1xi64> -> index
    %mark = arith.index_cast %mark_index : index to i64
    %value_idx = memref.extract_aligned_pointer_as_index %value_box : memref<5xi64> -> index
    %value_i64 = arith.index_cast %value_idx : index to i64
    %box_kind = arith.constant 1 : i64
    %pending = func.call @__ly_pending_push(%mark, %box_kind, %box_i64) : (i64, i64, i64) -> i64
    %pending_value = func.call @__ly_pending_push(%mark, %box_kind, %value_i64) : (i64, i64, i64) -> i64
    %hash_role_3 = arith.constant 1 : i64
    %hash = func.call @__ly_box_hash_key(%box_ptr, %hash_role_3) : (!llvm.ptr, i64) -> i64
    %found = func.call @__ly_dict_probe(%self, %box_ptr, %hash) : (memref<8xi64>, !llvm.ptr, i64) -> i64
    func.call @__ly_pending_pop(%pending) : (i64) -> ()

    %missing = arith.cmpi eq, %found, %minus_one : i64
    scf.if %missing {
      // Insert at slot len (all lower slots occupied).
      %required = arith.addi %len, %one : i64
      func.call @LyDict_EnsureCapacity(%self, %required) : (memref<8xi64>, i64) -> ()
      // Derived AFTER the growth: a view taken before it would name the freed
      // block. There is no lane that could have carried the stale one here.
      %keys = func.call @__ly_dict_keys(%self) : (memref<8xi64>) -> memref<?xi64>
      %values = func.call @__ly_dict_values(%self) : (memref<8xi64>) -> memref<?xi64>
      %present = func.call @__ly_dict_present(%self) : (memref<8xi64>) -> memref<?xi64>
      %slot_base_i64 = arith.muli %len, %handle_words : i64
      %box_words = arith.index_cast %handle_words : i64 to index
      %slot_base = arith.index_cast %slot_base_i64 : i64 to index
      scf.for %w = %c0 to %box_words step %c1 {
        %kw = memref.load %key_box[%w] : memref<5xi64>
        %vw = memref.load %value_box[%w] : memref<5xi64>
        %dst = arith.addi %slot_base, %w : index
        memref.store %kw, %keys[%dst] : memref<?xi64>
        memref.store %vw, %values[%dst] : memref<?xi64>
      }
      %len_slot_index = arith.index_cast %len : i64 to index
      %hashes = func.call @__ly_dict_hashes(%self) : (memref<8xi64>) -> memref<?xi64>
      memref.store %hash, %hashes[%len_slot_index] : memref<?xi64>
      memref.store %required, %self[%length_slot] : memref<8xi64>
      // The one writer that keeps the table in step instead of leaving it to
      // the stamp: a rebuild per insert would put the quadratic straight back.
      // The key is known absent here, so the slot is a clean one.
      %two_i64 = arith.constant 2 : i64
      %table = func.call @__ly_dict_table(%self) : (memref<8xi64>) -> memref<?xi8>
      %width = func.call @__ly_dict_self_index_width(%self) : (memref<8xi64>) -> i64
      %mask = func.call @__ly_dict_mask(%self) : (memref<8xi64>) -> i64
      %tslot = func.call @__ly_dict_clean_slot(%table, %width, %mask, %hash) : (memref<?xi8>, i64, i64, i64) -> i64
      %state = arith.addi %len, %two_i64 : i64
      func.call @__ly_dict_slot_store(%table, %width, %tslot, %state) : (memref<?xi8>, i64, i64, i64) -> ()
      %stamp = func.call @__ly_dict_table_stamp(%self) : (memref<8xi64>) -> memref<?xi64>
      memref.store %required, %stamp[%c0] : memref<?xi64>
    } else {
      // Replace: release the old value and the duplicate new key box.
      %values = func.call @__ly_dict_values(%self) : (memref<8xi64>) -> memref<?xi64>
      func.call @LyObject_ReleaseBoxedPayloadArraySlotRaw(%values, %found) : (memref<?xi64>, i64) -> ()
      func.call @LyObject_ReleaseBoxedPayloadRaw(%key_box) : (memref<5xi64>) -> ()
      %slot_base_i64 = arith.muli %found, %handle_words : i64
      %box_words = arith.index_cast %handle_words : i64 to index
      %slot_base = arith.index_cast %slot_base_i64 : i64 to index
      scf.for %w = %c0 to %box_words step %c1 {
        %vw = memref.load %value_box[%w] : memref<5xi64>
        %dst = arith.addi %slot_base, %w : index
        memref.store %vw, %values[%dst] : memref<?xi64>
      }
    }
    func.return
  }

  // dict.clear: release every present entry, zero the arrays, len = 0.
  func.func @LyDict_Clear(%self: memref<8xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "builtins.dict", ly.runtime.method = "clear"} {
    %absent_entry = arith.constant -1 : i64
    %absent_hash = arith.constant -1 : i64
    %zero = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16_words = func.call @__ly_box_word_count() : () -> i64
    %c16 = arith.index_cast %c16_words : i64 to index
    %length_slot = arith.constant 2 : index
    %keys = func.call @__ly_dict_keys(%self) : (memref<8xi64>) -> memref<?xi64>
    %values = func.call @__ly_dict_values(%self) : (memref<8xi64>) -> memref<?xi64>
    %present = func.call @__ly_dict_present(%self) : (memref<8xi64>) -> memref<?xi64>
    %len = memref.load %self[%length_slot] : memref<8xi64>
    %len_index = arith.index_cast %len : i64 to index
    scf.for %i = %c0 to %len_index step %c1 {
      %flag = memref.load %present[%i] : memref<?xi64>
      %is_present = arith.cmpi ne, %flag, %absent_hash : i64
      scf.if %is_present {
        %ii = arith.index_cast %i : index to i64
        func.call @LyObject_ReleaseBoxedPayloadArraySlotRaw(%keys, %ii) : (memref<?xi64>, i64) -> ()
        func.call @LyObject_ReleaseBoxedPayloadArraySlotRaw(%values, %ii) : (memref<?xi64>, i64) -> ()
      }
      memref.store %absent_entry, %present[%i] : memref<?xi64>
    }
    %total = arith.muli %len_index, %c16 : index
    scf.for %w = %c0 to %total step %c1 {
      memref.store %zero, %keys[%w] : memref<?xi64>
      memref.store %zero, %values[%w] : memref<?xi64>
    }
    memref.store %zero, %self[%length_slot] : memref<8xi64>
    func.return
  }

  // dict.copy: fresh arrays, every present entry's key and value retained.
  func.func @LyDict_Copy(%self: memref<8xi64> {ly.ownership.object_header}) -> memref<8xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.dict", ly.runtime.method = "copy", ly.runtime.result_contract = "builtins.dict"} {
    %absent_hash = arith.constant -1 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16_words = func.call @__ly_box_word_count() : () -> i64
    %c16 = arith.index_cast %c16_words : i64 to index
    %c2_slot = arith.constant 0 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %length_slot = arith.constant 2 : index
    %keys = func.call @__ly_dict_keys(%self) : (memref<8xi64>) -> memref<?xi64>
    %values = func.call @__ly_dict_values(%self) : (memref<8xi64>) -> memref<?xi64>
    %present = func.call @__ly_dict_present(%self) : (memref<8xi64>) -> memref<?xi64>
    %len = memref.load %self[%length_slot] : memref<8xi64>
    %fresh = func.call @__ly_dict_alloc(%len) : (i64) -> memref<8xi64>
    %fresh_keys = func.call @__ly_dict_keys(%fresh) : (memref<8xi64>) -> memref<?xi64>
    %fresh_values = func.call @__ly_dict_values(%fresh) : (memref<8xi64>) -> memref<?xi64>
    %fresh_present = func.call @__ly_dict_present(%fresh) : (memref<8xi64>) -> memref<?xi64>
    %hashes = func.call @__ly_dict_hashes(%self) : (memref<8xi64>) -> memref<?xi64>
    %fresh_hashes = func.call @__ly_dict_hashes(%fresh) : (memref<8xi64>) -> memref<?xi64>
    %len_index = arith.index_cast %len : i64 to index
    // Source entries are dense in runtime mode but may be sparse for
    // evidence-written dicts: compact while copying.
    %copied = scf.for %i = %c0 to %len_index step %c1 iter_args(%next = %c0) -> (index) {
      %flag = memref.load %present[%i] : memref<?xi64>
      %is_present = arith.cmpi ne, %flag, %absent_hash : i64
      %advanced = scf.if %is_present -> (index) {
        %src_base = arith.muli %i, %c16 : index
        %dst_base = arith.muli %next, %c16 : index
        scf.for %w = %c0 to %c16 step %c1 {
          %src = arith.addi %src_base, %w : index
          %dst = arith.addi %dst_base, %w : index
          %kw = memref.load %keys[%src] : memref<?xi64>
          %vw = memref.load %values[%src] : memref<?xi64>
          memref.store %kw, %fresh_keys[%dst] : memref<?xi64>
          memref.store %vw, %fresh_values[%dst] : memref<?xi64>
        }
        %key_entity_slot = arith.addi %dst_base, %c2_slot : index
        %key_entity = memref.load %fresh_keys[%key_entity_slot] : memref<?xi64>
        %value_entity = memref.load %fresh_values[%key_entity_slot] : memref<?xi64>
        func.call @__ly_handle_retain_raw(%key_entity) : (i64) -> ()
        func.call @__ly_handle_retain_raw(%value_entity) : (i64) -> ()
        %copied_hash = memref.load %hashes[%i] : memref<?xi64>
        memref.store %copied_hash, %fresh_hashes[%next] : memref<?xi64>
        %incremented = arith.addi %next, %c1 : index
        scf.yield %incremented : index
      } else {
        scf.yield %next : index
      }
      scf.yield %advanced : index
    }
    %copied_i64 = arith.index_cast %copied : index to i64
    memref.store %copied_i64, %fresh[%length_slot] : memref<8xi64>
    func.return %fresh : memref<8xi64>
  }

  // Insert-or-replace one entry (raw source boxes) into a dict; the source
  // key/value references are retained here. Returns the (possibly
  // reallocated) representation.
  func.func private @__ly_dict_store_from_slot(%self: memref<8xi64> {ly.ownership.object_header}, %src_keys: memref<?xi64>, %src_values: memref<?xi64>, %src_hashes: memref<?xi64>, %src_slot: index) {
    %minus_one = arith.constant -1 : i64
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16_words = func.call @__ly_box_word_count() : () -> i64
    %c16 = arith.index_cast %c16_words : i64 to index
    %c16_i64 = func.call @__ly_box_word_count() : () -> i64
    %c2_slot = arith.constant 0 : index
    %src_idx = memref.extract_aligned_pointer_as_index %src_keys : memref<?xi64> -> index
    %src_i64 = arith.index_cast %src_idx : index to i64
    %src_ptr = llvm.inttoptr %src_i64 : i64 to !llvm.ptr
    %slot_i64 = arith.index_cast %src_slot : index to i64
    %off = arith.muli %slot_i64, %c16_i64 : i64
    %key_entry = llvm.getelementptr %src_ptr[%off] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %cached = memref.load %src_hashes[%src_slot] : memref<?xi64>
    %unknown = arith.cmpi eq, %cached, %zero : i64
    %hash = scf.if %unknown -> (i64) {
      %hash_role_4 = arith.constant 1 : i64
      %computed = func.call @__ly_box_hash_key(%key_entry, %hash_role_4) : (!llvm.ptr, i64) -> i64
      memref.store %computed, %src_hashes[%src_slot] : memref<?xi64>
      scf.yield %computed : i64
    } else {
      scf.yield %cached : i64
    }
    %found = func.call @__ly_dict_probe(%self, %key_entry, %hash) : (memref<8xi64>, !llvm.ptr, i64) -> i64
    %missing = arith.cmpi eq, %found, %minus_one : i64
    %len_slot = arith.constant 2 : index
    scf.if %missing {
      %len = memref.load %self[%len_slot] : memref<8xi64>
      %required = arith.addi %len, %one : i64
      func.call @LyDict_EnsureCapacity(%self, %required) : (memref<8xi64>, i64) -> ()
      // Derived after the growth: the pre-growth views name freed blocks. No
      // lane could have carried them here, which is the point.
      %grown_keys = func.call @__ly_dict_keys(%self) : (memref<8xi64>) -> memref<?xi64>
      %grown_values = func.call @__ly_dict_values(%self) : (memref<8xi64>) -> memref<?xi64>
      %grown_present = func.call @__ly_dict_present(%self) : (memref<8xi64>) -> memref<?xi64>
      %dst_entry = arith.index_cast %len : i64 to index
      %src_base = arith.muli %src_slot, %c16 : index
      %dst_base = arith.muli %dst_entry, %c16 : index
      scf.for %w = %c0 to %c16 step %c1 {
        %src = arith.addi %src_base, %w : index
        %dst = arith.addi %dst_base, %w : index
        %kw = memref.load %src_keys[%src] : memref<?xi64>
        %vw = memref.load %src_values[%src] : memref<?xi64>
        memref.store %kw, %grown_keys[%dst] : memref<?xi64>
        memref.store %vw, %grown_values[%dst] : memref<?xi64>
      }
      %entity_index = arith.addi %dst_base, %c2_slot : index
      %key_entity = memref.load %grown_keys[%entity_index] : memref<?xi64>
      %value_entity = memref.load %grown_values[%entity_index] : memref<?xi64>
      func.call @__ly_handle_retain_raw(%key_entity) : (i64) -> ()
      func.call @__ly_handle_retain_raw(%value_entity) : (i64) -> ()
      %grown_hashes = func.call @__ly_dict_hashes(%self) : (memref<8xi64>) -> memref<?xi64>
      memref.store %hash, %grown_hashes[%dst_entry] : memref<?xi64>
      memref.store %required, %self[%len_slot] : memref<8xi64>
    } else {
      // Replace: release the old value, copy + retain the new one.
      %values = func.call @__ly_dict_values(%self) : (memref<8xi64>) -> memref<?xi64>
      func.call @LyObject_ReleaseBoxedPayloadArraySlotRaw(%values, %found) : (memref<?xi64>, i64) -> ()
      %dst_entry = arith.index_cast %found : i64 to index
      %src_base = arith.muli %src_slot, %c16 : index
      %dst_base = arith.muli %dst_entry, %c16 : index
      scf.for %w = %c0 to %c16 step %c1 {
        %src = arith.addi %src_base, %w : index
        %dst = arith.addi %dst_base, %w : index
        %vw = memref.load %src_values[%src] : memref<?xi64>
        memref.store %vw, %values[%dst] : memref<?xi64>
      }
      %entity_index = arith.addi %dst_base, %c2_slot : index
      %value_entity = memref.load %values[%entity_index] : memref<?xi64>
      func.call @__ly_handle_retain_raw(%value_entity) : (i64) -> ()
    }
    func.return
  }

  // dict.update(other): insert-or-replace every present entry of other.
  // dict.update returns None in CPython. It had a dict result here only
  // because the five-lane form spelled every reallocating mutation as "consume
  // the entity, hand back a new tuple" -- and that spelling is what forced the
  // loop-carried re-description below. With interior state behind the handle
  // the loop carries nothing, and the affine verifier has no transfer to
  // reconcile against a back edge.
  func.func @LyDict_Update(%self: memref<8xi64> {ly.ownership.object_header}, %other: memref<8xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "builtins.dict", ly.runtime.method = "update"} {
    %absent_hash = arith.constant -1 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %length_slot = arith.constant 2 : index
    %ok = func.call @__ly_dict_keys(%other) : (memref<8xi64>) -> memref<?xi64>
    %ov = func.call @__ly_dict_values(%other) : (memref<8xi64>) -> memref<?xi64>
    %op = func.call @__ly_dict_present(%other) : (memref<8xi64>) -> memref<?xi64>
    %oh = func.call @__ly_dict_hashes(%other) : (memref<8xi64>) -> memref<?xi64>
    %olen = memref.load %other[%length_slot] : memref<8xi64>
    %olen_index = arith.index_cast %olen : i64 to index
    scf.for %i = %c0 to %olen_index step %c1 {
      %flag = memref.load %op[%i] : memref<?xi64>
      %is_present = arith.cmpi ne, %flag, %absent_hash : i64
      scf.if %is_present {
        func.call @__ly_dict_store_from_slot(%self, %ok, %ov, %oh, %i) : (memref<8xi64>, memref<?xi64>, memref<?xi64>, memref<?xi64>, index) -> ()
      }
    }
    func.return
  }

  // dict | dict: copy of lhs updated with rhs.
  func.func @LyDict_OrOp(%self: memref<8xi64> {ly.ownership.object_header}, %other: memref<8xi64> {ly.ownership.object_header}) -> memref<8xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.dict", ly.runtime.method = "__or__", ly.runtime.result_contract = "builtins.dict"} {
    %copy = func.call @LyDict_Copy(%self) : (memref<8xi64>) -> memref<8xi64>
    func.call @LyDict_Update(%copy, %other) : (memref<8xi64>, memref<8xi64>) -> ()
    func.return %copy : memref<8xi64>
  }

  // dict == dict: same live size and every lhs entry matches in rhs.
  func.func @LyDict_EqBool(%self: memref<8xi64> {ly.ownership.object_header}, %other: memref<8xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.dict", ly.runtime.method = "__eq__"} {
    %absent_hash = arith.constant -1 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16_i64 = func.call @__ly_box_word_count() : () -> i64
    %minus_one = arith.constant -1 : i64
    %zero = arith.constant 0 : i64
    %true = arith.constant true
    %false = arith.constant false
    %length_slot = arith.constant 2 : index
    %keys = func.call @__ly_dict_keys(%self) : (memref<8xi64>) -> memref<?xi64>
    %values = func.call @__ly_dict_values(%self) : (memref<8xi64>) -> memref<?xi64>
    %present = func.call @__ly_dict_present(%self) : (memref<8xi64>) -> memref<?xi64>
    %ov = func.call @__ly_dict_values(%other) : (memref<8xi64>) -> memref<?xi64>
    %llen = memref.load %self[%length_slot] : memref<8xi64>
    %rlen = memref.load %other[%length_slot] : memref<8xi64>
    %same_len = arith.cmpi eq, %llen, %rlen : i64
    %result = scf.if %same_len -> (i1) {
      %llen_index = arith.index_cast %llen : i64 to index
      %keys_idx = memref.extract_aligned_pointer_as_index %keys : memref<?xi64> -> index
      %keys_i64 = arith.index_cast %keys_idx : index to i64
      %keys_ptr = llvm.inttoptr %keys_i64 : i64 to !llvm.ptr
      %values_idx = memref.extract_aligned_pointer_as_index %values : memref<?xi64> -> index
      %values_i64 = arith.index_cast %values_idx : index to i64
      %values_ptr = llvm.inttoptr %values_i64 : i64 to !llvm.ptr
      %ovalues_idx = memref.extract_aligned_pointer_as_index %ov : memref<?xi64> -> index
      %ovalues_i64 = arith.index_cast %ovalues_idx : index to i64
      %ovalues_ptr = llvm.inttoptr %ovalues_i64 : i64 to !llvm.ptr
      %all = scf.for %i = %c0 to %llen_index step %c1 iter_args(%acc = %true) -> (i1) {
        %next = scf.if %acc -> (i1) {
          %flag = memref.load %present[%i] : memref<?xi64>
          %is_present = arith.cmpi ne, %flag, %absent_hash : i64
          %entry_ok = scf.if %is_present -> (i1) {
            %ii = arith.index_cast %i : index to i64
            %off = arith.muli %ii, %c16_i64 : i64
            %key_entry = llvm.getelementptr %keys_ptr[%off] : (!llvm.ptr, i64) -> !llvm.ptr, i64
            %hash_role_5 = arith.constant 1 : i64
            %key_hash = func.call @__ly_box_hash_key(%key_entry, %hash_role_5) : (!llvm.ptr, i64) -> i64
            %slot = func.call @__ly_dict_probe(%other, %key_entry, %key_hash) : (memref<8xi64>, !llvm.ptr, i64) -> i64
            %found = arith.cmpi ne, %slot, %minus_one : i64
            %value_ok = scf.if %found -> (i1) {
              %lvalue = llvm.getelementptr %values_ptr[%off] : (!llvm.ptr, i64) -> !llvm.ptr, i64
              %roff = arith.muli %slot, %c16_i64 : i64
              %rvalue = llvm.getelementptr %ovalues_ptr[%roff] : (!llvm.ptr, i64) -> !llvm.ptr, i64
              %veq = func.call @__ly_box_equal(%lvalue, %rvalue) : (!llvm.ptr, !llvm.ptr) -> i1
              scf.yield %veq : i1
            } else {
              scf.yield %false : i1
            }
            scf.yield %value_ok : i1
          } else {
            scf.yield %true : i1
          }
          scf.yield %entry_ok : i1
        } else {
          scf.yield %false : i1
        }
        scf.yield %next : i1
      }
      scf.yield %all : i1
    } else {
      scf.yield %false : i1
    }
    func.return %result : i1
  }

  func.func @LyDict_NeBool(%self: memref<8xi64> {ly.ownership.object_header}, %other: memref<8xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.dict", ly.runtime.method = "__ne__"} {
    %eq = func.call @LyDict_EqBool(%self, %other) : (memref<8xi64>, memref<8xi64>) -> i1
    %true = arith.constant true
    %ne = arith.xori %eq, %true : i1
    func.return %ne : i1
  }

  // Release the reference parked at a values-array slot (dict.pop's caller
  // side runs this AFTER retaining the popped value into its own binding).
  func.func @LyDict_ReleaseParked(%self: memref<8xi64> {ly.ownership.object_header}, %slot: i64) attributes {ly.runtime.contract = "builtins.dict", ly.runtime.primitive = "release_parked"} {
    %values = func.call @__ly_dict_values(%self) : (memref<8xi64>) -> memref<?xi64>
    func.call @LyObject_ReleaseBoxedPayloadArraySlotRaw(%values, %slot) : (memref<?xi64>, i64) -> ()
    func.return
  }

  // dict.pop support: remove the probed entry WITHOUT releasing its value;
  // the popped value box is parked in the (now free) slot past the new end
  // of the values array for the caller to read. Returns the entry's former
  // slot, or -1 when the key is missing.
  func.func @LyDict_PopSlot(%self: memref<8xi64> {ly.ownership.object_header}, %key_box: memref<5xi64>) -> i64 attributes {ly.runtime.contract = "builtins.dict", ly.runtime.primitive = "pop_slot"} {
    %absent_entry = arith.constant -1 : i64
    %minus_one = arith.constant -1 : i64
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16_words = func.call @__ly_box_word_count() : () -> i64
    %c16 = arith.index_cast %c16_words : i64 to index
    %length_slot = arith.constant 2 : index
    %keys = func.call @__ly_dict_keys(%self) : (memref<8xi64>) -> memref<?xi64>
    %values = func.call @__ly_dict_values(%self) : (memref<8xi64>) -> memref<?xi64>
    %present = func.call @__ly_dict_present(%self) : (memref<8xi64>) -> memref<?xi64>
    %slot = func.call @LyDict_LookupBox(%self, %key_box) : (memref<8xi64>, memref<5xi64>) -> i64
    %missing = arith.cmpi eq, %slot, %minus_one : i64
    scf.if %missing {
    } else {
      %len = memref.load %self[%length_slot] : memref<8xi64>
      %new_len = arith.subi %len, %one : i64
      %park = arith.index_cast %new_len : i64 to index
      %park_base = arith.muli %park, %c16 : index
      // Park the popped value's box words in scratch before the tail shift
      // overwrites the slot (the park slot itself is inside the shifted
      // range, so stage through locals).
      %scratch = memref.alloca() : memref<5xi64>
      %slot_index = arith.index_cast %slot : i64 to index
      %slot_base = arith.muli %slot_index, %c16 : index
      scf.for %w = %c0 to %c16 step %c1 {
        %src = arith.addi %slot_base, %w : index
        %word = memref.load %values[%src] : memref<?xi64>
        memref.store %word, %scratch[%w] : memref<5xi64>
      }
      // Release the key only; the value's reference transfers to the caller.
      func.call @LyObject_ReleaseBoxedPayloadArraySlotRaw(%keys, %slot) : (memref<?xi64>, i64) -> ()
      %from = arith.addi %slot_index, %c1 : index
      %len_index = arith.index_cast %len : i64 to index
      scf.for %j = %from to %len_index step %c1 {
        %dst_entry = arith.subi %j, %c1 : index
        %src_base = arith.muli %j, %c16 : index
        %dst_base = arith.muli %dst_entry, %c16 : index
        scf.for %w = %c0 to %c16 step %c1 {
          %src = arith.addi %src_base, %w : index
          %dst = arith.addi %dst_base, %w : index
          %kw = memref.load %keys[%src] : memref<?xi64>
          %vw = memref.load %values[%src] : memref<?xi64>
          memref.store %kw, %keys[%dst] : memref<?xi64>
          memref.store %vw, %values[%dst] : memref<?xi64>
        }
      }
      func.call @__ly_dict_shift_hashes(%self, %from, %len_index) : (memref<8xi64>, index, index) -> ()
      scf.for %w = %c0 to %c16 step %c1 {
        %dst = arith.addi %park_base, %w : index
        %kw_zero = arith.constant 0 : i64
        memref.store %kw_zero, %keys[%dst] : memref<?xi64>
        %word = memref.load %scratch[%w] : memref<5xi64>
        memref.store %word, %values[%dst] : memref<?xi64>
      }
      memref.store %absent_entry, %present[%park] : memref<?xi64>
      memref.store %new_len, %self[%length_slot] : memref<8xi64>
    }
    func.return %slot : i64
  }

  // dict.__repr__: `{k0: v0, k1: v1}` over present slots (capacity order); key
  // and value each repr'd through the uniform boxed-method hook. Same manual
  // loop-ownership as the sequence reprs (Concat borrows).
  func.func @LyDict_Repr(%self: memref<8xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.dict", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %absent_hash = arith.constant -1 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c16_i64 = func.call @__ly_box_word_count() : () -> i64
    %capacity_slot = arith.constant 3 : index
    %keys_slot = arith.constant 4 : index
    %values_slot = arith.constant 5 : index
    %present = func.call @__ly_dict_present(%self) : (memref<8xi64>) -> memref<?xi64>
    %capacity = memref.load %self[%capacity_slot] : memref<8xi64>
    %capacity_idx = arith.index_cast %capacity : i64 to index
    %keys_i64 = memref.load %self[%keys_slot] : memref<8xi64>
    %keys_ptr = llvm.inttoptr %keys_i64 : i64 to !llvm.ptr
    %values_i64 = memref.load %self[%values_slot] : memref<8xi64>
    %values_ptr = llvm.inttoptr %values_i64 : i64 to !llvm.ptr

    %open_ref = memref.get_global @__ly_repr_lbrace : memref<1xi8>
    %open_dyn = memref.cast %open_ref : memref<1xi8> to memref<?xi8>
    %r0_h, %r0_b = func.call @__ly_unicode_from_valid_utf8(%open_dyn, %c0, %c1_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)

    %loop:3 = scf.for %i = %c0 to %capacity_idx step %c1 iter_args(%rh = %r0_h, %rb = %r0_b, %emitted = %c0_i64) -> (memref<2xi64>, memref<?xi8>, i64) {
      %slot = memref.load %present[%i] : memref<?xi64>
      %is_present = arith.cmpi ne, %slot, %absent_hash : i64
      %entry:3 = scf.if %is_present -> (memref<2xi64>, memref<?xi8>, i64) {
        %i_i64 = arith.index_cast %i : index to i64
        // separator ", " when this is not the first emitted entry
        %has_prev = arith.cmpi sgt, %emitted, %c0_i64 : i64
        %sep:2 = scf.if %has_prev -> (memref<2xi64>, memref<?xi8>) {
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
        // key repr
        %kbox = llvm.getelementptr %keys_ptr[%off] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        %kclass_word = llvm.load %kbox : !llvm.ptr -> i64
        %kclass = func.call @__ly_slot_class(%kclass_word) : (i64) -> i64
        %krh, %krb = func.call @__ly_repr_boxed_or_default(%kbox, %kclass) : (!llvm.ptr, i64) -> (memref<2xi64>, memref<?xi8>)
        %k1h, %k1b = func.call @LyUnicode_Concat(%sep#0, %sep#1, %krh, %krb) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
        func.call @LyUnicode_DecRef(%sep#0) : (memref<2xi64>) -> ()
        func.call @LyUnicode_DecRef(%krh) : (memref<2xi64>) -> ()
        // ": "
        %colon_ref = memref.get_global @__ly_repr_colon : memref<2xi8>
        %colon_dyn = memref.cast %colon_ref : memref<2xi8> to memref<?xi8>
        %coh, %cob = func.call @__ly_unicode_from_valid_utf8(%colon_dyn, %c0, %c2_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
        %k2h, %k2b = func.call @LyUnicode_Concat(%k1h, %k1b, %coh, %cob) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
        func.call @LyUnicode_DecRef(%k1h) : (memref<2xi64>) -> ()
        func.call @LyUnicode_DecRef(%coh) : (memref<2xi64>) -> ()
        // value repr
        %vbox = llvm.getelementptr %values_ptr[%off] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        %vclass_word = llvm.load %vbox : !llvm.ptr -> i64
        %vclass = func.call @__ly_slot_class(%vclass_word) : (i64) -> i64
        %vrh, %vrb = func.call @__ly_repr_boxed_or_default(%vbox, %vclass) : (!llvm.ptr, i64) -> (memref<2xi64>, memref<?xi8>)
        %k3h, %k3b = func.call @LyUnicode_Concat(%k2h, %k2b, %vrh, %vrb) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
        func.call @LyUnicode_DecRef(%k2h) : (memref<2xi64>) -> ()
        func.call @LyUnicode_DecRef(%vrh) : (memref<2xi64>) -> ()
        %next_emitted = arith.addi %emitted, %c1_i64 : i64
        scf.yield %k3h, %k3b, %next_emitted : memref<2xi64>, memref<?xi8>, i64
      } else {
        scf.yield %rh, %rb, %emitted : memref<2xi64>, memref<?xi8>, i64
      }
      scf.yield %entry#0, %entry#1, %entry#2 : memref<2xi64>, memref<?xi8>, i64
    }

    %close_ref = memref.get_global @__ly_repr_rbrace : memref<1xi8>
    %close_dyn = memref.cast %close_ref : memref<1xi8> to memref<?xi8>
    %clh, %clb = func.call @__ly_unicode_from_valid_utf8(%close_dyn, %c0, %c1_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %out_h, %out_b = func.call @LyUnicode_Concat(%loop#0, %loop#1, %clh, %clb) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.call @LyUnicode_DecRef(%loop#0) : (memref<2xi64>) -> ()
    func.call @LyUnicode_DecRef(%clh) : (memref<2xi64>) -> ()
    func.return %out_h, %out_b : memref<2xi64>, memref<?xi8>
  }

  // One operand, because there is one entity. The five-operand spelling was
  // the deallocator being handed the entity's whole representation, which is
  // also what made `findDeallocatorForValueGroup` disambiguate deallocators by
  // matching a TUPLE OF TYPES -- rfc/memory-safety-proof.md calls that the
  // negation of the `Provenance` rule rather than a narrower version of it.
  func.func @LyDict_DecRef(%self: memref<8xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.dict", ly.runtime.deallocator} {
    %absent_hash = arith.constant -1 : i64
    %storage = memref.cast %self : memref<8xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    %capacity_slot = arith.constant 3 : index
    %keys_slot = arith.constant 4 : index
    %values_slot = arith.constant 5 : index
    %present_slot = arith.constant 6 : index
    %lower = arith.constant 0 : index
    %step = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %keys = func.call @__ly_dict_keys(%self) : (memref<8xi64>) -> memref<?xi64>
    %values = func.call @__ly_dict_values(%self) : (memref<8xi64>) -> memref<?xi64>
    %present = func.call @__ly_dict_present(%self) : (memref<8xi64>) -> memref<?xi64>
    %capacity = memref.load %self[%capacity_slot] : memref<8xi64>
    %capacity_index = arith.index_cast %capacity : i64 to index
    scf.for %i = %lower to %capacity_index step %step {
      %occupied = memref.load %present[%i] : memref<?xi64>
      %is_present = arith.cmpi ne, %occupied, %absent_hash : i64
      scf.if %is_present {
        %logical_index = arith.index_cast %i : index to i64
        func.call @LyObject_ReleaseBoxedPayloadArraySlotRaw(%keys, %logical_index) : (memref<?xi64>, i64) -> ()
        func.call @LyObject_ReleaseBoxedPayloadArraySlotRaw(%values, %logical_index) : (memref<?xi64>, i64) -> ()
      }
    }
    // Keys sit at offset zero of the one block the four arrays share; an
    // empty dict has none of its own.
    %keys_word = memref.load %self[%keys_slot] : memref<8xi64>
    %had_block = arith.cmpi sgt, %capacity, %zero : i64
    scf.if %had_block {
      func.call @free_raw_i64_ptr(%keys_word) : (i64) -> ()
    }
    memref.dealloc %self : memref<8xi64>
    cf.br ^done

  ^done:
    func.return
  }

  // The dict's own table walk: one state word per slot (0 unused, 1 dummy,
  // dense index + 2), the hash read from the dict's hashes array. The probe
  // sequence is `__ly_table_lookup`'s, which the set keeps with the hash
  // beside each state; ⛔ the dict does not, because its hashes array already
  // holds them and a second copy was 16 bytes per slot of table.
  func.func private @__ly_dict_lookup(%table: memref<?xi8>, %width: i64, %mask: i64, %items_ptr: !llvm.ptr, %hashes: memref<?xi64>, %elem_box: !llvm.ptr, %hash: i64) -> i64 {
    %minus_one = arith.constant -1 : i64
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %probe_scale = arith.constant 5 : i64
    %nine = arith.constant 9 : i64
    %c16 = func.call @__ly_box_word_count() : () -> i64
    %shift = arith.constant 5 : i64
    %true = arith.constant true
    %false = arith.constant false
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %i0 = arith.andi %hash, %mask : i64
    %walk:4 = scf.while (%i = %i0, %p = %hash, %ans = %minus_one, %done = %false)
        : (i64, i64, i64, i1) -> (i64, i64, i64, i1) {
      %again = arith.xori %done, %true : i1
      scf.condition(%again) %i, %p, %ans, %done : i64, i64, i64, i1
    } do {
    ^body(%i: i64, %p: i64, %ans: i64, %done: i1):
      %limit = arith.addi %i, %nine : i64
      %linear = arith.cmpi ule, %limit, %mask : i64
      %probes = arith.select %linear, %nine, %zero : i1, i64
      %count = arith.addi %probes, %one : i64
      %count_index = arith.index_cast %count : i64 to index
      %run:2 = scf.for %k = %c0 to %count_index step %c1
          iter_args(%a = %ans, %d = %done) -> (i64, i1) {
        %step:2 = scf.if %d -> (i64, i1) {
          scf.yield %a, %d : i64, i1
        } else {
          %kk = arith.index_cast %k : index to i64
          %s = arith.addi %i, %kk : i64
          %state = func.call @__ly_dict_slot_load(%table, %width, %s) : (memref<?xi8>, i64, i64) -> i64
          %unused = arith.cmpi eq, %state, %zero : i64
          %seen:2 = scf.if %unused -> (i64, i1) {
            scf.yield %minus_one, %true : i64, i1
          } else {
            %live = arith.cmpi sge, %state, %two : i64
            %hit:2 = scf.if %live -> (i64, i1) {
              %dense_h = arith.subi %state, %two : i64
              %hash_index = arith.index_cast %dense_h : i64 to index
              %entry_hash = memref.load %hashes[%hash_index] : memref<?xi64>
              %same_hash = arith.cmpi eq, %entry_hash, %hash : i64
              %cmp:2 = scf.if %same_hash -> (i64, i1) {
                %dense = arith.subi %state, %two : i64
                %off = arith.muli %dense, %c16 : i64
                %entry = llvm.getelementptr %items_ptr[%off] : (!llvm.ptr, i64) -> !llvm.ptr, i64
                %eq = func.call @__ly_box_equal(%entry, %elem_box) : (!llvm.ptr, !llvm.ptr) -> i1
                %found = arith.select %eq, %dense, %a : i1, i64
                scf.yield %found, %eq : i64, i1
              } else {
                scf.yield %a, %false : i64, i1
              }
              scf.yield %cmp#0, %cmp#1 : i64, i1
            } else {
              scf.yield %a, %false : i64, i1
            }
            scf.yield %hit#0, %hit#1 : i64, i1
          }
          scf.yield %seen#0, %seen#1 : i64, i1
        }
        scf.yield %step#0, %step#1 : i64, i1
      }
      %np = arith.shrui %p, %shift : i64
      %i5 = arith.muli %i, %probe_scale : i64
      %i51 = arith.addi %i5, %one : i64
      %i5p = arith.addi %i51, %np : i64
      %ni = arith.andi %i5p, %mask : i64
      scf.yield %ni, %np, %run#0, %run#1 : i64, i64, i64, i1
    }
    func.return %walk#2 : i64
  }
  func.func private @__ly_dict_clean_slot(%table: memref<?xi8>, %width: i64, %mask: i64, %hash: i64) -> i64 {
    %minus_one = arith.constant -1 : i64
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %probe_scale = arith.constant 5 : i64
    %nine = arith.constant 9 : i64
    %shift = arith.constant 5 : i64
    %true = arith.constant true
    %false = arith.constant false
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %i0 = arith.andi %hash, %mask : i64
    %walk:4 = scf.while (%i = %i0, %p = %hash, %ans = %minus_one, %done = %false)
        : (i64, i64, i64, i1) -> (i64, i64, i64, i1) {
      %again = arith.xori %done, %true : i1
      scf.condition(%again) %i, %p, %ans, %done : i64, i64, i64, i1
    } do {
    ^body(%i: i64, %p: i64, %ans: i64, %done: i1):
      %limit = arith.addi %i, %nine : i64
      %linear = arith.cmpi ule, %limit, %mask : i64
      %probes = arith.select %linear, %nine, %zero : i1, i64
      %count = arith.addi %probes, %one : i64
      %count_index = arith.index_cast %count : i64 to index
      %run:2 = scf.for %k = %c0 to %count_index step %c1
          iter_args(%a = %ans, %d = %done) -> (i64, i1) {
        %step:2 = scf.if %d -> (i64, i1) {
          scf.yield %a, %d : i64, i1
        } else {
          %kk = arith.index_cast %k : index to i64
          %s = arith.addi %i, %kk : i64
          %state = func.call @__ly_dict_slot_load(%table, %width, %s) : (memref<?xi8>, i64, i64) -> i64
          %unused = arith.cmpi eq, %state, %zero : i64
          %pick = arith.select %unused, %s, %a : i1, i64
          scf.yield %pick, %unused : i64, i1
        }
        scf.yield %step#0, %step#1 : i64, i1
      }
      %np = arith.shrui %p, %shift : i64
      %i5 = arith.muli %i, %probe_scale : i64
      %i51 = arith.addi %i5, %one : i64
      %i5p = arith.addi %i51, %np : i64
      %ni = arith.andi %i5p, %mask : i64
      scf.yield %ni, %np, %run#0, %run#1 : i64, i64, i64, i1
    }
    func.return %walk#2 : i64
  }
}
