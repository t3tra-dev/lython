// `set` and `frozenset` -- CPython's Objects/setobject.c.
//
// Signature source (1:1 correspondence target):
//   https://github.com/python/typeshed/blob/main/stdlib/builtins.pyi

module attributes {
  ly.typing.manifest,
  ly.runtime.contracts = ["builtins.set"]
} {
  // ===== declared here, defined in another runtime file or built by the lowering =====
  func.func private @LyObject_ReleaseBoxedPayloadArraySlotRaw(%payload: memref<?xi64>, %logical_index: i64)
  func.func private @LyObject_ReleaseBoxedPayloadRaw(%box: memref<5xi64>)
  func.func private @LyObject_ReleaseStorageToZero(%storage: memref<?xi64>) -> i1 attributes {ly.runtime.contract = "builtins.object", ly.runtime.primitive = "release_to_zero"}
  func.func private @LyUnicode_Concat(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_bytes: memref<?xi8>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__add__"}
  func.func private @LyUnicode_DecRef(%header: memref<2xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.str", ly.runtime.deallocator}
  func.func private @LyUnicode_FromBytes(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 4 : i64, ly.runtime.contract = "builtins.str", ly.runtime.initializer = "__new__"}
  func.func private @__ly_box_equal(%lhs: !llvm.ptr, %rhs: !llvm.ptr) -> i1
  func.func private @__ly_box_hash(%box: !llvm.ptr) -> i64
  func.func private @__ly_box_move_slot(%dst: memref<?xi64>, %d: i64, %src: memref<?xi64>, %s: i64)
  func.func private @__ly_box_word_count() -> i64
  func.func private @__ly_dict_raise_missing_key(%key_box: !llvm.ptr) attributes {ly.runtime.contract = "builtins.dict", ly.runtime.primitive = "raise_missing_key_ptr"}
  func.func private @__ly_global_view_i64(%pointer: i64, %size: i64) -> memref<?xi64>
  func.func private @__ly_handle_retain_raw(%entity: i64)
  func.func private @__ly_repr_boxed_or_default(%box_ptr: !llvm.ptr, %class_id: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.object", ly.runtime.primitive = "repr_boxed_or_default", ly.runtime.result_contract = "builtins.str"}
  memref.global "private" constant @__ly_repr_comma : memref<2xi8>
  memref.global "private" constant @__ly_repr_frozenset_empty : memref<11xi8>
  memref.global "private" constant @__ly_repr_frozenset_open : memref<11xi8>
  memref.global "private" constant @__ly_repr_lbrace : memref<1xi8>
  memref.global "private" constant @__ly_repr_rbrace : memref<1xi8>
  memref.global "private" constant @__ly_repr_rparen : memref<1xi8>
  memref.global "private" constant @__ly_repr_set_empty : memref<5xi8>
  func.func private @__ly_slot_class(%word: i64) -> i64
  func.func private @free_raw_i64_ptr(%address: i64)

  py.class @set attributes {
    base_names = ["MutableSet"], ly.typing.params = ["T"],
    ly.typing.base_args = [[!py.contract<"$T">]],
    ly.runtime.contract = "builtins.set", ly.runtime.required,
    ly.runtime.required_deallocator,
    ly.runtime.required_initializers = ["__new__"],
    ly.runtime.required_methods = ["__len__"],
    ly.typing.structural_mutators = ["add", "update", "intersection_update",
                                     "difference_update",
                                     "symmetric_difference_update"],
    method_names = ["__init__", "add", "__len__", "__iter__",
                    "__contains__", "discard", "remove", "clear", "copy",
                    "update", "intersection_update", "difference_update",
                    "symmetric_difference_update",
                    "union", "intersection", "difference",
                    "symmetric_difference", "issubset", "issuperset",
                    "isdisjoint", "__eq__", "__ne__", "__or__", "__and__",
                    "__sub__", "__xor__", "__le__", "__lt__", "__ge__",
                    "__gt__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"builtins.set">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.set">, !py.contract<"$T">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.set">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.set">] -> [!py.protocol<"Iterator", [!py.contract<"$T">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.set">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.set">, !py.contract<"$T">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.set">, !py.contract<"$T">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.set">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.set">] -> [!py.contract<"builtins.set", [!py.contract<"$T">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.set">, !py.protocol<"Iterable", [!py.contract<"$T">]>] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.set">, !py.protocol<"Iterable", [!py.contract<"$T">]>] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.set">, !py.protocol<"Iterable", [!py.contract<"$T">]>] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.set">, !py.protocol<"Iterable", [!py.contract<"$T">]>] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.set">, !py.protocol<"Iterable", [!py.contract<"$T">]>] -> [!py.contract<"builtins.set", [!py.contract<"$T">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.set">, !py.protocol<"Iterable", [!py.contract<"$T">]>] -> [!py.contract<"builtins.set", [!py.contract<"$T">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.set">, !py.protocol<"Iterable", [!py.contract<"$T">]>] -> [!py.contract<"builtins.set", [!py.contract<"$T">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.set">, !py.protocol<"Iterable", [!py.contract<"$T">]>] -> [!py.contract<"builtins.set", [!py.contract<"$T">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.set">, !py.protocol<"Iterable", [!py.contract<"$T">]>] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.set">, !py.protocol<"Iterable", [!py.contract<"$T">]>] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.set">, !py.protocol<"Iterable", [!py.contract<"$T">]>] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.set">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.set">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.set">, !py.contract<"builtins.set", [!py.contract<"$T">]>] -> [!py.contract<"builtins.set", [!py.contract<"$T">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.set">, !py.contract<"builtins.set", [!py.contract<"$T">]>] -> [!py.contract<"builtins.set", [!py.contract<"$T">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.set">, !py.contract<"builtins.set", [!py.contract<"$T">]>] -> [!py.contract<"builtins.set", [!py.contract<"$T">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.set">, !py.contract<"builtins.set", [!py.contract<"$T">]>] -> [!py.contract<"builtins.set", [!py.contract<"$T">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.set">, !py.contract<"builtins.set">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.set">, !py.contract<"builtins.set">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.set">, !py.contract<"builtins.set">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.set">, !py.contract<"builtins.set">] -> [!py.contract<"builtins.bool">]>
    ],
    method_kinds = ["instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance"]
  } {}

  py.class @frozenset attributes {
    base_names = ["AbstractSet", "Hashable"],
    ly.typing.params = ["T"],
    ly.typing.param_variance = ["covariant"],
    ly.typing.base_args = [[!py.contract<"$T">], []],
    ly.runtime.contract = "builtins.frozenset", ly.runtime.required,
    ly.runtime.required_deallocator,
    ly.runtime.required_initializers = ["__new__"],
    ly.runtime.required_methods = ["__len__"],
    method_names = ["__new__", "__new__", "__init__", "__init__", "__len__",
                    "__iter__",
                    "__contains__", "__hash__", "__eq__", "__ne__",
                    "union", "intersection", "difference",
                    "symmetric_difference", "issubset", "issuperset",
                    "isdisjoint", "__or__", "__and__", "__sub__", "__xor__",
                    "__le__", "__lt__", "__ge__", "__gt__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.frozenset">>, !py.contract<"builtins.set", [!py.contract<"$T">]>] -> [!py.self]>,
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.frozenset">>, !py.contract<"builtins.list", [!py.contract<"$T">]>] -> [!py.self]>,
      !py.protocol<"Callable", [!py.contract<"builtins.frozenset">, !py.contract<"builtins.set", [!py.contract<"$T">]>] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.frozenset">, !py.contract<"builtins.list", [!py.contract<"$T">]>] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.frozenset">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.frozenset">] -> [!py.protocol<"Iterator", [!py.contract<"$T">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.frozenset">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.frozenset">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.frozenset">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.frozenset">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.frozenset">, !py.contract<"builtins.frozenset", [!py.contract<"$T">]>] -> [!py.contract<"builtins.frozenset", [!py.contract<"$T">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.frozenset">, !py.contract<"builtins.frozenset", [!py.contract<"$T">]>] -> [!py.contract<"builtins.frozenset", [!py.contract<"$T">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.frozenset">, !py.contract<"builtins.frozenset", [!py.contract<"$T">]>] -> [!py.contract<"builtins.frozenset", [!py.contract<"$T">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.frozenset">, !py.contract<"builtins.frozenset", [!py.contract<"$T">]>] -> [!py.contract<"builtins.frozenset", [!py.contract<"$T">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.frozenset">, !py.contract<"builtins.frozenset">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.frozenset">, !py.contract<"builtins.frozenset">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.frozenset">, !py.contract<"builtins.frozenset">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.frozenset">, !py.contract<"builtins.frozenset", [!py.contract<"$T">]>] -> [!py.contract<"builtins.frozenset", [!py.contract<"$T">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.frozenset">, !py.contract<"builtins.frozenset", [!py.contract<"$T">]>] -> [!py.contract<"builtins.frozenset", [!py.contract<"$T">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.frozenset">, !py.contract<"builtins.frozenset", [!py.contract<"$T">]>] -> [!py.contract<"builtins.frozenset", [!py.contract<"$T">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.frozenset">, !py.contract<"builtins.frozenset", [!py.contract<"$T">]>] -> [!py.contract<"builtins.frozenset", [!py.contract<"$T">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.frozenset">, !py.contract<"builtins.frozenset">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.frozenset">, !py.contract<"builtins.frozenset">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.frozenset">, !py.contract<"builtins.frozenset">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.frozenset">, !py.contract<"builtins.frozenset">] -> [!py.contract<"builtins.bool">]>
    ],
    method_kinds = ["classmethod", "classmethod", "instance", "instance",
                    "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance"]
  } {}

  // set.__repr__ / frozenset.__repr__: `{e0, e1, ...}`, `set()` and
  // `frozenset()` for the empty ones, the same uniform element dispatch as
  // LyList_Repr. Without these, `print({1, 2})` reached the lowering as
  // "runtime manifest has no builtins.set.__repr__ method" and a set nested
  // in a printed container aborted the process, while frozenset fell back to
  // the address form `<frozenset object at 0x...>`.
  //
  // Deviation, noted: the elements come out in the table's own order (which
  // is insertion order for a set this compiler builds), not CPython's hash
  // order. The two agree for the small ints a reader is most likely to
  // print, and nothing else can be matched without adopting CPython's table.
  func.func private @__ly_set_repr_body(%len: i64, %items_ptr: !llvm.ptr, %open_h: memref<2xi64> {ly.ownership.object_header}, %open_b: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0], ly.ownership.transfer_args = [2]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c16_i64 = func.call @__ly_box_word_count() : () -> i64
    %len_idx = arith.index_cast %len : i64 to index
    %loop:2 = scf.for %i = %c0 to %len_idx step %c1 iter_args(%rh = %open_h, %rb = %open_b) -> (memref<2xi64>, memref<?xi8>) {
      %i_i64 = arith.index_cast %i : index to i64
      %is_pos = arith.cmpi sgt, %i_i64, %c0_i64 : i64
      %sep:2 = scf.if %is_pos -> (memref<2xi64>, memref<?xi8>) {
        %sep_ref = memref.get_global @__ly_repr_comma : memref<2xi8>
        %sep_dyn = memref.cast %sep_ref : memref<2xi8> to memref<?xi8>
        %sh, %sb = func.call @LyUnicode_FromBytes(%sep_dyn, %c0, %c2_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
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
    %close_ref = memref.get_global @__ly_repr_rbrace : memref<1xi8>
    %close_dyn = memref.cast %close_ref : memref<1xi8> to memref<?xi8>
    %clh, %clb = func.call @LyUnicode_FromBytes(%close_dyn, %c0, %c1_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %out_h, %out_b = func.call @LyUnicode_Concat(%loop#0, %loop#1, %clh, %clb) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.call @LyUnicode_DecRef(%loop#0) : (memref<2xi64>) -> ()
    func.call @LyUnicode_DecRef(%clh) : (memref<2xi64>) -> ()
    func.return %out_h, %out_b : memref<2xi64>, memref<?xi8>
  }

  func.func @LySet_Repr(%self: memref<9xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.set", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    // The printed order is the TABLE's, and an append leaves the dense array
    // out of that order until this runs.
    %raw_order = memref.cast %self : memref<9xi64> to memref<?xi64>
    func.call @__ly_set_raw_reorder(%raw_order) : (memref<?xi64>) -> ()
    %c1_i64 = arith.constant 1 : i64
    %c5_i64 = arith.constant 5 : i64
    %zero = arith.constant 0 : i64
    %length_slot = arith.constant 2 : index
    %len = memref.load %self[%length_slot] : memref<9xi64>
    %empty = arith.cmpi eq, %len, %zero : i64
    %result:2 = scf.if %empty -> (memref<2xi64>, memref<?xi8>) {
      %e_ref = memref.get_global @__ly_repr_set_empty : memref<5xi8>
      %e_dyn = memref.cast %e_ref : memref<5xi8> to memref<?xi8>
      %eh, %eb = func.call @LyUnicode_FromBytes(%e_dyn, %c0, %c5_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %eh, %eb : memref<2xi64>, memref<?xi8>
    } else {
      %open_ref = memref.get_global @__ly_repr_lbrace : memref<1xi8>
      %open_dyn = memref.cast %open_ref : memref<1xi8> to memref<?xi8>
      %oh, %ob = func.call @LyUnicode_FromBytes(%open_dyn, %c0, %c1_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
      %items = func.call @__ly_set_items(%self) : (memref<9xi64>) -> memref<?xi64>
      %items_idx = memref.extract_aligned_pointer_as_index %items : memref<?xi64> -> index
      %items_i64 = arith.index_cast %items_idx : index to i64
      %items_ptr = llvm.inttoptr %items_i64 : i64 to !llvm.ptr
      %body:2 = func.call @__ly_set_repr_body(%len, %items_ptr, %oh, %ob) : (i64, !llvm.ptr, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %body#0, %body#1 : memref<2xi64>, memref<?xi8>
    }
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyFrozenSet_Repr(%self: memref<9xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.frozenset", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    // The printed order is the TABLE's, and an append leaves the dense array
    // out of that order until this runs.
    %raw_order = memref.cast %self : memref<9xi64> to memref<?xi64>
    func.call @__ly_set_raw_reorder(%raw_order) : (memref<?xi64>) -> ()
    %c1_i64 = arith.constant 1 : i64
    %c11_i64 = arith.constant 11 : i64
    %zero = arith.constant 0 : i64
    %length_slot = arith.constant 2 : index
    %len = memref.load %self[%length_slot] : memref<9xi64>
    %empty = arith.cmpi eq, %len, %zero : i64
    %result:2 = scf.if %empty -> (memref<2xi64>, memref<?xi8>) {
      %e_ref = memref.get_global @__ly_repr_frozenset_empty : memref<11xi8>
      %e_dyn = memref.cast %e_ref : memref<11xi8> to memref<?xi8>
      %eh, %eb = func.call @LyUnicode_FromBytes(%e_dyn, %c0, %c11_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
      scf.yield %eh, %eb : memref<2xi64>, memref<?xi8>
    } else {
      %open_ref = memref.get_global @__ly_repr_frozenset_open : memref<11xi8>
      %open_dyn = memref.cast %open_ref : memref<11xi8> to memref<?xi8>
      %oh, %ob = func.call @LyUnicode_FromBytes(%open_dyn, %c0, %c11_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
      %items = func.call @__ly_frozenset_items(%self) : (memref<9xi64>) -> memref<?xi64>
      %items_idx = memref.extract_aligned_pointer_as_index %items : memref<?xi64> -> index
      %items_i64 = arith.index_cast %items_idx : index to i64
      %items_ptr = llvm.inttoptr %items_i64 : i64 to !llvm.ptr
      %body:2 = func.call @__ly_set_repr_body(%len, %items_ptr, %oh, %ob) : (i64, !llvm.ptr, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      %rp_ref = memref.get_global @__ly_repr_rparen : memref<1xi8>
      %rp_dyn = memref.cast %rp_ref : memref<1xi8> to memref<?xi8>
      %rph, %rpb = func.call @LyUnicode_FromBytes(%rp_dyn, %c0, %c1_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
      %out_h, %out_b = func.call @LyUnicode_Concat(%body#0, %body#1, %rph, %rpb) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      func.call @LyUnicode_DecRef(%body#0) : (memref<2xi64>) -> ()
      func.call @LyUnicode_DecRef(%rph) : (memref<2xi64>) -> ()
      scf.yield %out_h, %out_b : memref<2xi64>, memref<?xi8>
    }
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  // ===== builtins.set / builtins.frozenset: one entity, one root =====
  //
  // Two contracts, one physical layout, `memref<9xi64>`:
  //
  //   word 0  refcount            word 5  table base address
  //   word 1  class id (21 / 23)  word 6  table mask (size - 1)
  //   word 2  used (live entries) word 7  fill (live + dummies)
  //   word 3  capacity            word 8  order flag (__ly_set_raw_reorder)
  //   word 4  items base address
  //
  // ITERATION ORDER IS THE TABLE'S, and it is bought without teaching a single
  // walk about the table: the DENSE ARRAY IS KEPT IN TABLE-SLOT ORDER, so
  // `items[0..used)` -- what __repr__, the for-loop lowering, list(s) and every
  // algebra scan below already read -- is CPython's order by construction.
  // Insert therefore places at a rank rather than appending, and the table's
  // state words carry the dense index so a probe still answers in one step.
  //
  // Why NOT walk the table at each of those sites instead: the order consumers
  // are spread across this file, the emitter and the runtime lowering, and the
  // `items_view` primitive is the only thing they share. Ordering the array is
  // one place to be right; teaching every reader the table is many.
  //
  // Why NOT keep the array insertion-ordered and sort at iteration: CPython's
  // order is not sorted, it is the table's -- it coincides with sorted for
  // small ints only because a small int hashes to itself.
  //
  // ⛔ THE COST, measured, because ordering the array is what buys it. An
  // insert is O(n): it shifts the dense tail and renumbers the slots after the
  // one it takes. That is the same class the dense array was already in (the
  // probe used to be a linear scan of every live entry), so nothing regressed
  // asymptotically, and the table pays for it on the other side --
  //
  //   20k adds     2.7 s -> 5.4 s     (the shift, against an O(n) probe)
  //   200k `in`    4.6 s -> 2.6 s     (one table lookup, against that scan)
  //
  // -- but O(1) inserts are reachable and this is where to start: append, mark
  // the handle dirty, and permute into slot order inside the `items_view`
  // primitive, which is the ONE place both this file and the C++ lowering go
  // through to see the array. Not done here because the primitive is declared
  // `ly.runtime.interior_word` and the ownership walk treats it as a pure
  // view, so making it write is a change to that contract rather than to a
  // loop.
  //
  // Words 0-7 are the layout in Passes/Runtime/ABI/ContainerLayout.h, which
  // dict and list already use; a set spends 5 and 6 (a mapping's secondary
  // array and present flags) and 7 on the table instead.
  //
  // Every algorithm below lives once in a helper taking the handle as
  // `memref<?xi64>`, and the per-contract functions are thin wrappers.
  //
  // Why not one copy of each loop per contract instead: that is how two copies
  // of a probe loop drift apart while both keep compiling, which is why
  // 68feee7 parameterised the shared sequence helpers by length rather than
  // duplicating them.
  //
  // Why the items array is an ADDRESS in the handle and not a value beside it:
  // a growth writes the new address THROUGH the handle, so every holder
  // observes it with no further action and a mutation has nothing to rename.
  // That is what lets add / update / intersection_update / difference_update /
  // symmetric_difference_update be void and non-transferring
  // (rfc/memory-safety-proof.md, `Interior`), and those five plus
  // frozenset.__init__ were the six `transfer_args` declarations these two
  // contracts owned.
  func.func private @boxed_int_value(%meta_bits: i64, %digits_bits: i64) -> i64

  func.func private @LySet_Shape() -> memref<9xi64> attributes {ly.runtime.contract = "builtins.set", ly.runtime.shape}

  // ---- the width-agnostic core --------------------------------------------
  // Every helper below takes the handle as `memref<?xi64>` and reads or writes
  // only words 2..7, which both widths share. A caller obtains one with
  // `memref.cast %self : memref<Nxi64> to memref<?xi64>`.

  func.func private @__ly_set_raw_len(%self: memref<?xi64>) -> i64 {
    %length_slot = arith.constant 2 : index
    %len = memref.load %self[%length_slot] : memref<?xi64>
    func.return %len : i64
  }

  // Borrowed view of the items array, derived at the point of use. The view's
  // SSA name is not an identity: identity is the handle, so a view taken after
  // a growth and one taken before it name the same slot of the same entity.
  // Marked ly.runtime.interior_word so the ownership walk pins whatever the
  // handle came from across the view's uses -- a plain private helper would
  // leave the walk nothing to follow from the call.
  func.func private @__ly_set_raw_items(%self: memref<?xi64>) -> memref<?xi64> attributes {ly.runtime.interior_word} {
    %capacity_slot = arith.constant 3 : index
    %items_slot = arith.constant 4 : index
    %handle_words = func.call @__ly_box_word_count() : () -> i64
    %capacity = memref.load %self[%capacity_slot] : memref<?xi64>
    %words = arith.muli %capacity, %handle_words : i64
    %base = memref.load %self[%items_slot] : memref<?xi64>
    %view = func.call @__ly_global_view_i64(%base, %words) : (i64, i64) -> memref<?xi64>
    func.return %view : memref<?xi64>
  }

  // ---- the hash table ------------------------------------------------------
  // CPython Objects/setobject.c, transcribed: PySet_MINSIZE 8, LINEAR_PROBES 9,
  // PERTURB_SHIFT 5, the `fill*5 >= mask*3` growth trigger and the
  // `used > 50000 ? used*2 : used*4` target. Each slot is one word
  // (`__ly_set_slot_pack`): the state and, above it, the hash's low 32 bits
  //
  // -- which is CPython's `setentry` with the key pointer replaced by the dense
  // index, since the box already lives in the items array.
  //
  // Why the state word and not CPython's (key == NULL, hash == 0 / -1) pair:
  // an element whose hash IS 0 is ordinary here (hash(0) == 0, and word 15 of a
  // box uses 0 for "not yet cached"), so the two conditions CPython folds into
  // the hash field have to be spelled apart or a set containing 0 reads as
  // empty at its own slot.
  // A set table slot is ONE word: the entry's state in the low 32 bits
  // (0 unused, 1 dummy, n >= 2 the dense index n - 2) and the low 32 bits of
  // its hash above them, which is what a probe compares before it calls
  // equality. ⛔ Not the full hash: it would be a second word per slot, 16
  // bytes per entry at the table's load. The walk itself uses the full hash
  // the caller has, and a resize asks the entries for theirs again.
  func.func private @__ly_set_probe_hash(%hash: i64) -> i64 {
    %low32 = arith.constant 4294967295 : i64
    %h = arith.andi %hash, %low32 : i64
    func.return %h : i64
  }

  func.func private @__ly_set_slot_pack(%state: i64, %hash: i64) -> i64 {
    %low32 = arith.constant 4294967295 : i64
    %thirty_two = arith.constant 32 : i64
    %h = arith.andi %hash, %low32 : i64
    %high = arith.shli %h, %thirty_two : i64
    %st = arith.andi %state, %low32 : i64
    %word = arith.ori %high, %st : i64
    func.return %word : i64
  }

  func.func private @__ly_set_raw_table(%self: memref<?xi64>) -> memref<?xi64> attributes {ly.runtime.interior_word} {
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %table_slot = arith.constant 5 : index
    %mask_slot = arith.constant 6 : index
    %mask = memref.load %self[%mask_slot] : memref<?xi64>
    %size = arith.addi %mask, %one : i64
    %words = arith.muli %size, %one : i64
    %base = memref.load %self[%table_slot] : memref<?xi64>
    %view = func.call @__ly_global_view_i64(%base, %words) : (i64, i64) -> memref<?xi64>
    func.return %view : memref<?xi64>
  }

  // A zeroed table of %size slots; returns its base address. Plain memref.alloc
  // with no alignment attribute, so free_raw_i64_ptr can release it later (the
  // same convention as the items array).
  func.func private @__ly_set_table_new(%size: i64) -> i64 {
    %zero = arith.constant 0 : i64
    %two = arith.constant 2 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %one_w = arith.constant 1 : i64
    %words = arith.muli %size, %one_w : i64
    %words_index = arith.index_cast %words : i64 to index
    %table = memref.alloc(%words_index) : memref<?xi64>
    scf.for %w = %c0 to %words_index step %c1 {
      memref.store %zero, %table[%w] : memref<?xi64>
    }
    %table_index = memref.extract_aligned_pointer_as_index %table : memref<?xi64> -> index
    %table_word = arith.index_cast %table_index : index to i64
    func.return %table_word : i64
  }

  // set_lookkey: the DENSE index of the entry equal to %elem_box, or -1.
  // The set's spelling: table at handle word 5, mask at 6, items at 4.
  func.func private @__ly_set_table_lookup(%self: memref<?xi64>, %elem_box: !llvm.ptr, %hash: i64) -> i64 {
    %mask_slot = arith.constant 6 : index
    %mask = memref.load %self[%mask_slot] : memref<?xi64>
    %table = func.call @__ly_set_raw_table(%self) : (memref<?xi64>) -> memref<?xi64>
    %items = func.call @__ly_set_raw_items(%self) : (memref<?xi64>) -> memref<?xi64>
    %items_idx = memref.extract_aligned_pointer_as_index %items : memref<?xi64> -> index
    %items_i64 = arith.index_cast %items_idx : index to i64
    %items_ptr = llvm.inttoptr %items_i64 : i64 to !llvm.ptr
    %found = func.call @__ly_table_lookup(%table, %mask, %items_ptr, %elem_box, %hash) : (memref<?xi64>, i64, !llvm.ptr, !llvm.ptr, i64) -> i64
    func.return %found : i64
  }
  // set_lookkey over a table and an items array named directly, so the dict can
  // use it too: the two carry the table in different handle words and the dict
  // derives its mask instead of storing it, and neither difference reaches the
  // probe. The probe sequence is CPython's -- LINEAR_PROBES of 9 followed by
  // i*5 + 1 + perturb -- and that is the part worth having once.
  func.func private @__ly_table_lookup(%table: memref<?xi64>, %mask: i64, %items_ptr: !llvm.ptr, %elem_box: !llvm.ptr, %hash: i64) -> i64 {
    %low32 = arith.constant 4294967295 : i64
    %thirty_two = arith.constant 32 : i64
    %hash32 = arith.andi %hash, %low32 : i64
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
          %state_index = arith.index_cast %s : i64 to index
          %state_word = memref.load %table[%state_index] : memref<?xi64>
          %state = arith.andi %state_word, %low32 : i64
          %unused = arith.cmpi eq, %state, %zero : i64
          %seen:2 = scf.if %unused -> (i64, i1) {
            scf.yield %minus_one, %true : i64, i1
          } else {
            %live = arith.cmpi sge, %state, %two : i64
            %hit:2 = scf.if %live -> (i64, i1) {
              %entry_hash = arith.shrui %state_word, %thirty_two : i64
              %same_hash = arith.cmpi eq, %entry_hash, %hash32 : i64
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

  // set_add_entry's probe half: (dense index or -1, target slot, target was
  // unused rather than a dummy). The caller writes the entry, because only it
  // knows where the box comes from.
  //
  // ⭐ The freeslot is the LAST dummy in the run, not the first: CPython
  // assigns `freeslot = entry` without a null check, and taking the first
  // instead disagrees with CPython on 20 of 1632 measured insert/discard
  // sequences -- always a pair of neighbours in the printed order.
  func.func private @__ly_set_table_add_probe(%self: memref<?xi64>, %elem_box: !llvm.ptr, %hash: i64) -> (i64, i64, i1) {
    %low32 = arith.constant 4294967295 : i64
    %thirty_two = arith.constant 32 : i64
    %hash32 = arith.andi %hash, %low32 : i64
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
    %mask_slot = arith.constant 6 : index
    %mask = memref.load %self[%mask_slot] : memref<?xi64>
    %table = func.call @__ly_set_raw_table(%self) : (memref<?xi64>) -> memref<?xi64>
    %items = func.call @__ly_set_raw_items(%self) : (memref<?xi64>) -> memref<?xi64>
    %items_idx = memref.extract_aligned_pointer_as_index %items : memref<?xi64> -> index
    %items_i64 = arith.index_cast %items_idx : index to i64
    %items_ptr = llvm.inttoptr %items_i64 : i64 to !llvm.ptr
    %i0 = arith.andi %hash, %mask : i64
    // (i, perturb, found, slot, freeslot, done)
    %walk:6 = scf.while (%i = %i0, %p = %hash, %found = %minus_one,
                         %slot = %minus_one, %free = %minus_one, %done = %false)
        : (i64, i64, i64, i64, i64, i1) -> (i64, i64, i64, i64, i64, i1) {
      %again = arith.xori %done, %true : i1
      scf.condition(%again) %i, %p, %found, %slot, %free, %done : i64, i64, i64, i64, i64, i1
    } do {
    ^body(%i: i64, %p: i64, %found: i64, %slot: i64, %free: i64, %done: i1):
      %limit = arith.addi %i, %nine : i64
      %linear = arith.cmpi ule, %limit, %mask : i64
      %probes = arith.select %linear, %nine, %zero : i1, i64
      %count = arith.addi %probes, %one : i64
      %count_index = arith.index_cast %count : i64 to index
      %run:4 = scf.for %k = %c0 to %count_index step %c1
          iter_args(%f = %found, %sl = %slot, %fr = %free, %d = %done)
          -> (i64, i64, i64, i1) {
        %step:4 = scf.if %d -> (i64, i64, i64, i1) {
          scf.yield %f, %sl, %fr, %d : i64, i64, i64, i1
        } else {
          %kk = arith.index_cast %k : index to i64
          %s = arith.addi %i, %kk : i64
          %state_index = arith.index_cast %s : i64 to index
          %state_word = memref.load %table[%state_index] : memref<?xi64>
          %state = arith.andi %state_word, %low32 : i64
          %unused = arith.cmpi eq, %state, %zero : i64
          %seen:4 = scf.if %unused -> (i64, i64, i64, i1) {
            // found_unused_or_dummy: a dummy already seen wins the slot.
            %reuse = arith.cmpi sge, %fr, %zero : i64
            %target = arith.select %reuse, %fr, %s : i1, i64
            scf.yield %minus_one, %target, %fr, %true : i64, i64, i64, i1
          } else {
            %live = arith.cmpi sge, %state, %two : i64
            %hit:4 = scf.if %live -> (i64, i64, i64, i1) {
              %entry_hash = arith.shrui %state_word, %thirty_two : i64
              %same_hash = arith.cmpi eq, %entry_hash, %hash32 : i64
              %cmp:4 = scf.if %same_hash -> (i64, i64, i64, i1) {
                %dense = arith.subi %state, %two : i64
                %off = arith.muli %dense, %c16 : i64
                %entry = llvm.getelementptr %items_ptr[%off] : (!llvm.ptr, i64) -> !llvm.ptr, i64
                %eq = func.call @__ly_box_equal(%entry, %elem_box) : (!llvm.ptr, !llvm.ptr) -> i1
                %hit_dense = arith.select %eq, %dense, %f : i1, i64
                %hit_slot = arith.select %eq, %s, %sl : i1, i64
                scf.yield %hit_dense, %hit_slot, %fr, %eq : i64, i64, i64, i1
              } else {
                scf.yield %f, %sl, %fr, %false : i64, i64, i64, i1
              }
              scf.yield %cmp#0, %cmp#1, %cmp#2, %cmp#3 : i64, i64, i64, i1
            } else {
              // A dummy. CPython overwrites freeslot, so the last one wins.
              scf.yield %f, %sl, %s, %false : i64, i64, i64, i1
            }
            scf.yield %hit#0, %hit#1, %hit#2, %hit#3 : i64, i64, i64, i1
          }
          scf.yield %seen#0, %seen#1, %seen#2, %seen#3 : i64, i64, i64, i1
        }
        scf.yield %step#0, %step#1, %step#2, %step#3 : i64, i64, i64, i1
      }
      %np = arith.shrui %p, %shift : i64
      %i5 = arith.muli %i, %probe_scale : i64
      %i51 = arith.addi %i5, %one : i64
      %i5p = arith.addi %i51, %np : i64
      %ni = arith.andi %i5p, %mask : i64
      scf.yield %ni, %np, %run#0, %run#1, %run#2, %run#3 : i64, i64, i64, i64, i64, i1
    }
    // The slot came from a dummy exactly when the run recorded one AND it is
    // the slot chosen; `fill` only moves when a never-used slot is consumed.
    %was_free = arith.cmpi eq, %walk#4, %walk#3 : i64
    %from_unused = arith.xori %was_free, %true : i1
    func.return %walk#2, %walk#3, %from_unused : i64, i64, i1
  }

  // set_insert_clean: the slot a hash lands in when no key can already be
  // present (a resize, or a merge into an empty receiver).
  func.func private @__ly_set_table_clean_slot(%table: memref<?xi64>, %mask: i64, %hash: i64) -> i64 {
    %low32 = arith.constant 4294967295 : i64
    %thirty_two = arith.constant 32 : i64
    %hash32 = arith.andi %hash, %low32 : i64
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
          %state_index = arith.index_cast %s : i64 to index
          %state_word = memref.load %table[%state_index] : memref<?xi64>
          %state = arith.andi %state_word, %low32 : i64
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

  // Rewrite the dense array so that it is the table's slot order again, given
  // that every occupied slot's state currently names an index into %src_items.
  // The receiver's own items array is REPLACED, so the source may be the array
  // being replaced (a resize) or another set's (a merge into an empty
  // receiver); %retain says which -- entries that move keep their reference,
  // entries copied from another set gain one.
  func.func private @__ly_set_table_rebuild_dense(%self: memref<?xi64>, %src_items: memref<?xi64>, %retain: i1) {
    %low32 = arith.constant 4294967295 : i64
    %thirty_two = arith.constant 32 : i64
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16_words = func.call @__ly_box_word_count() : () -> i64
    %c16 = arith.index_cast %c16_words : i64 to index
    %c16_i64 = func.call @__ly_box_word_count() : () -> i64
    %entity_slot = arith.constant 0 : index
    %length_slot = arith.constant 2 : index
    %capacity_slot = arith.constant 3 : index
    %items_slot = arith.constant 4 : index
    %mask_slot = arith.constant 6 : index
    %mask = memref.load %self[%mask_slot] : memref<?xi64>
    %capacity = memref.load %self[%capacity_slot] : memref<?xi64>
    %table = func.call @__ly_set_raw_table(%self) : (memref<?xi64>) -> memref<?xi64>
    %payload_words = arith.muli %capacity, %c16_i64 : i64
    %payload_index = arith.index_cast %payload_words : i64 to index
    %fresh = memref.alloc(%payload_index) : memref<?xi64>
    %slots = arith.addi %mask, %one : i64
    %slots_index = arith.index_cast %slots : i64 to index
    %placed = scf.for %s = %c0 to %slots_index step %c1 iter_args(%k = %c0) -> (index) {
      %ss = arith.index_cast %s : index to i64
      %state_index = arith.index_cast %ss : i64 to index
      %state_word = memref.load %table[%state_index] : memref<?xi64>
      %state = arith.andi %state_word, %low32 : i64
      %live = arith.cmpi sge, %state, %two : i64
      %next = scf.if %live -> (index) {
        %src_dense = arith.subi %state, %two : i64
        %src_base_i64 = arith.muli %src_dense, %c16_i64 : i64
        %src_base = arith.index_cast %src_base_i64 : i64 to index
        %dst_base = arith.muli %k, %c16 : index
        scf.for %w = %c0 to %c16 step %c1 {
          %src = arith.addi %src_base, %w : index
          %dst = arith.addi %dst_base, %w : index
          %word = memref.load %src_items[%src] : memref<?xi64>
          memref.store %word, %fresh[%dst] : memref<?xi64>
        }
        scf.if %retain {
          %entity_index = arith.addi %dst_base, %entity_slot : index
          %entity = memref.load %fresh[%entity_index] : memref<?xi64>
          func.call @__ly_handle_retain_raw(%entity) : (i64) -> ()
        }
        %kk = arith.index_cast %k : index to i64
        %new_state = arith.addi %kk, %two : i64
        %kept_hash = arith.shrui %state_word, %thirty_two : i64
        %new_word = func.call @__ly_set_slot_pack(%new_state, %kept_hash) : (i64, i64) -> i64
        memref.store %new_word, %table[%state_index] : memref<?xi64>
        %inc = arith.addi %k, %c1 : index
        scf.yield %inc : index
      } else {
        scf.yield %k : index
      }
      scf.yield %next : index
    }
    // Zero the tail so a stale handle in an unused slot never looks live.
    %placed_words = arith.muli %placed, %c16 : index
    scf.for %w = %placed_words to %payload_index step %c1 {
      memref.store %zero, %fresh[%w] : memref<?xi64>
    }
    %old_items = memref.load %self[%items_slot] : memref<?xi64>
    %fresh_index = memref.extract_aligned_pointer_as_index %fresh : memref<?xi64> -> index
    %fresh_word = arith.index_cast %fresh_index : index to i64
    memref.store %fresh_word, %self[%items_slot] : memref<?xi64>
    %placed_i64 = arith.index_cast %placed : index to i64
    memref.store %placed_i64, %self[%length_slot] : memref<?xi64>
    func.call @free_raw_i64_ptr(%old_items) : (i64) -> ()
    func.return
  }

  // set_table_resize. The old table is walked in SLOT order, which is what
  // makes the outcome a function of the table rather than of the insertion
  // history -- and then the dense array is permuted to the new slot order.
  func.func private @__ly_set_raw_resize(%self: memref<?xi64>, %minused: i64) {
    %low32 = arith.constant 4294967295 : i64
    %thirty_two = arith.constant 32 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %eight = arith.constant 8 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %false = arith.constant false
    %length_slot = arith.constant 2 : index
    %table_slot = arith.constant 5 : index
    %mask_slot = arith.constant 6 : index
    %fill_slot = arith.constant 7 : index
    %newsize = scf.while (%size = %eight) : (i64) -> i64 {
      %small = arith.cmpi sle, %size, %minused : i64
      scf.condition(%small) %size : i64
    } do {
    ^body(%size: i64):
      %doubled = arith.muli %size, %two : i64
      scf.yield %doubled : i64
    }
    %newmask = arith.subi %newsize, %one : i64
    %newbase = func.call @__ly_set_table_new(%newsize) : (i64) -> i64
    %newwords = arith.muli %newsize, %one : i64
    %newtable = func.call @__ly_global_view_i64(%newbase, %newwords) : (i64, i64) -> memref<?xi64>
    %oldmask = memref.load %self[%mask_slot] : memref<?xi64>
    %oldbase = memref.load %self[%table_slot] : memref<?xi64>
    %oldtable = func.call @__ly_set_raw_table(%self) : (memref<?xi64>) -> memref<?xi64>
    // ⛔ The probe walk needs the FULL hash (CPython's perturbation reaches
    // its high bits, and that is what makes the slot -- and so the iteration
    // order -- CPython's), and a slot keeps only 32 bits of it; so a resize
    // asks each entry again, as CPython would have to without `setentry.hash`.
    %entry_words = func.call @__ly_box_word_count() : () -> i64
    %old_items = func.call @__ly_set_raw_items(%self) : (memref<?xi64>) -> memref<?xi64>
    %old_items_idx = memref.extract_aligned_pointer_as_index %old_items : memref<?xi64> -> index
    %old_items_i64 = arith.index_cast %old_items_idx : index to i64
    %old_items_ptr = llvm.inttoptr %old_items_i64 : i64 to !llvm.ptr
    %oldslots = arith.addi %oldmask, %one : i64
    %oldslots_index = arith.index_cast %oldslots : i64 to index
    scf.for %s = %c0 to %oldslots_index step %c1 {
      %ss = arith.index_cast %s : index to i64
      %state_index = arith.index_cast %ss : i64 to index
      %state_word = memref.load %oldtable[%state_index] : memref<?xi64>
      %state = arith.andi %state_word, %low32 : i64
      %live = arith.cmpi sge, %state, %two : i64
      scf.if %live {
        %dense = arith.subi %state, %two : i64
        %entry_off = arith.muli %dense, %entry_words : i64
        %entry = llvm.getelementptr %old_items_ptr[%entry_off] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        %hash = func.call @__ly_set_entry_hash(%entry) : (!llvm.ptr) -> i64
        %slot = func.call @__ly_set_table_clean_slot(%newtable, %newmask, %hash) : (memref<?xi64>, i64, i64) -> i64
        %dst_index = arith.index_cast %slot : i64 to index
        %placed = func.call @__ly_set_slot_pack(%state, %hash) : (i64, i64) -> i64
        memref.store %placed, %newtable[%dst_index] : memref<?xi64>
      }
    }
    memref.store %newbase, %self[%table_slot] : memref<?xi64>
    memref.store %newmask, %self[%mask_slot] : memref<?xi64>
    func.call @free_raw_i64_ptr(%oldbase) : (i64) -> ()
    %items = func.call @__ly_set_raw_items(%self) : (memref<?xi64>) -> memref<?xi64>
    func.call @__ly_set_table_rebuild_dense(%self, %items, %false) : (memref<?xi64>, memref<?xi64>, i1) -> ()
    %used = memref.load %self[%length_slot] : memref<?xi64>
    memref.store %used, %self[%fill_slot] : memref<?xi64>
    func.return
  }

  // Write a new entry at %slot, whose 16 box words come from
  // %src_items[%src_slot]. Returns the entity word of the placed box so the
  // caller can retain it (LySet_AddBox's caller already did; a copy out of
  // another set has not).
  //
  // The dense array is in table order, so the entry belongs at the RANK of its
  // slot -- one walk of the tail both finds that rank and shifts the dense
  // indices the insert displaces.
  func.func private @__ly_set_raw_place(%self: memref<?xi64>, %slot: i64, %from_unused: i1, %src_items: memref<?xi64>, %src_slot: i64, %hash: i64) -> i64 {
    %minus_one = arith.constant -1 : i64
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %fill_limit_scale = arith.constant 3 : i64
    %probe_scale = arith.constant 5 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16_words = func.call @__ly_box_word_count() : () -> i64
    %c16 = arith.index_cast %c16_words : i64 to index
    %c16_i64 = func.call @__ly_box_word_count() : () -> i64
    %big = arith.constant 50000 : i64
    %four = arith.constant 4 : i64
    %entity_slot = arith.constant 0 : index
    %length_slot = arith.constant 2 : index
    %mask_slot = arith.constant 6 : index
    %fill_slot = arith.constant 7 : index
    %used = memref.load %self[%length_slot] : memref<?xi64>
    %required = arith.addi %used, %one : i64
    func.call @__ly_set_raw_ensure_capacity(%self, %required) : (memref<?xi64>, i64) -> ()
    %items = func.call @__ly_set_raw_items(%self) : (memref<?xi64>) -> memref<?xi64>
    %table = func.call @__ly_set_raw_table(%self) : (memref<?xi64>) -> memref<?xi64>
    %mask = memref.load %self[%mask_slot] : memref<?xi64>
    // ⭐ APPEND, and let the ORDER be recovered later.
    //
    // What stood here kept the dense array in TABLE-SLOT order, so that
    // iteration and repr could walk it straight and come out in the order
    // CPython's set_next hands entries back. Holding a sorted array under
    // insertion costs a scan of the table to find the rank (and to bump every
    // live index after it) and a memmove of the dense tail: O(n) per add, so
    // building a set was quadratic -- 32,000 elements took 673 ms against
    // CPython 3.14's 17.6 ms.
    //
    // The entry now goes at the end and word 8 records that the dense order no
    // longer matches the table's. `__ly_set_raw_reorder` restores it, once, in
    // front of the two things that can observe it: repr, and the creation of an
    // iterator. That is the same walk CPython does per iteration anyway, moved
    // off the insert.
    %order_slot = arith.constant 8 : index
    %rank = arith.addi %used, %zero : i64
    %rank_index = arith.index_cast %rank : i64 to index
    memref.store %one, %self[%order_slot] : memref<?xi64>
    %src_base_i64 = arith.muli %src_slot, %c16_i64 : i64
    %src_base = arith.index_cast %src_base_i64 : i64 to index
    %dst_base = arith.muli %rank_index, %c16 : index
    scf.for %w = %c0 to %c16 step %c1 {
      %src = arith.addi %src_base, %w : index
      %dst = arith.addi %dst_base, %w : index
      %word = memref.load %src_items[%src] : memref<?xi64>
      memref.store %word, %items[%dst] : memref<?xi64>
    }
    %entity_index = arith.addi %dst_base, %entity_slot : index
    %entity = memref.load %items[%entity_index] : memref<?xi64>
    %state = arith.addi %rank, %two : i64
    %slot_index = arith.index_cast %slot : i64 to index
    %slot_word = func.call @__ly_set_slot_pack(%state, %hash) : (i64, i64) -> i64
    memref.store %slot_word, %table[%slot_index] : memref<?xi64>
    memref.store %required, %self[%length_slot] : memref<?xi64>
    scf.if %from_unused {
      %fill = memref.load %self[%fill_slot] : memref<?xi64>
      %new_fill = arith.addi %fill, %one : i64
      memref.store %new_fill, %self[%fill_slot] : memref<?xi64>
      %loaded = arith.muli %new_fill, %probe_scale : i64
      %room = arith.muli %mask, %fill_limit_scale : i64
      %crowded = arith.cmpi sge, %loaded, %room : i64
      scf.if %crowded {
        %huge = arith.cmpi sgt, %required, %big : i64
        %factor = arith.select %huge, %two, %four : i1, i64
        %target = arith.muli %required, %factor : i64
        func.call @__ly_set_raw_resize(%self, %target) : (memref<?xi64>, i64) -> ()
      }
    }
    func.return %entity : i64
  }

  // Rewrite the dense array in TABLE-SLOT order and renumber the states, once,
  // when the order flag says an append left it out of order. This is the walk
  // CPython's set_next does per iteration; doing it here rather than per insert
  // is what makes `set.add` O(1).
  //
  // Why a scratch array rather than an in-place permutation: the permutation is
  // a set of cycles over 16-word blocks, and following them in place needs a
  // block of scratch anyway plus a visited bitmap. One pass through a scratch
  // copy is the same asymptotics and no bookkeeping.
  func.func private @__ly_set_raw_reorder(%self: memref<?xi64>) {
    %low32 = arith.constant 4294967295 : i64
    %thirty_two = arith.constant 32 : i64
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16_i64 = func.call @__ly_box_word_count() : () -> i64
    %capacity_slot = arith.constant 3 : index
    %mask_slot = arith.constant 6 : index
    %order_slot = arith.constant 8 : index
    %flag = memref.load %self[%order_slot] : memref<?xi64>
    %stale = arith.cmpi ne, %flag, %zero : i64
    scf.if %stale {
      %items = func.call @__ly_set_raw_items(%self) : (memref<?xi64>) -> memref<?xi64>
      %table = func.call @__ly_set_raw_table(%self) : (memref<?xi64>) -> memref<?xi64>
      %mask = memref.load %self[%mask_slot] : memref<?xi64>
      %capacity = memref.load %self[%capacity_slot] : memref<?xi64>
      %words = arith.muli %capacity, %c16_i64 : i64
      %words_index = arith.index_cast %words : i64 to index
      %scratch = memref.alloc(%words_index) : memref<?xi64>
      %slots = arith.addi %mask, %one : i64
      %slots_index = arith.index_cast %slots : i64 to index
      %placed = scf.for %s = %c0 to %slots_index step %c1 iter_args(%n = %zero) -> (i64) {
        %ss = arith.index_cast %s : index to i64
        %state_index = arith.index_cast %ss : i64 to index
        %state_word = memref.load %table[%state_index] : memref<?xi64>
        %state = arith.andi %state_word, %low32 : i64
        %live = arith.cmpi sge, %state, %two : i64
        %next = scf.if %live -> (i64) {
          %dense = arith.subi %state, %two : i64
          func.call @__ly_box_move_slot(%scratch, %n, %items, %dense) : (memref<?xi64>, i64, memref<?xi64>, i64) -> ()
          %new_state = arith.addi %n, %two : i64
          %kept_hash = arith.shrui %state_word, %thirty_two : i64
          %new_word = func.call @__ly_set_slot_pack(%new_state, %kept_hash) : (i64, i64) -> i64
          memref.store %new_word, %table[%state_index] : memref<?xi64>
          %bumped = arith.addi %n, %one : i64
          scf.yield %bumped : i64
        } else {
          scf.yield %n : i64
        }
        scf.yield %next : i64
      }
      %used_words_i64 = arith.muli %placed, %c16_i64 : i64
      %used_words = arith.index_cast %used_words_i64 : i64 to index
      scf.for %w = %c0 to %used_words step %c1 {
        %word = memref.load %scratch[%w] : memref<?xi64>
        memref.store %word, %items[%w] : memref<?xi64>
      }
      %scratch_index = memref.extract_aligned_pointer_as_index %scratch : memref<?xi64> -> index
      %scratch_word = arith.index_cast %scratch_index : index to i64
      func.call @free_raw_i64_ptr(%scratch_word) : (i64) -> ()
      memref.store %zero, %self[%order_slot] : memref<?xi64>
    }
    func.return
  }

  // The reorder as a contract primitive, so the lowering can ask for it where
  // it builds a set iterator (it walks the dense array itself from there).
  func.func @LySet_Reorder(%self: memref<9xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "builtins.set", ly.runtime.primitive = "reorder"} {
    %raw = memref.cast %self : memref<9xi64> to memref<?xi64>
    func.call @__ly_set_raw_reorder(%raw) : (memref<?xi64>) -> ()
    func.return
  }

  func.func @LyFrozenSet_Reorder(%self: memref<9xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "builtins.frozenset", ly.runtime.primitive = "reorder"} {
    %raw = memref.cast %self : memref<9xi64> to memref<?xi64>
    func.call @__ly_set_raw_reorder(%raw) : (memref<?xi64>) -> ()
    func.return
  }

  // set_discard_entry's write half: the slot becomes a dummy (so the probe
  // sequences that ran through it still reach what is behind it), and the
  // dense array closes the gap.
  func.func private @__ly_set_raw_discard_dense(%self: memref<?xi64>, %dense: i64) {
    %low32 = arith.constant 4294967295 : i64
    %thirty_two = arith.constant 32 : i64
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %minus_one = arith.constant -1 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16_words = func.call @__ly_box_word_count() : () -> i64
    %c16 = arith.index_cast %c16_words : i64 to index
    %length_slot = arith.constant 2 : index
    %mask_slot = arith.constant 6 : index
    %mask = memref.load %self[%mask_slot] : memref<?xi64>
    %table = func.call @__ly_set_raw_table(%self) : (memref<?xi64>) -> memref<?xi64>
    %target_state = arith.addi %dense, %two : i64
    %slots = arith.addi %mask, %one : i64
    %slots_index = arith.index_cast %slots : i64 to index
    scf.for %s = %c0 to %slots_index step %c1 {
      %ss = arith.index_cast %s : index to i64
      %state_index = arith.index_cast %ss : i64 to index
      %state_word = memref.load %table[%state_index] : memref<?xi64>
      %state = arith.andi %state_word, %low32 : i64
      %is_target = arith.cmpi eq, %state, %target_state : i64
      scf.if %is_target {
        memref.store %one, %table[%state_index] : memref<?xi64>
      }
      %after = arith.cmpi sgt, %state, %target_state : i64
      scf.if %after {
        %lowered = arith.subi %state_word, %one : i64
        memref.store %lowered, %table[%state_index] : memref<?xi64>
      }
    }
    %len = memref.load %self[%length_slot] : memref<?xi64>
    %items = func.call @__ly_set_raw_items(%self) : (memref<?xi64>) -> memref<?xi64>
    func.call @LyObject_ReleaseBoxedPayloadArraySlotRaw(%items, %dense) : (memref<?xi64>, i64) -> ()
    %slot_index = arith.index_cast %dense : i64 to index
    %from = arith.addi %slot_index, %c1 : index
    %len_index = arith.index_cast %len : i64 to index
    scf.for %j = %from to %len_index step %c1 {
      %dst_entry = arith.subi %j, %c1 : index
      %src_base = arith.muli %j, %c16 : index
      %dst_base = arith.muli %dst_entry, %c16 : index
      scf.for %w = %c0 to %c16 step %c1 {
        %src = arith.addi %src_base, %w : index
        %dst = arith.addi %dst_base, %w : index
        %word = memref.load %items[%src] : memref<?xi64>
        memref.store %word, %items[%dst] : memref<?xi64>
      }
    }
    %new_len = arith.subi %len, %one : i64
    %last = arith.index_cast %new_len : i64 to index
    %last_base = arith.muli %last, %c16 : index
    scf.for %w = %c0 to %c16 step %c1 {
      %dst = arith.addi %last_base, %w : index
      memref.store %zero, %items[%dst] : memref<?xi64>
    }
    memref.store %new_len, %self[%length_slot] : memref<?xi64>
    func.return
  }

  // Allocate the items array and publish words 0..7. The caller has already
  // zeroed its own dead words, which is the only part that knows the width.
  func.func private @__ly_set_raw_init(%self: memref<?xi64>, %class_id: i64, %length: i64) {
    %one = arith.constant 1 : i64
    %zero = arith.constant 0 : i64
    // PySet_MINSIZE, matching the table's own floor below. It was 64 while the
    // table's was 8, which put 8 KB of 16-word element boxes behind a set that
    // had reserved eight slots.
    %minimum_capacity = arith.constant 8 : i64
    %handle_words = func.call @__ly_box_word_count() : () -> i64
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %length_slot = arith.constant 2 : index
    %capacity_slot = arith.constant 3 : index
    %items_slot = arith.constant 4 : index
    %table_slot = arith.constant 5 : index
    %mask_slot = arith.constant 6 : index
    %fill_slot = arith.constant 7 : index
    %min_table = arith.constant 8 : i64

    %needs_min_capacity = arith.cmpi slt, %length, %minimum_capacity : i64
    %capacity = arith.select %needs_min_capacity, %minimum_capacity, %length : i1, i64
    %payload_words = arith.muli %capacity, %handle_words : i64
    %payload_words_index = arith.index_cast %payload_words : i64 to index
    // Plain memref.alloc with no alignment attribute is a bare malloc, so the
    // aligned pointer IS the allocated pointer and free_raw_i64_ptr can
    // release it later (same convention as __ly_list_alloc).
    %items = memref.alloc(%payload_words_index) : memref<?xi64>
    %items_index = memref.extract_aligned_pointer_as_index %items : memref<?xi64> -> index
    %items_word = arith.index_cast %items_index : index to i64

    memref.store %one, %self[%refcount_slot] : memref<?xi64>
    memref.store %class_id, %self[%layout_slot] : memref<?xi64>
    memref.store %length, %self[%length_slot] : memref<?xi64>
    memref.store %capacity, %self[%capacity_slot] : memref<?xi64>
    memref.store %items_word, %self[%items_slot] : memref<?xi64>
    // PySet_MINSIZE, and it is the table's size rather than the items array's:
    // the two grow on different triggers (the table on load factor, the items
    // array on count) and only the table's schedule is observable.
    %table_word = func.call @__ly_set_table_new(%min_table) : (i64) -> i64
    %min_mask = arith.subi %min_table, %one : i64
    memref.store %table_word, %self[%table_slot] : memref<?xi64>
    memref.store %min_mask, %self[%mask_slot] : memref<?xi64>
    memref.store %zero, %self[%fill_slot] : memref<?xi64>
    func.return
  }

  // Grow the items array in place. Void and non-transferring for the same
  // reason as @LyList_EnsureCapacity: the new base is written into the handle,
  // which every holder already names, so there is nothing to hand back and no
  // reference for a caller to re-acquire.
  func.func private @__ly_set_raw_ensure_capacity(%self: memref<?xi64>, %required: i64) {
    // PySet_MINSIZE, the floor `__ly_set_raw_init` starts from.
    %minimum_capacity = arith.constant 8 : i64
    %handle_words = func.call @__ly_box_word_count() : () -> i64
    %two = arith.constant 2 : i64
    %capacity_slot = arith.constant 3 : index
    %items_slot = arith.constant 4 : index
    %lower = arith.constant 0 : index
    %step = arith.constant 1 : index

    %capacity = memref.load %self[%capacity_slot] : memref<?xi64>
    %needs_grow = arith.cmpi slt, %capacity, %required : i64
    scf.if %needs_grow {
      %items = func.call @__ly_set_raw_items(%self) : (memref<?xi64>) -> memref<?xi64>
      %old_items_word = memref.load %self[%items_slot] : memref<?xi64>
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
      %new_items_index = memref.extract_aligned_pointer_as_index %new_items : memref<?xi64> -> index
      %new_items_word = arith.index_cast %new_items_index : index to i64
      // Publish capacity and the new base together, then free the old block.
      memref.store %new_capacity, %self[%capacity_slot] : memref<?xi64>
      memref.store %new_items_word, %self[%items_slot] : memref<?xi64>
      func.call @free_raw_i64_ptr(%old_items_word) : (i64) -> ()
    }
    func.return
  }

  // Hash-first probe among slots 0..len-1 (dense insertion order; each
  // entry's hash is computed here -- a box carries none).
  // Returns the slot index or -1. Length BY VALUE and items by view: every
  // caller either holds a set that is not growing under it, or wants to probe
  // a bounded prefix (symmetric_difference_update does), and a length taken
  // from the handle here would deny it that.
  func.func private @__ly_set_probe(%len: i64, %items: memref<?xi64>, %elem_box: !llvm.ptr, %elem_hash: i64) -> i64 {
    %minus_one = arith.constant -1 : i64
    %zero = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = func.call @__ly_box_word_count() : () -> i64
    %len_index = arith.index_cast %len : i64 to index
    %items_idx = memref.extract_aligned_pointer_as_index %items : memref<?xi64> -> index
    %items_i64 = arith.index_cast %items_idx : index to i64
    %items_ptr = llvm.inttoptr %items_i64 : i64 to !llvm.ptr
    %found = scf.for %i = %c0 to %len_index step %c1 iter_args(%acc = %minus_one) -> (i64) {
      %not_yet = arith.cmpi eq, %acc, %minus_one : i64
      %next = scf.if %not_yet -> (i64) {
        %ii = arith.index_cast %i : index to i64
        %base = arith.muli %ii, %c16 : i64
        %entry = llvm.getelementptr %items_ptr[%base] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        %entry_hash = func.call @__ly_box_hash(%entry) : (!llvm.ptr) -> i64
        %hash_matches = arith.cmpi eq, %entry_hash, %elem_hash : i64
        %matched = scf.if %hash_matches -> (i64) {
          %eq = func.call @__ly_box_equal(%entry, %elem_box) : (!llvm.ptr, !llvm.ptr) -> i1
          %slot_or = arith.select %eq, %ii, %minus_one : i1, i64
          scf.yield %slot_or : i64
        } else {
          scf.yield %minus_one : i64
        }
        scf.yield %matched : i64
      } else {
        scf.yield %acc : i64
      }
      scf.yield %next : i64
    }
    func.return %found : i64
  }

  // Probe a handle for a raw box: the dense index, or -1. One table lookup,
  // where this used to be a linear scan of every live entry.
  func.func private @__ly_set_raw_probe(%self: memref<?xi64>, %elem_box: !llvm.ptr, %elem_hash: i64) -> i64 {
    %found = func.call @__ly_set_table_lookup(%self, %elem_box, %elem_hash) : (memref<?xi64>, !llvm.ptr, i64) -> i64
    func.return %found : i64
  }

  // Remove the entry at dense index `found`.
  func.func private @__ly_set_raw_remove_slot(%self: memref<?xi64>, %found: i64) {
    func.call @__ly_set_raw_discard_dense(%self, %found) : (memref<?xi64>, i64) -> ()
    func.return
  }

  // The hash of a raw box. ⛔ Recomputed, not cached in the box: a box is a
  // value's handle and carries no hash (BoxLayout.h); the set's table keeps
  // its own copy for the probes that matter, and the set algebra that asks
  // here (union, difference, comparisons) hashes each element once per call.
  func.func private @__ly_set_entry_hash(%entry: !llvm.ptr) -> i64 {
    %hash = func.call @__ly_box_hash(%entry) : (!llvm.ptr) -> i64
    func.return %hash : i64
  }

  // Insert the box at src_slot of a raw array if the receiver does not already
  // hold it, retaining it on the way in. Void: the growth publishes the new
  // items base through the handle, so there is nothing to hand back.
  func.func private @__ly_set_raw_insert_slot(%self: memref<?xi64>, %src_items: memref<?xi64>, %src_slot: index) {
    %minus_one = arith.constant -1 : i64
    %c16_i64 = func.call @__ly_box_word_count() : () -> i64
    %src_idx = memref.extract_aligned_pointer_as_index %src_items : memref<?xi64> -> index
    %src_i64 = arith.index_cast %src_idx : index to i64
    %src_ptr = llvm.inttoptr %src_i64 : i64 to !llvm.ptr
    %slot_i64 = arith.index_cast %src_slot : index to i64
    %off = arith.muli %slot_i64, %c16_i64 : i64
    %entry = llvm.getelementptr %src_ptr[%off] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %hash = func.call @__ly_set_entry_hash(%entry) : (!llvm.ptr) -> i64
    %probe:3 = func.call @__ly_set_table_add_probe(%self, %entry, %hash) : (memref<?xi64>, !llvm.ptr, i64) -> (i64, i64, i1)
    %missing = arith.cmpi eq, %probe#0, %minus_one : i64
    scf.if %missing {
      %entity = func.call @__ly_set_raw_place(%self, %probe#1, %probe#2, %src_items, %slot_i64, %hash) : (memref<?xi64>, i64, i1, memref<?xi64>, i64, i64) -> i64
      func.call @__ly_handle_retain_raw(%entity) : (i64) -> ()
    }
    func.return
  }

  // Insert every slot of a raw (olen, oi) array the receiver does not hold.
  // This is `set_update_internal` for a NON-set iterable: CPython adds those in
  // the source's own order, one set_add_entry each. A set source goes through
  // @__ly_set_raw_merge_set instead, which has three orderings this does not.
  func.func private @__ly_set_raw_merge(%self: memref<?xi64>, %olen: i64, %oi: memref<?xi64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %olen_index = arith.index_cast %olen : i64 to index
    scf.for %i = %c0 to %olen_index step %c1 {
      func.call @__ly_set_raw_insert_slot(%self, %oi, %i) : (memref<?xi64>, memref<?xi64>, index) -> ()
    }
    func.return
  }

  // set_merge: the receiver absorbs another SET. All three branches are
  // CPython's, and all three are load-bearing for the ORDER of the result --
  // dropping the two fast paths disagrees with python3.14 on 274 of 2000
  // measured `set.copy()` pairs, and dropping the up-front resize on 1342.
  func.func private @__ly_set_raw_merge_set(%self: memref<?xi64>, %other: memref<?xi64>) {
    %low32 = arith.constant 4294967295 : i64
    %thirty_two = arith.constant 32 : i64
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %fill_limit_scale = arith.constant 3 : i64
    %probe_scale = arith.constant 5 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16_words = func.call @__ly_box_word_count() : () -> i64
    %c16 = arith.index_cast %c16_words : i64 to index
    %c16_i64 = func.call @__ly_box_word_count() : () -> i64
    %true = arith.constant true
    %entity_slot = arith.constant 0 : index
    %length_slot = arith.constant 2 : index
    %mask_slot = arith.constant 6 : index
    %fill_slot = arith.constant 7 : index
    %oused = memref.load %other[%length_slot] : memref<?xi64>
    %nonempty = arith.cmpi sgt, %oused, %zero : i64
    scf.if %nonempty {
      %fill0 = memref.load %self[%fill_slot] : memref<?xi64>
      %mask0 = memref.load %self[%mask_slot] : memref<?xi64>
      %used0 = memref.load %self[%length_slot] : memref<?xi64>
      %after = arith.addi %fill0, %oused : i64
      %loaded = arith.muli %after, %probe_scale : i64
      %room = arith.muli %mask0, %fill_limit_scale : i64
      %crowded = arith.cmpi sge, %loaded, %room : i64
      scf.if %crowded {
        %total = arith.addi %used0, %oused : i64
        %target = arith.muli %total, %two : i64
        func.call @__ly_set_raw_resize(%self, %target) : (memref<?xi64>, i64) -> ()
      }
      %fill = memref.load %self[%fill_slot] : memref<?xi64>
      %mask = memref.load %self[%mask_slot] : memref<?xi64>
      %omask = memref.load %other[%mask_slot] : memref<?xi64>
      %ofill = memref.load %other[%fill_slot] : memref<?xi64>
      %empty = arith.cmpi eq, %fill, %zero : i64
      %same_mask = arith.cmpi eq, %mask, %omask : i64
      %no_dummies = arith.cmpi eq, %ofill, %oused : i64
      %m0 = arith.andi %empty, %same_mask : i1
      %wholesale = arith.andi %m0, %no_dummies : i1
      scf.if %wholesale {
        // Identical geometry and nothing to skip: the table and the dense array
        // are copied verbatim, which is the only branch that preserves the
        // source's slot positions exactly.
        func.call @__ly_set_raw_ensure_capacity(%self, %oused) : (memref<?xi64>, i64) -> ()
        %items = func.call @__ly_set_raw_items(%self) : (memref<?xi64>) -> memref<?xi64>
        %oitems = func.call @__ly_set_raw_items(%other) : (memref<?xi64>) -> memref<?xi64>
        %words_i64 = arith.muli %oused, %c16_i64 : i64
        %words = arith.index_cast %words_i64 : i64 to index
        scf.for %w = %c0 to %words step %c1 {
          %word = memref.load %oitems[%w] : memref<?xi64>
          memref.store %word, %items[%w] : memref<?xi64>
        }
        %oused_index = arith.index_cast %oused : i64 to index
        scf.for %e = %c0 to %oused_index step %c1 {
          %base = arith.muli %e, %c16 : index
          %entity_index = arith.addi %base, %entity_slot : index
          %entity = memref.load %items[%entity_index] : memref<?xi64>
          func.call @__ly_handle_retain_raw(%entity) : (i64) -> ()
        }
        %table = func.call @__ly_set_raw_table(%self) : (memref<?xi64>) -> memref<?xi64>
        %otable = func.call @__ly_set_raw_table(%other) : (memref<?xi64>) -> memref<?xi64>
        %slots = arith.addi %mask, %one : i64
        %table_words_i64 = arith.muli %slots, %one : i64
        %table_words = arith.index_cast %table_words_i64 : i64 to index
        scf.for %w = %c0 to %table_words step %c1 {
          %word = memref.load %otable[%w] : memref<?xi64>
          memref.store %word, %table[%w] : memref<?xi64>
        }
        memref.store %oused, %self[%length_slot] : memref<?xi64>
        memref.store %ofill, %self[%fill_slot] : memref<?xi64>
      } else {
        scf.if %empty {
          // Nothing can already be present, so every entry lands by
          // set_insert_clean; the dense array is then permuted into the slot
          // order those choices produced.
          func.call @__ly_set_raw_ensure_capacity(%self, %oused) : (memref<?xi64>, i64) -> ()
          %table = func.call @__ly_set_raw_table(%self) : (memref<?xi64>) -> memref<?xi64>
          %otable = func.call @__ly_set_raw_table(%other) : (memref<?xi64>) -> memref<?xi64>
          %oitems = func.call @__ly_set_raw_items(%other) : (memref<?xi64>) -> memref<?xi64>
          %oentry_words = func.call @__ly_box_word_count() : () -> i64
          %oitems_idx = memref.extract_aligned_pointer_as_index %oitems : memref<?xi64> -> index
          %oitems_i64 = arith.index_cast %oitems_idx : index to i64
          %oitems_ptr = llvm.inttoptr %oitems_i64 : i64 to !llvm.ptr
          %oslots = arith.addi %omask, %one : i64
          %oslots_index = arith.index_cast %oslots : i64 to index
          scf.for %s = %c0 to %oslots_index step %c1 {
            %ss = arith.index_cast %s : index to i64
            %state_index = arith.index_cast %ss : i64 to index
            %state_word = memref.load %otable[%state_index] : memref<?xi64>
            %state = arith.andi %state_word, %low32 : i64
            %live = arith.cmpi sge, %state, %two : i64
            scf.if %live {
              %odense = arith.subi %state, %two : i64
              %oentry_off = arith.muli %odense, %oentry_words : i64
              %oentry = llvm.getelementptr %oitems_ptr[%oentry_off] : (!llvm.ptr, i64) -> !llvm.ptr, i64
              %hash = func.call @__ly_set_entry_hash(%oentry) : (!llvm.ptr) -> i64
              %slot = func.call @__ly_set_table_clean_slot(%table, %mask, %hash) : (memref<?xi64>, i64, i64) -> i64
              %dst_index = arith.index_cast %slot : i64 to index
              %placed = func.call @__ly_set_slot_pack(%state, %hash) : (i64, i64) -> i64
              memref.store %placed, %table[%dst_index] : memref<?xi64>
            }
          }
          func.call @__ly_set_table_rebuild_dense(%self, %oitems, %true) : (memref<?xi64>, memref<?xi64>, i1) -> ()
          memref.store %oused, %self[%fill_slot] : memref<?xi64>
        } else {
          // Duplicates are possible: ordinary insertions, in the source's slot
          // order, which is its dense order.
          %oitems = func.call @__ly_set_raw_items(%other) : (memref<?xi64>) -> memref<?xi64>
          %oused_index = arith.index_cast %oused : i64 to index
          scf.for %e = %c0 to %oused_index step %c1 {
            func.call @__ly_set_raw_insert_slot(%self, %oitems, %e) : (memref<?xi64>, memref<?xi64>, index) -> ()
          }
        }
      }
    }
    func.return
  }

  // Discard a raw box if present; answers whether it was.
  func.func private @__ly_set_raw_discard_box(%self: memref<?xi64>, %entry: !llvm.ptr, %hash: i64) -> i1 {
    %minus_one = arith.constant -1 : i64
    %found = func.call @__ly_set_table_lookup(%self, %entry, %hash) : (memref<?xi64>, !llvm.ptr, i64) -> i64
    %present = arith.cmpi ne, %found, %minus_one : i64
    scf.if %present {
      func.call @__ly_set_raw_discard_dense(%self, %found) : (memref<?xi64>, i64) -> ()
    }
    func.return %present : i1
  }

  // set_difference_update_internal's tail: dummies past a fifth of the table
  // are resized away. Without it a repeated discard/add cycle keeps a table
  // CPython would have rebuilt, and the two orders drift apart for good.
  func.func private @__ly_set_raw_shed_dummies(%self: memref<?xi64>) {
    %probe_scale = arith.constant 5 : i64
    %two = arith.constant 2 : i64
    %big = arith.constant 50000 : i64
    %length_slot = arith.constant 2 : index
    %mask_slot = arith.constant 6 : index
    %fill_slot = arith.constant 7 : index
    %fill = memref.load %self[%fill_slot] : memref<?xi64>
    %used = memref.load %self[%length_slot] : memref<?xi64>
    %mask = memref.load %self[%mask_slot] : memref<?xi64>
    %dummies = arith.subi %fill, %used : i64
    %scaled = arith.muli %dummies, %probe_scale : i64
    %crowded = arith.cmpi sge, %scaled, %mask : i64
    scf.if %crowded {
      %huge = arith.cmpi sgt, %used, %big : i64
      %doubled = arith.muli %used, %two : i64
      %target = arith.select %huge, %doubled, %used : i1, i64
      func.call @__ly_set_raw_resize(%self, %target) : (memref<?xi64>, i64) -> ()
    }
    func.return
  }

  // set_swap_bodies, narrowed to what an in-place update needs: the receiver
  // takes over the freshly built set's storage and the temporary is left
  // holding the receiver's, so releasing it frees what the receiver dropped.
  func.func private @__ly_set_raw_swap_bodies(%lhs: memref<?xi64>, %rhs: memref<?xi64>) {
    %c2 = arith.constant 2 : index
    // Through word 8, which is the dense array's order flag: it belongs to the
    // payload it describes, so a body swap that left it behind would tell one
    // handle the other's array is in order.
    %c8 = arith.constant 9 : index
    %c1 = arith.constant 1 : index
    scf.for %w = %c2 to %c8 step %c1 {
      %l = memref.load %lhs[%w] : memref<?xi64>
      %r = memref.load %rhs[%w] : memref<?xi64>
      memref.store %r, %lhs[%w] : memref<?xi64>
      memref.store %l, %rhs[%w] : memref<?xi64>
    }
    func.return
  }

  // Insert each entry of %a whose membership in %b equals `want_present`. The
  // scan is over %a's DENSE array, which is its table order, so the result is
  // built in the order CPython's set_next hands the entries out.
  func.func private @__ly_set_raw_select(%self: memref<?xi64>, %a: memref<?xi64>, %b: memref<?xi64>, %want_present: i1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16_i64 = func.call @__ly_box_word_count() : () -> i64
    %minus_one = arith.constant -1 : i64
    %length_slot = arith.constant 2 : index
    %alen = memref.load %a[%length_slot] : memref<?xi64>
    %ai = func.call @__ly_set_raw_items(%a) : (memref<?xi64>) -> memref<?xi64>
    %alen_index = arith.index_cast %alen : i64 to index
    %ai_idx = memref.extract_aligned_pointer_as_index %ai : memref<?xi64> -> index
    %ai_i64 = arith.index_cast %ai_idx : index to i64
    %ai_ptr = llvm.inttoptr %ai_i64 : i64 to !llvm.ptr
    scf.for %i = %c0 to %alen_index step %c1 {
      %ii = arith.index_cast %i : index to i64
      %off = arith.muli %ii, %c16_i64 : i64
      %entry = llvm.getelementptr %ai_ptr[%off] : (!llvm.ptr, i64) -> !llvm.ptr, i64
      %entry_hash = func.call @__ly_set_entry_hash(%entry) : (!llvm.ptr) -> i64
      %in_b = func.call @__ly_set_table_lookup(%b, %entry, %entry_hash) : (memref<?xi64>, !llvm.ptr, i64) -> i64
      %present = arith.cmpi ne, %in_b, %minus_one : i64
      %keep = arith.cmpi eq, %present, %want_present : i1
      scf.if %keep {
        func.call @__ly_set_raw_insert_slot(%self, %ai, %i) : (memref<?xi64>, memref<?xi64>, index) -> ()
      }
    }
    func.return
  }

  // Drop from the receiver every entry whose membership in %other equals
  // %drop_present, walking the receiver back to front so a dense removal never
  // moves an entry the walk has still to visit.
  //
  // ⛔ Why NOT the compacting forward filter this replaces: a discard leaves a
  // DUMMY behind, and the dummy is what a later insert reuses -- rebuilding the
  // dense prefix without it would put the receiver in a state no CPython
  // sequence reaches, and the next insertion would then land somewhere CPython
  // does not put it.
  func.func private @__ly_set_raw_drop_matching(%self: memref<?xi64>, %other: memref<?xi64>, %drop_present: i1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16_i64 = func.call @__ly_box_word_count() : () -> i64
    %minus_one = arith.constant -1 : i64
    %length_slot = arith.constant 2 : index
    %len = memref.load %self[%length_slot] : memref<?xi64>
    %len_index = arith.index_cast %len : i64 to index
    scf.for %n = %c0 to %len_index step %c1 {
      %back = arith.subi %len_index, %n : index
      %i = arith.subi %back, %c1 : index
      %items = func.call @__ly_set_raw_items(%self) : (memref<?xi64>) -> memref<?xi64>
      %items_idx = memref.extract_aligned_pointer_as_index %items : memref<?xi64> -> index
      %items_i64 = arith.index_cast %items_idx : index to i64
      %items_ptr = llvm.inttoptr %items_i64 : i64 to !llvm.ptr
      %ii = arith.index_cast %i : index to i64
      %off = arith.muli %ii, %c16_i64 : i64
      %entry = llvm.getelementptr %items_ptr[%off] : (!llvm.ptr, i64) -> !llvm.ptr, i64
      %entry_hash = func.call @__ly_set_entry_hash(%entry) : (!llvm.ptr) -> i64
      %in_other = func.call @__ly_set_table_lookup(%other, %entry, %entry_hash) : (memref<?xi64>, !llvm.ptr, i64) -> i64
      %present = arith.cmpi ne, %in_other, %minus_one : i64
      %drop = arith.cmpi eq, %present, %drop_present : i1
      scf.if %drop {
        func.call @__ly_set_raw_discard_dense(%self, %ii) : (memref<?xi64>, i64) -> ()
      }
    }
    func.return
  }

  // set_symmetric_difference_update's loop: each entry of %other is discarded
  // from the receiver if present and added if not. %other is not mutated, so
  // its dense array is a stable walk -- the `so is other` case is a clear and
  // never reaches here.
  func.func private @__ly_set_raw_toggle(%self: memref<?xi64>, %other: memref<?xi64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %c16_i64 = func.call @__ly_box_word_count() : () -> i64
    %length_slot = arith.constant 2 : index
    %olen = memref.load %other[%length_slot] : memref<?xi64>
    %oi = func.call @__ly_set_raw_items(%other) : (memref<?xi64>) -> memref<?xi64>
    %olen_index = arith.index_cast %olen : i64 to index
    %oi_idx = memref.extract_aligned_pointer_as_index %oi : memref<?xi64> -> index
    %oi_i64 = arith.index_cast %oi_idx : index to i64
    %oi_ptr = llvm.inttoptr %oi_i64 : i64 to !llvm.ptr
    scf.for %i = %c0 to %olen_index step %c1 {
      %ii = arith.index_cast %i : index to i64
      %off = arith.muli %ii, %c16_i64 : i64
      %entry = llvm.getelementptr %oi_ptr[%off] : (!llvm.ptr, i64) -> !llvm.ptr, i64
      %hash = func.call @__ly_set_entry_hash(%entry) : (!llvm.ptr) -> i64
      %removed = func.call @__ly_set_raw_discard_box(%self, %entry, %hash) : (memref<?xi64>, !llvm.ptr, i64) -> i1
      %absent = arith.xori %removed, %true : i1
      scf.if %absent {
        func.call @__ly_set_raw_insert_slot(%self, %oi, %i) : (memref<?xi64>, memref<?xi64>, index) -> ()
      }
    }
    func.return
  }

  // Subset core: every element of (alen, ai) is in (blen, bi).
  func.func private @__ly_set_subset_lens(%alen: i64, %ai: memref<?xi64>, %blen: i64, %bi: memref<?xi64>) -> i1 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16_i64 = func.call @__ly_box_word_count() : () -> i64
    %minus_one = arith.constant -1 : i64
    %true = arith.constant true
    %false = arith.constant false
    %alen_index = arith.index_cast %alen : i64 to index
    %ai_idx = memref.extract_aligned_pointer_as_index %ai : memref<?xi64> -> index
    %ai_i64 = arith.index_cast %ai_idx : index to i64
    %ai_ptr = llvm.inttoptr %ai_i64 : i64 to !llvm.ptr
    %all = scf.for %i = %c0 to %alen_index step %c1 iter_args(%ok = %true) -> (i1) {
      %next = scf.if %ok -> (i1) {
        %ii = arith.index_cast %i : index to i64
        %off = arith.muli %ii, %c16_i64 : i64
        %entry = llvm.getelementptr %ai_ptr[%off] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        %entry_hash = func.call @__ly_box_hash(%entry) : (!llvm.ptr) -> i64
        %in_b = func.call @__ly_set_probe(%blen, %bi, %entry, %entry_hash) : (i64, memref<?xi64>, !llvm.ptr, i64) -> i64
        %present = arith.cmpi ne, %in_b, %minus_one : i64
        scf.yield %present : i1
      } else {
        scf.yield %false : i1
      }
      scf.yield %next : i1
    }
    func.return %all : i1
  }

  // Disjointness core: no element of (alen, ai) is in (blen, bi).
  func.func private @__ly_set_disjoint_lens(%alen: i64, %ai: memref<?xi64>, %blen: i64, %bi: memref<?xi64>) -> i1 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16_i64 = func.call @__ly_box_word_count() : () -> i64
    %minus_one = arith.constant -1 : i64
    %true = arith.constant true
    %false = arith.constant false
    %alen_index = arith.index_cast %alen : i64 to index
    %ai_idx = memref.extract_aligned_pointer_as_index %ai : memref<?xi64> -> index
    %ai_i64 = arith.index_cast %ai_idx : index to i64
    %ai_ptr = llvm.inttoptr %ai_i64 : i64 to !llvm.ptr
    %disjoint = scf.for %i = %c0 to %alen_index step %c1 iter_args(%ok = %true) -> (i1) {
      %next = scf.if %ok -> (i1) {
        %ii = arith.index_cast %i : index to i64
        %off = arith.muli %ii, %c16_i64 : i64
        %entry = llvm.getelementptr %ai_ptr[%off] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        %entry_hash = func.call @__ly_box_hash(%entry) : (!llvm.ptr) -> i64
        %in_b = func.call @__ly_set_probe(%blen, %bi, %entry, %entry_hash) : (i64, memref<?xi64>, !llvm.ptr, i64) -> i64
        %absent = arith.cmpi eq, %in_b, %minus_one : i64
        scf.yield %absent : i1
      } else {
        scf.yield %false : i1
      }
      scf.yield %next : i1
    }
    func.return %disjoint : i1
  }

  // Release every element box and reset the length to zero, keeping the items
  // allocation (set.clear).
  func.func private @__ly_set_raw_clear(%self: memref<?xi64>) {
    %zero = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16_words = func.call @__ly_box_word_count() : () -> i64
    %c16 = arith.index_cast %c16_words : i64 to index
    %length_slot = arith.constant 2 : index
    %len = memref.load %self[%length_slot] : memref<?xi64>
    %items = func.call @__ly_set_raw_items(%self) : (memref<?xi64>) -> memref<?xi64>
    %len_index = arith.index_cast %len : i64 to index
    scf.for %i = %c0 to %len_index step %c1 {
      %ii = arith.index_cast %i : index to i64
      func.call @LyObject_ReleaseBoxedPayloadArraySlotRaw(%items, %ii) : (memref<?xi64>, i64) -> ()
    }
    %total = arith.muli %len_index, %c16 : index
    scf.for %w = %c0 to %total step %c1 {
      memref.store %zero, %items[%w] : memref<?xi64>
    }
    memref.store %zero, %self[%length_slot] : memref<?xi64>
    // set_clear_internal drops back to the minimum table, dummies and all.
    %one = arith.constant 1 : i64
    %min_table = arith.constant 8 : i64
    %table_slot = arith.constant 5 : index
    %mask_slot = arith.constant 6 : index
    %fill_slot = arith.constant 7 : index
    %old_table = memref.load %self[%table_slot] : memref<?xi64>
    %fresh = func.call @__ly_set_table_new(%min_table) : (i64) -> i64
    %min_mask = arith.subi %min_table, %one : i64
    memref.store %fresh, %self[%table_slot] : memref<?xi64>
    memref.store %min_mask, %self[%mask_slot] : memref<?xi64>
    memref.store %zero, %self[%fill_slot] : memref<?xi64>
    func.call @free_raw_i64_ptr(%old_table) : (i64) -> ()
    func.return
  }

  // Release every element box and free the items array. The handle's own
  // storage is deallocated by the caller, which is the only part that knows
  // the width.
  func.func private @__ly_set_raw_release_payload(%self: memref<?xi64>) {
    %items_slot = arith.constant 4 : index
    %lower = arith.constant 0 : index
    %step = arith.constant 1 : index
    %length_slot = arith.constant 2 : index
    %length = memref.load %self[%length_slot] : memref<?xi64>
    %items = func.call @__ly_set_raw_items(%self) : (memref<?xi64>) -> memref<?xi64>
    %length_index = arith.index_cast %length : i64 to index
    scf.for %i = %lower to %length_index step %step {
      %logical_index = arith.index_cast %i : index to i64
      func.call @LyObject_ReleaseBoxedPayloadArraySlotRaw(%items, %logical_index) : (memref<?xi64>, i64) -> ()
    }
    %items_word = memref.load %self[%items_slot] : memref<?xi64>
    func.call @free_raw_i64_ptr(%items_word) : (i64) -> ()
    %table_slot = arith.constant 5 : index
    %table_word = memref.load %self[%table_slot] : memref<?xi64>
    func.call @free_raw_i64_ptr(%table_word) : (i64) -> ()
    func.return
  }

  // ---- builtins.set --------------------------------------------------------

  func.func private @__ly_set_alloc(%length: i64) -> memref<9xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.set"], ly.ownership.owned_results = [0]} {
    %zero = arith.constant 0 : i64
    %class_id = arith.constant 21 : i64
    %order_slot = arith.constant 8 : index
    %handle_bytes = arith.constant 72 : index
    %handle_block = memref.alloc(%handle_bytes) {alignment = 16 : i64} : memref<?xi8>
    %handle_at = arith.constant 0 : index
    %self = memref.view %handle_block[%handle_at][] {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<?xi8> to memref<9xi64>
    memref.store %zero, %self[%order_slot] : memref<9xi64>
    %raw = memref.cast %self : memref<9xi64> to memref<?xi64>
    func.call @__ly_set_raw_init(%raw, %class_id, %length) : (memref<?xi64>, i64, i64) -> ()
    func.return %self : memref<9xi64>
  }

  // Borrowed view of the items array. Per-contract so the primitive carries the
  // contract name the ownership walk keys on; the body is the shared one.
  func.func private @__ly_set_items(%self: memref<9xi64>) -> memref<?xi64> attributes {ly.runtime.contract = "builtins.set", ly.runtime.interior_word, ly.runtime.primitive = "items_view"} {
    %raw = memref.cast %self : memref<9xi64> to memref<?xi64>
    %view = func.call @__ly_set_raw_items(%raw) : (memref<?xi64>) -> memref<?xi64>
    func.return %view : memref<?xi64>
  }

  // set_copy: an empty set of the MINIMUM table size, then set_merge. The
  // empty-and-same-mask branch of the merge is what makes a small copy come out
  // byte-identical; going through a bulk fill of the dense array instead would
  // give the copy no table at all.
  func.func private @__ly_set_copy_alloc(%src: memref<?xi64>) -> memref<9xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.set"], ly.ownership.owned_results = [0]} {
    // The merge below can take the source's table and dense array VERBATIM,
    // which copies whatever order the source is in; every other path rebuilds
    // the order through `place` and gets it back at the next repr. So the
    // source is put in order first, and the copy inherits one that is already
    // right.
    func.call @__ly_set_raw_reorder(%src) : (memref<?xi64>) -> ()
    %zero = arith.constant 0 : i64
    %self = func.call @__ly_set_alloc(%zero) : (i64) -> memref<9xi64>
    %raw = memref.cast %self : memref<9xi64> to memref<?xi64>
    func.call @__ly_set_raw_merge_set(%raw, %src) : (memref<?xi64>, memref<?xi64>) -> ()
    func.return %self : memref<9xi64>
  }

  func.func @LySet_FromLength(%length: i64 {ly.runtime.default_i64 = 0 : i64}) -> memref<9xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 21 : i64, ly.runtime.contract = "builtins.set", ly.runtime.initializer = "__new__", ly.runtime.result_contract = "builtins.set"} {
    %self = func.call @__ly_set_alloc(%length) : (i64) -> memref<9xi64>
    func.return %self : memref<9xi64>
  }

  func.func @LySet_Len(%self: memref<9xi64> {ly.ownership.object_header}) -> i64 attributes {ly.runtime.contract = "builtins.set", ly.runtime.method = "__len__"} {
    %length_slot = arith.constant 2 : index
    %length = memref.load %self[%length_slot] : memref<9xi64>
    func.return %length : i64
  }

  // The set contract declares __bool__, so `if s:` dispatched to it and the
  // lowering refused -- "runtime manifest has no builtins.set.__bool__
  // method". list and dict have no declaration and fall to __len__, which is
  // why theirs worked. Implemented rather than the declaration removed: a set
  // IS falsy when empty and typeshed says the method exists, so the missing
  // half was the implementation.
  func.func @LySet_Bool(%self: memref<9xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.set", ly.runtime.method = "__bool__"} {
    %length_slot = arith.constant 2 : index
    %zero = arith.constant 0 : i64
    %length = memref.load %self[%length_slot] : memref<9xi64>
    %non_empty = arith.cmpi ne, %length, %zero : i64
    func.return %non_empty : i1
  }

  // Runtime set insert with a boxed element (any hashable class; raises
  // TypeError for unhashable elements). The caller retained the box; a
  // duplicate element consumes it here. Void and non-transferring: the growth
  // publishes the new items base through the handle.
  func.func @LySet_AddBox(%self: memref<9xi64> {ly.ownership.object_header}, %elem_box: memref<5xi64>) attributes {ly.runtime.contract = "builtins.set", ly.runtime.primitive = "add_box"} {
    %zero = arith.constant 0 : i64
    %minus_one = arith.constant -1 : i64
    %box_idx = memref.extract_aligned_pointer_as_index %elem_box : memref<5xi64> -> index
    %box_i64 = arith.index_cast %box_idx : index to i64
    %box_ptr = llvm.inttoptr %box_i64 : i64 to !llvm.ptr
    %hash = func.call @__ly_box_hash(%box_ptr) : (!llvm.ptr) -> i64
    %raw = memref.cast %self : memref<9xi64> to memref<?xi64>
    %src = memref.cast %elem_box : memref<5xi64> to memref<?xi64>
    %probe:3 = func.call @__ly_set_table_add_probe(%raw, %box_ptr, %hash) : (memref<?xi64>, !llvm.ptr, i64) -> (i64, i64, i1)
    %missing = arith.cmpi eq, %probe#0, %minus_one : i64
    scf.if %missing {
      %entity = func.call @__ly_set_raw_place(%raw, %probe#1, %probe#2, %src, %zero, %hash) : (memref<?xi64>, i64, i1, memref<?xi64>, i64, i64) -> i64
    } else {
      // Duplicate: consume the caller's retained box.
      func.call @LyObject_ReleaseBoxedPayloadRaw(%elem_box) : (memref<5xi64>) -> ()
    }
    func.return
  }

  // Membership probe with a BORROWED transient box (not consumed).
  func.func @LySet_ContainsBox(%self: memref<9xi64> {ly.ownership.object_header}, %elem_box: memref<5xi64>) -> i1 attributes {ly.runtime.contract = "builtins.set", ly.runtime.primitive = "contains_box"} {
    %minus_one = arith.constant -1 : i64
    %box_idx = memref.extract_aligned_pointer_as_index %elem_box : memref<5xi64> -> index
    %box_i64 = arith.index_cast %box_idx : index to i64
    %box_ptr = llvm.inttoptr %box_i64 : i64 to !llvm.ptr
    %hash = func.call @__ly_box_hash(%box_ptr) : (!llvm.ptr) -> i64
    %raw = memref.cast %self : memref<9xi64> to memref<?xi64>
    %found = func.call @__ly_set_raw_probe(%raw, %box_ptr, %hash) : (memref<?xi64>, !llvm.ptr, i64) -> i64
    %result = arith.cmpi ne, %found, %minus_one : i64
    func.return %result : i1
  }

  // set.discard: remove when present, silent otherwise.
  func.func @LySet_DiscardBox(%self: memref<9xi64> {ly.ownership.object_header}, %elem_box: memref<5xi64>) attributes {ly.runtime.contract = "builtins.set", ly.runtime.primitive = "discard_box"} {
    %minus_one = arith.constant -1 : i64
    %box_idx = memref.extract_aligned_pointer_as_index %elem_box : memref<5xi64> -> index
    %box_i64 = arith.index_cast %box_idx : index to i64
    %box_ptr = llvm.inttoptr %box_i64 : i64 to !llvm.ptr
    %hash = func.call @__ly_box_hash(%box_ptr) : (!llvm.ptr) -> i64
    %raw = memref.cast %self : memref<9xi64> to memref<?xi64>
    %found = func.call @__ly_set_raw_probe(%raw, %box_ptr, %hash) : (memref<?xi64>, !llvm.ptr, i64) -> i64
    %present = arith.cmpi ne, %found, %minus_one : i64
    scf.if %present {
      func.call @__ly_set_raw_remove_slot(%raw, %found) : (memref<?xi64>, i64) -> ()
    }
    func.return
  }

  // set.remove: KeyError carrying the element's repr on a miss.
  func.func @LySet_RemoveBox(%self: memref<9xi64> {ly.ownership.object_header}, %elem_box: memref<5xi64>) attributes {ly.runtime.contract = "builtins.set", ly.runtime.primitive = "remove_box"} {
    %minus_one = arith.constant -1 : i64
    %box_idx = memref.extract_aligned_pointer_as_index %elem_box : memref<5xi64> -> index
    %box_i64 = arith.index_cast %box_idx : index to i64
    %box_ptr = llvm.inttoptr %box_i64 : i64 to !llvm.ptr
    %hash = func.call @__ly_box_hash(%box_ptr) : (!llvm.ptr) -> i64
    %raw = memref.cast %self : memref<9xi64> to memref<?xi64>
    %found = func.call @__ly_set_raw_probe(%raw, %box_ptr, %hash) : (memref<?xi64>, !llvm.ptr, i64) -> i64
    %missing = arith.cmpi eq, %found, %minus_one : i64
    scf.if %missing {
      func.call @__ly_dict_raise_missing_key(%box_ptr) : (!llvm.ptr) -> ()
    }
    func.call @__ly_set_raw_remove_slot(%raw, %found) : (memref<?xi64>, i64) -> ()
    func.return
  }

  // set.clear / set.copy and the binary set algebra.
  func.func @LySet_Clear(%self: memref<9xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "builtins.set", ly.runtime.method = "clear"} {
    %raw = memref.cast %self : memref<9xi64> to memref<?xi64>
    func.call @__ly_set_raw_clear(%raw) : (memref<?xi64>) -> ()
    func.return
  }

  func.func @LySet_Copy(%self: memref<9xi64> {ly.ownership.object_header}) -> memref<9xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.set", ly.runtime.method = "copy", ly.runtime.result_contract = "builtins.set"} {
    %raw = memref.cast %self : memref<9xi64> to memref<?xi64>
    %copy = func.call @__ly_set_copy_alloc(%raw) : (memref<?xi64>) -> memref<9xi64>
    func.return %copy : memref<9xi64>
  }

  // set_union: a copy of the receiver, then set_merge of the argument.
  func.func @LySet_Union(%lhs: memref<9xi64> {ly.ownership.object_header}, %rhs: memref<9xi64> {ly.ownership.object_header}) -> memref<9xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.set", ly.runtime.method = "union", ly.runtime.result_contract = "builtins.set"} {
    %lraw = memref.cast %lhs : memref<9xi64> to memref<?xi64>
    %rraw = memref.cast %rhs : memref<9xi64> to memref<?xi64>
    %result = func.call @__ly_set_copy_alloc(%lraw) : (memref<?xi64>) -> memref<9xi64>
    %raw = memref.cast %result : memref<9xi64> to memref<?xi64>
    func.call @__ly_set_raw_merge_set(%raw, %rraw) : (memref<?xi64>, memref<?xi64>) -> ()
    func.return %result : memref<9xi64>
  }

  // set_intersection scans the SMALLER operand and probes the larger, and the
  // scan order is the result's insertion order -- so the swap is not only the
  // faster arrangement, it is which answer CPython prints (44 of 2000 measured
  // pairs disagree without it).
  func.func @LySet_Intersection(%lhs: memref<9xi64> {ly.ownership.object_header}, %rhs: memref<9xi64> {ly.ownership.object_header}) -> memref<9xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.set", ly.runtime.method = "intersection", ly.runtime.result_contract = "builtins.set"} {
    %zero = arith.constant 0 : i64
    %true = arith.constant true
    %length_slot = arith.constant 2 : index
    %result = func.call @__ly_set_alloc(%zero) : (i64) -> memref<9xi64>
    %llen = memref.load %lhs[%length_slot] : memref<9xi64>
    %rlen = memref.load %rhs[%length_slot] : memref<9xi64>
    %lraw = memref.cast %lhs : memref<9xi64> to memref<?xi64>
    %rraw = memref.cast %rhs : memref<9xi64> to memref<?xi64>
    %raw = memref.cast %result : memref<9xi64> to memref<?xi64>
    %rhs_bigger = arith.cmpi sgt, %rlen, %llen : i64
    scf.if %rhs_bigger {
      func.call @__ly_set_raw_select(%raw, %lraw, %rraw, %true) : (memref<?xi64>, memref<?xi64>, memref<?xi64>, i1) -> ()
    } else {
      func.call @__ly_set_raw_select(%raw, %rraw, %lraw, %true) : (memref<?xi64>, memref<?xi64>, memref<?xi64>, i1) -> ()
    }
    func.return %result : memref<9xi64>
  }

  // set_difference. When the receiver is more than four times the argument,
  // CPython copies it and discards the common part instead of rebuilding --
  // and a discard leaves a dummy where a rebuild leaves nothing, so the two
  // spellings print different orders (195 of 2000 measured pairs).
  func.func @LySet_Difference(%lhs: memref<9xi64> {ly.ownership.object_header}, %rhs: memref<9xi64> {ly.ownership.object_header}) -> memref<9xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.set", ly.runtime.method = "difference", ly.runtime.result_contract = "builtins.set"} {
    %zero = arith.constant 0 : i64
    %two = arith.constant 2 : i64
    %false = arith.constant false
    %true = arith.constant true
    %length_slot = arith.constant 2 : index
    %llen = memref.load %lhs[%length_slot] : memref<9xi64>
    %rlen = memref.load %rhs[%length_slot] : memref<9xi64>
    %lraw = memref.cast %lhs : memref<9xi64> to memref<?xi64>
    %rraw = memref.cast %rhs : memref<9xi64> to memref<?xi64>
    %quarter = arith.shrsi %llen, %two : i64
    %much_bigger = arith.cmpi sgt, %quarter, %rlen : i64
    %result = scf.if %much_bigger -> (memref<9xi64>) {
      %copied = func.call @__ly_set_copy_alloc(%lraw) : (memref<?xi64>) -> memref<9xi64>
      %craw = memref.cast %copied : memref<9xi64> to memref<?xi64>
      func.call @__ly_set_raw_drop_matching(%craw, %rraw, %true) : (memref<?xi64>, memref<?xi64>, i1) -> ()
      scf.yield %copied : memref<9xi64>
    } else {
      %fresh = func.call @__ly_set_alloc(%zero) : (i64) -> memref<9xi64>
      %fraw = memref.cast %fresh : memref<9xi64> to memref<?xi64>
      func.call @__ly_set_raw_select(%fraw, %lraw, %rraw, %false) : (memref<?xi64>, memref<?xi64>, memref<?xi64>, i1) -> ()
      scf.yield %fresh : memref<9xi64>
    }
    func.return %result : memref<9xi64>
  }

  // set_symmetric_difference: a copy of the ARGUMENT, then the receiver's
  // entries toggled into it one at a time.
  //
  // ⛔ Why NOT difference(l, r) followed by difference(r, l), which is what
  // this used to do and what the identity suggests: the two halves are two
  // independent build orders concatenated, and CPython's is one build order
  // through a single table.
  func.func @LySet_SymmetricDifference(%lhs: memref<9xi64> {ly.ownership.object_header}, %rhs: memref<9xi64> {ly.ownership.object_header}) -> memref<9xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.set", ly.runtime.method = "symmetric_difference", ly.runtime.result_contract = "builtins.set"} {
    %lraw = memref.cast %lhs : memref<9xi64> to memref<?xi64>
    %rraw = memref.cast %rhs : memref<9xi64> to memref<?xi64>
    %result = func.call @__ly_set_copy_alloc(%rraw) : (memref<?xi64>) -> memref<9xi64>
    %raw = memref.cast %result : memref<9xi64> to memref<?xi64>
    func.call @__ly_set_raw_toggle(%raw, %lraw) : (memref<?xi64>, memref<?xi64>) -> ()
    func.return %result : memref<9xi64>
  }

  func.func @LySet_IsSubset(%lhs: memref<9xi64> {ly.ownership.object_header}, %rhs: memref<9xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.set", ly.runtime.method = "issubset"} {
    %length_slot = arith.constant 2 : index
    %llen = memref.load %lhs[%length_slot] : memref<9xi64>
    %li = func.call @__ly_set_items(%lhs) : (memref<9xi64>) -> memref<?xi64>
    %rlen = memref.load %rhs[%length_slot] : memref<9xi64>
    %ri = func.call @__ly_set_items(%rhs) : (memref<9xi64>) -> memref<?xi64>
    %r = func.call @__ly_set_subset_lens(%llen, %li, %rlen, %ri) : (i64, memref<?xi64>, i64, memref<?xi64>) -> i1
    func.return %r : i1
  }

  func.func @LySet_IsSuperset(%lhs: memref<9xi64> {ly.ownership.object_header}, %rhs: memref<9xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.set", ly.runtime.method = "issuperset"} {
    %r = func.call @LySet_IsSubset(%rhs, %lhs) : (memref<9xi64>, memref<9xi64>) -> i1
    func.return %r : i1
  }

  func.func @LySet_IsDisjoint(%lhs: memref<9xi64> {ly.ownership.object_header}, %rhs: memref<9xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.set", ly.runtime.method = "isdisjoint"} {
    %length_slot = arith.constant 2 : index
    %llen = memref.load %lhs[%length_slot] : memref<9xi64>
    %li = func.call @__ly_set_items(%lhs) : (memref<9xi64>) -> memref<?xi64>
    %rlen = memref.load %rhs[%length_slot] : memref<9xi64>
    %ri = func.call @__ly_set_items(%rhs) : (memref<9xi64>) -> memref<?xi64>
    %r = func.call @__ly_set_disjoint_lens(%llen, %li, %rlen, %ri) : (i64, memref<?xi64>, i64, memref<?xi64>) -> i1
    func.return %r : i1
  }

  func.func @LySet_EqBool(%lhs: memref<9xi64> {ly.ownership.object_header}, %rhs: memref<9xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.set", ly.runtime.method = "__eq__"} {
    %false = arith.constant false
    %length_slot = arith.constant 2 : index
    %llen = memref.load %lhs[%length_slot] : memref<9xi64>
    %rlen = memref.load %rhs[%length_slot] : memref<9xi64>
    %same_len = arith.cmpi eq, %llen, %rlen : i64
    %result = scf.if %same_len -> (i1) {
      %sub = func.call @LySet_IsSubset(%lhs, %rhs) : (memref<9xi64>, memref<9xi64>) -> i1
      scf.yield %sub : i1
    } else {
      scf.yield %false : i1
    }
    func.return %result : i1
  }

  func.func @LySet_NeBool(%lhs: memref<9xi64> {ly.ownership.object_header}, %rhs: memref<9xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.set", ly.runtime.method = "__ne__"} {
    %eq = func.call @LySet_EqBool(%lhs, %rhs) : (memref<9xi64>, memref<9xi64>) -> i1
    %true = arith.constant true
    %ne = arith.xori %eq, %true : i1
    func.return %ne : i1
  }

  // Operator aliases over the set algebra (| & - ^ and the subset order).
  func.func @LySet_OrOp(%lhs: memref<9xi64> {ly.ownership.object_header}, %rhs: memref<9xi64> {ly.ownership.object_header}) -> memref<9xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.set", ly.runtime.method = "__or__", ly.runtime.result_contract = "builtins.set"} {
    %r = func.call @LySet_Union(%lhs, %rhs) : (memref<9xi64>, memref<9xi64>) -> memref<9xi64>
    func.return %r : memref<9xi64>
  }

  func.func @LySet_AndOp(%lhs: memref<9xi64> {ly.ownership.object_header}, %rhs: memref<9xi64> {ly.ownership.object_header}) -> memref<9xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.set", ly.runtime.method = "__and__", ly.runtime.result_contract = "builtins.set"} {
    %r = func.call @LySet_Intersection(%lhs, %rhs) : (memref<9xi64>, memref<9xi64>) -> memref<9xi64>
    func.return %r : memref<9xi64>
  }

  func.func @LySet_SubOp(%lhs: memref<9xi64> {ly.ownership.object_header}, %rhs: memref<9xi64> {ly.ownership.object_header}) -> memref<9xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.set", ly.runtime.method = "__sub__", ly.runtime.result_contract = "builtins.set"} {
    %r = func.call @LySet_Difference(%lhs, %rhs) : (memref<9xi64>, memref<9xi64>) -> memref<9xi64>
    func.return %r : memref<9xi64>
  }

  func.func @LySet_XorOp(%lhs: memref<9xi64> {ly.ownership.object_header}, %rhs: memref<9xi64> {ly.ownership.object_header}) -> memref<9xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.set", ly.runtime.method = "__xor__", ly.runtime.result_contract = "builtins.set"} {
    %r = func.call @LySet_SymmetricDifference(%lhs, %rhs) : (memref<9xi64>, memref<9xi64>) -> memref<9xi64>
    func.return %r : memref<9xi64>
  }

  func.func @LySet_LeBool(%lhs: memref<9xi64> {ly.ownership.object_header}, %rhs: memref<9xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.set", ly.runtime.method = "__le__"} {
    %r = func.call @LySet_IsSubset(%lhs, %rhs) : (memref<9xi64>, memref<9xi64>) -> i1
    func.return %r : i1
  }

  func.func @LySet_LtBool(%lhs: memref<9xi64> {ly.ownership.object_header}, %rhs: memref<9xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.set", ly.runtime.method = "__lt__"} {
    %length_slot = arith.constant 2 : index
    %sub = func.call @LySet_IsSubset(%lhs, %rhs) : (memref<9xi64>, memref<9xi64>) -> i1
    %llen = memref.load %lhs[%length_slot] : memref<9xi64>
    %rlen = memref.load %rhs[%length_slot] : memref<9xi64>
    %proper = arith.cmpi ne, %llen, %rlen : i64
    %r = arith.andi %sub, %proper : i1
    func.return %r : i1
  }

  func.func @LySet_GeBool(%lhs: memref<9xi64> {ly.ownership.object_header}, %rhs: memref<9xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.set", ly.runtime.method = "__ge__"} {
    %r = func.call @LySet_IsSubset(%rhs, %lhs) : (memref<9xi64>, memref<9xi64>) -> i1
    func.return %r : i1
  }

  func.func @LySet_GtBool(%lhs: memref<9xi64> {ly.ownership.object_header}, %rhs: memref<9xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.set", ly.runtime.method = "__gt__"} {
    %length_slot = arith.constant 2 : index
    %sub = func.call @LySet_IsSubset(%rhs, %lhs) : (memref<9xi64>, memref<9xi64>) -> i1
    %llen = memref.load %lhs[%length_slot] : memref<9xi64>
    %rlen = memref.load %rhs[%length_slot] : memref<9xi64>
    %proper = arith.cmpi ne, %llen, %rlen : i64
    %r = arith.andi %sub, %proper : i1
    func.return %r : i1
  }

  // In-place update family. All four are VOID and declare no transfer_args:
  // the growth publishes through the handle and the removals leave the handle
  // naming the same entity, so there is no renamed representation to hand back.
  // These four plus LySet_AddBox and LyFrozenSet_Init were the six
  // transfer_args declarations this block owned.
  //
  // Each one is CPython's, and the three that REMOVE are not the same shape:
  // intersection_update builds the intersection and swaps bodies (so the
  // receiver ends up with the intersection's table), while difference_update
  // and symmetric_difference_update mutate the receiver's own table and leave
  // the dummies where they fall.
  func.func @LySet_UpdateM(%self: memref<9xi64> {ly.ownership.object_header}, %other: memref<9xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "builtins.set", ly.runtime.method = "update"} {
    %raw = memref.cast %self : memref<9xi64> to memref<?xi64>
    %oraw = memref.cast %other : memref<9xi64> to memref<?xi64>
    func.call @__ly_set_raw_merge_set(%raw, %oraw) : (memref<?xi64>, memref<?xi64>) -> ()
    func.return
  }

  func.func @LySet_IntersectionUpdate(%self: memref<9xi64> {ly.ownership.object_header}, %other: memref<9xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "builtins.set", ly.runtime.method = "intersection_update"} {
    %fresh = func.call @LySet_Intersection(%self, %other) : (memref<9xi64>, memref<9xi64>) -> memref<9xi64>
    %raw = memref.cast %self : memref<9xi64> to memref<?xi64>
    %fraw = memref.cast %fresh : memref<9xi64> to memref<?xi64>
    func.call @__ly_set_raw_swap_bodies(%raw, %fraw) : (memref<?xi64>, memref<?xi64>) -> ()
    // The temporary now holds what the receiver dropped, so releasing it is
    // the discard -- exactly what set_swap_bodies + Py_DECREF does.
    func.call @LySet_DecRef(%fresh) : (memref<9xi64>) -> ()
    func.return
  }

  func.func @LySet_DifferenceUpdate(%self: memref<9xi64> {ly.ownership.object_header}, %other: memref<9xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "builtins.set", ly.runtime.method = "difference_update"} {
    %true = arith.constant true
    %raw = memref.cast %self : memref<9xi64> to memref<?xi64>
    %oraw = memref.cast %other : memref<9xi64> to memref<?xi64>
    func.call @__ly_set_raw_drop_matching(%raw, %oraw, %true) : (memref<?xi64>, memref<?xi64>, i1) -> ()
    func.call @__ly_set_raw_shed_dummies(%raw) : (memref<?xi64>) -> ()
    func.return
  }

  func.func @LySet_SymmetricDifferenceUpdate(%self: memref<9xi64> {ly.ownership.object_header}, %other: memref<9xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "builtins.set", ly.runtime.method = "symmetric_difference_update"} {
    %raw = memref.cast %self : memref<9xi64> to memref<?xi64>
    %oraw = memref.cast %other : memref<9xi64> to memref<?xi64>
    %self_idx = memref.extract_aligned_pointer_as_index %self : memref<9xi64> -> index
    %other_idx = memref.extract_aligned_pointer_as_index %other : memref<9xi64> -> index
    %same = arith.cmpi eq, %self_idx, %other_idx : index
    scf.if %same {
      // s ^= s is a clear, and it has to be spelled: the toggle loop would
      // walk the very array it is emptying.
      func.call @__ly_set_raw_clear(%raw) : (memref<?xi64>) -> ()
    } else {
      func.call @__ly_set_raw_toggle(%raw, %oraw) : (memref<?xi64>, memref<?xi64>) -> ()
    }
    func.return
  }

  func.func @LySet_DecRef(%self: memref<9xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.set", ly.runtime.deallocator} {
    %storage = memref.cast %self : memref<9xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    func.call @__ly_set_raw_release_payload(%storage) : (memref<?xi64>) -> ()
    memref.dealloc %self : memref<9xi64>
    cf.br ^done

  ^done:
    func.return
  }

  // ===== impls: frozenset =====
  // Physically a set (dense insertion-ordered boxed slots) with class id 23
  // and no mutators. The wrappers delegate to the shared core rather than to
  // the LySet_* wrappers: a frozenset is not a set, and the LySet_* wrappers
  // name `builtins.set` in their contracts. Hashing is CPython's frozenset_hash
  // (commutative shuffle-xor over entry hashes, so equal frozensets hash equal
  // regardless of insertion order).
  func.func private @LyFrozenSet_Shape() -> memref<9xi64> attributes {ly.runtime.contract = "builtins.frozenset", ly.runtime.shape}

  func.func private @__ly_frozenset_alloc(%length: i64) -> memref<9xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.frozenset"], ly.ownership.owned_results = [0]} {
    %zero = arith.constant 0 : i64
    %class_id = arith.constant 23 : i64
    %order_slot = arith.constant 8 : index
    %handle_bytes = arith.constant 72 : index
    %handle_block = memref.alloc(%handle_bytes) {alignment = 16 : i64} : memref<?xi8>
    %handle_at = arith.constant 0 : index
    %self = memref.view %handle_block[%handle_at][] {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<?xi8> to memref<9xi64>
    memref.store %zero, %self[%order_slot] : memref<9xi64>
    %raw = memref.cast %self : memref<9xi64> to memref<?xi64>
    func.call @__ly_set_raw_init(%raw, %class_id, %length) : (memref<?xi64>, i64, i64) -> ()
    func.return %self : memref<9xi64>
  }

  func.func private @__ly_frozenset_items(%self: memref<9xi64>) -> memref<?xi64> attributes {ly.runtime.contract = "builtins.frozenset", ly.runtime.interior_word, ly.runtime.primitive = "items_view"} {
    %raw = memref.cast %self : memref<9xi64> to memref<?xi64>
    %view = func.call @__ly_set_raw_items(%raw) : (memref<?xi64>) -> memref<?xi64>
    func.return %view : memref<?xi64>
  }

  // frozenset(iterable): dedupe through the set probe (the source may be a
  // list/tuple with repeats; sets are already unique but share the loop). The
  // source arrives as a count and an items array rather than as a contract's
  // shape, because it is polymorphic over all four sequence contracts and
  // their shapes no longer agree.
  func.func @LyFrozenSet_FromElements(%olen: i64, %oi: memref<?xi64>) -> memref<9xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 23 : i64, ly.runtime.contract = "builtins.frozenset", ly.runtime.initializer = "__new__", ly.runtime.result_contract = "builtins.frozenset"} {
    %zero = arith.constant 0 : i64
    %self = func.call @__ly_frozenset_alloc(%zero) : (i64) -> memref<9xi64>
    %raw = memref.cast %self : memref<9xi64> to memref<?xi64>
    // __ly_set_raw_merge already skips a slot the destination holds, so the
    // dedupe is the merge itself rather than a second probe here.
    func.call @__ly_set_raw_merge(%raw, %olen, %oi) : (memref<?xi64>, i64, memref<?xi64>) -> ()
    func.return %self : memref<9xi64>
  }

  // __init__ is a no-op: frozenset is immutable, so __new__ (FromElements)
  // already absorbed the source; CPython splits construction the same way.
  // Void, like LyRange_Init and LyFloat_Init: the three-lane form returned a
  // renamed receiver and so had to declare transfer_args = [0] with
  // owned_results = [0], which is the shape this conversion exists to remove.
  func.func @LyFrozenSet_Init(%self: memref<9xi64> {ly.ownership.object_header}, %olen: i64, %oi: memref<?xi64>) attributes {ly.runtime.contract = "builtins.frozenset", ly.runtime.method = "__init__"} {
    func.return
  }

  func.func @LyFrozenSet_Len(%self: memref<9xi64> {ly.ownership.object_header}) -> i64 attributes {ly.runtime.contract = "builtins.frozenset", ly.runtime.method = "__len__"} {
    %length_slot = arith.constant 2 : index
    %length = memref.load %self[%length_slot] : memref<9xi64>
    func.return %length : i64
  }

  func.func @LyFrozenSet_Bool(%self: memref<9xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.frozenset", ly.runtime.method = "__bool__"} {
    %length_slot = arith.constant 2 : index
    %zero = arith.constant 0 : i64
    %length = memref.load %self[%length_slot] : memref<9xi64>
    %non_empty = arith.cmpi ne, %length, %zero : i64
    func.return %non_empty : i1
  }

  func.func @LyFrozenSet_ContainsBox(%self: memref<9xi64> {ly.ownership.object_header}, %elem_box: memref<5xi64>) -> i1 attributes {ly.runtime.contract = "builtins.frozenset", ly.runtime.primitive = "contains_box"} {
    %minus_one = arith.constant -1 : i64
    %box_idx = memref.extract_aligned_pointer_as_index %elem_box : memref<5xi64> -> index
    %box_i64 = arith.index_cast %box_idx : index to i64
    %box_ptr = llvm.inttoptr %box_i64 : i64 to !llvm.ptr
    %hash = func.call @__ly_box_hash(%box_ptr) : (!llvm.ptr) -> i64
    %raw = memref.cast %self : memref<9xi64> to memref<?xi64>
    %found = func.call @__ly_set_raw_probe(%raw, %box_ptr, %hash) : (memref<?xi64>, !llvm.ptr, i64) -> i64
    %result = arith.cmpi ne, %found, %minus_one : i64
    func.return %result : i1
  }

  // CPython setobject.c frozenset_hash: commutative shuffle-xor, then a
  // length-dependent scramble (all arithmetic wraps mod 2^64 by design).
  func.func @LyFrozenSet_Hash(%self: memref<9xi64> {ly.ownership.object_header}) -> i64 attributes {ly.runtime.contract = "builtins.frozenset", ly.runtime.method = "__hash__"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16_i64 = func.call @__ly_box_word_count() : () -> i64
    %c16_shift = arith.constant 16 : i64
    %magic1 = arith.constant 89869747 : i64
    %magic2 = arith.constant 3644798167 : i64
    %magic3 = arith.constant 1927868237 : i64
    %magic4 = arith.constant 69069 : i64
    %magic5 = arith.constant 907133923 : i64
    %minus_one = arith.constant -1 : i64
    %substitute = arith.constant 590923713 : i64
    %one = arith.constant 1 : i64
    %zero = arith.constant 0 : i64
    %length_slot = arith.constant 2 : index
    %len = memref.load %self[%length_slot] : memref<9xi64>
    %items = func.call @__ly_frozenset_items(%self) : (memref<9xi64>) -> memref<?xi64>
    %len_index = arith.index_cast %len : i64 to index
    %items_idx = memref.extract_aligned_pointer_as_index %items : memref<?xi64> -> index
    %items_i64 = arith.index_cast %items_idx : index to i64
    %items_ptr = llvm.inttoptr %items_i64 : i64 to !llvm.ptr
    %mixed = scf.for %i = %c0 to %len_index step %c1 iter_args(%h = %zero) -> (i64) {
      %ii = arith.index_cast %i : index to i64
      %off = arith.muli %ii, %c16_i64 : i64
      %entry = llvm.getelementptr %items_ptr[%off] : (!llvm.ptr, i64) -> !llvm.ptr, i64
      %eh = func.call @__ly_box_hash(%entry) : (!llvm.ptr) -> i64
      %x1 = arith.xori %eh, %magic1 : i64
      %shifted = arith.shli %eh, %c16_shift : i64
      %x2 = arith.xori %x1, %shifted : i64
      %shuffled = arith.muli %x2, %magic2 : i64
      %next = arith.xori %h, %shuffled : i64
      scf.yield %next : i64
    }
    %lenp1 = arith.addi %len, %one : i64
    %scale = arith.muli %lenp1, %magic3 : i64
    %h2 = arith.xori %mixed, %scale : i64
    %h3 = arith.muli %h2, %magic4 : i64
    %h4 = arith.addi %h3, %magic5 : i64
    %is_minus_one = arith.cmpi eq, %h4, %minus_one : i64
    %final = arith.select %is_minus_one, %substitute, %h4 : i1, i64
    func.return %final : i64
  }

  func.func @LyFrozenSet_EqBool(%lhs: memref<9xi64> {ly.ownership.object_header}, %rhs: memref<9xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.frozenset", ly.runtime.method = "__eq__"} {
    %false = arith.constant false
    %length_slot = arith.constant 2 : index
    %llen = memref.load %lhs[%length_slot] : memref<9xi64>
    %rlen = memref.load %rhs[%length_slot] : memref<9xi64>
    %same_len = arith.cmpi eq, %llen, %rlen : i64
    %result = scf.if %same_len -> (i1) {
      %sub = func.call @LyFrozenSet_IsSubset(%lhs, %rhs) : (memref<9xi64>, memref<9xi64>) -> i1
      scf.yield %sub : i1
    } else {
      scf.yield %false : i1
    }
    func.return %result : i1
  }

  func.func @LyFrozenSet_NeBool(%lhs: memref<9xi64> {ly.ownership.object_header}, %rhs: memref<9xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.frozenset", ly.runtime.method = "__ne__"} {
    %eq = func.call @LyFrozenSet_EqBool(%lhs, %rhs) : (memref<9xi64>, memref<9xi64>) -> i1
    %true = arith.constant true
    %ne = arith.xori %eq, %true : i1
    func.return %ne : i1
  }

  // The frozenset algebra is the set algebra with the other handle width; each
  // one is the same CPython function, so the shapes are kept side by side
  // rather than each simplified on its own.
  func.func private @__ly_frozenset_copy_alloc(%src: memref<?xi64>) -> memref<9xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.frozenset"], ly.ownership.owned_results = [0]} {
    %zero = arith.constant 0 : i64
    %self = func.call @__ly_frozenset_alloc(%zero) : (i64) -> memref<9xi64>
    %raw = memref.cast %self : memref<9xi64> to memref<?xi64>
    func.call @__ly_set_raw_merge_set(%raw, %src) : (memref<?xi64>, memref<?xi64>) -> ()
    func.return %self : memref<9xi64>
  }

  func.func @LyFrozenSet_Union(%lhs: memref<9xi64> {ly.ownership.object_header}, %rhs: memref<9xi64> {ly.ownership.object_header}) -> memref<9xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.frozenset", ly.runtime.method = "union", ly.runtime.result_contract = "builtins.frozenset"} {
    %lraw = memref.cast %lhs : memref<9xi64> to memref<?xi64>
    %rraw = memref.cast %rhs : memref<9xi64> to memref<?xi64>
    %result = func.call @__ly_frozenset_copy_alloc(%lraw) : (memref<?xi64>) -> memref<9xi64>
    %raw = memref.cast %result : memref<9xi64> to memref<?xi64>
    func.call @__ly_set_raw_merge_set(%raw, %rraw) : (memref<?xi64>, memref<?xi64>) -> ()
    func.return %result : memref<9xi64>
  }

  func.func @LyFrozenSet_Intersection(%lhs: memref<9xi64> {ly.ownership.object_header}, %rhs: memref<9xi64> {ly.ownership.object_header}) -> memref<9xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.frozenset", ly.runtime.method = "intersection", ly.runtime.result_contract = "builtins.frozenset"} {
    %zero = arith.constant 0 : i64
    %true = arith.constant true
    %length_slot = arith.constant 2 : index
    %result = func.call @__ly_frozenset_alloc(%zero) : (i64) -> memref<9xi64>
    %llen = memref.load %lhs[%length_slot] : memref<9xi64>
    %rlen = memref.load %rhs[%length_slot] : memref<9xi64>
    %lraw = memref.cast %lhs : memref<9xi64> to memref<?xi64>
    %rraw = memref.cast %rhs : memref<9xi64> to memref<?xi64>
    %raw = memref.cast %result : memref<9xi64> to memref<?xi64>
    %rhs_bigger = arith.cmpi sgt, %rlen, %llen : i64
    scf.if %rhs_bigger {
      func.call @__ly_set_raw_select(%raw, %lraw, %rraw, %true) : (memref<?xi64>, memref<?xi64>, memref<?xi64>, i1) -> ()
    } else {
      func.call @__ly_set_raw_select(%raw, %rraw, %lraw, %true) : (memref<?xi64>, memref<?xi64>, memref<?xi64>, i1) -> ()
    }
    func.return %result : memref<9xi64>
  }

  func.func @LyFrozenSet_Difference(%lhs: memref<9xi64> {ly.ownership.object_header}, %rhs: memref<9xi64> {ly.ownership.object_header}) -> memref<9xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.frozenset", ly.runtime.method = "difference", ly.runtime.result_contract = "builtins.frozenset"} {
    %zero = arith.constant 0 : i64
    %two = arith.constant 2 : i64
    %false = arith.constant false
    %true = arith.constant true
    %length_slot = arith.constant 2 : index
    %llen = memref.load %lhs[%length_slot] : memref<9xi64>
    %rlen = memref.load %rhs[%length_slot] : memref<9xi64>
    %lraw = memref.cast %lhs : memref<9xi64> to memref<?xi64>
    %rraw = memref.cast %rhs : memref<9xi64> to memref<?xi64>
    %quarter = arith.shrsi %llen, %two : i64
    %much_bigger = arith.cmpi sgt, %quarter, %rlen : i64
    %result = scf.if %much_bigger -> (memref<9xi64>) {
      %copied = func.call @__ly_frozenset_copy_alloc(%lraw) : (memref<?xi64>) -> memref<9xi64>
      %craw = memref.cast %copied : memref<9xi64> to memref<?xi64>
      func.call @__ly_set_raw_drop_matching(%craw, %rraw, %true) : (memref<?xi64>, memref<?xi64>, i1) -> ()
      scf.yield %copied : memref<9xi64>
    } else {
      %fresh = func.call @__ly_frozenset_alloc(%zero) : (i64) -> memref<9xi64>
      %fraw = memref.cast %fresh : memref<9xi64> to memref<?xi64>
      func.call @__ly_set_raw_select(%fraw, %lraw, %rraw, %false) : (memref<?xi64>, memref<?xi64>, memref<?xi64>, i1) -> ()
      scf.yield %fresh : memref<9xi64>
    }
    func.return %result : memref<9xi64>
  }

  func.func @LyFrozenSet_SymmetricDifference(%lhs: memref<9xi64> {ly.ownership.object_header}, %rhs: memref<9xi64> {ly.ownership.object_header}) -> memref<9xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.frozenset", ly.runtime.method = "symmetric_difference", ly.runtime.result_contract = "builtins.frozenset"} {
    %lraw = memref.cast %lhs : memref<9xi64> to memref<?xi64>
    %rraw = memref.cast %rhs : memref<9xi64> to memref<?xi64>
    %result = func.call @__ly_frozenset_copy_alloc(%rraw) : (memref<?xi64>) -> memref<9xi64>
    %raw = memref.cast %result : memref<9xi64> to memref<?xi64>
    func.call @__ly_set_raw_toggle(%raw, %lraw) : (memref<?xi64>, memref<?xi64>) -> ()
    func.return %result : memref<9xi64>
  }

  func.func @LyFrozenSet_OrOp(%lhs: memref<9xi64> {ly.ownership.object_header}, %rhs: memref<9xi64> {ly.ownership.object_header}) -> memref<9xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.frozenset", ly.runtime.method = "__or__", ly.runtime.result_contract = "builtins.frozenset"} {
    %result = func.call @LyFrozenSet_Union(%lhs, %rhs) : (memref<9xi64>, memref<9xi64>) -> memref<9xi64>
    func.return %result : memref<9xi64>
  }

  func.func @LyFrozenSet_AndOp(%lhs: memref<9xi64> {ly.ownership.object_header}, %rhs: memref<9xi64> {ly.ownership.object_header}) -> memref<9xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.frozenset", ly.runtime.method = "__and__", ly.runtime.result_contract = "builtins.frozenset"} {
    %result = func.call @LyFrozenSet_Intersection(%lhs, %rhs) : (memref<9xi64>, memref<9xi64>) -> memref<9xi64>
    func.return %result : memref<9xi64>
  }

  func.func @LyFrozenSet_SubOp(%lhs: memref<9xi64> {ly.ownership.object_header}, %rhs: memref<9xi64> {ly.ownership.object_header}) -> memref<9xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.frozenset", ly.runtime.method = "__sub__", ly.runtime.result_contract = "builtins.frozenset"} {
    %result = func.call @LyFrozenSet_Difference(%lhs, %rhs) : (memref<9xi64>, memref<9xi64>) -> memref<9xi64>
    func.return %result : memref<9xi64>
  }

  func.func @LyFrozenSet_XorOp(%lhs: memref<9xi64> {ly.ownership.object_header}, %rhs: memref<9xi64> {ly.ownership.object_header}) -> memref<9xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.frozenset", ly.runtime.method = "__xor__", ly.runtime.result_contract = "builtins.frozenset"} {
    %result = func.call @LyFrozenSet_SymmetricDifference(%lhs, %rhs) : (memref<9xi64>, memref<9xi64>) -> memref<9xi64>
    func.return %result : memref<9xi64>
  }

  func.func @LyFrozenSet_IsSubset(%lhs: memref<9xi64> {ly.ownership.object_header}, %rhs: memref<9xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.frozenset", ly.runtime.method = "issubset"} {
    %length_slot = arith.constant 2 : index
    %llen = memref.load %lhs[%length_slot] : memref<9xi64>
    %li = func.call @__ly_frozenset_items(%lhs) : (memref<9xi64>) -> memref<?xi64>
    %rlen = memref.load %rhs[%length_slot] : memref<9xi64>
    %ri = func.call @__ly_frozenset_items(%rhs) : (memref<9xi64>) -> memref<?xi64>
    %r = func.call @__ly_set_subset_lens(%llen, %li, %rlen, %ri) : (i64, memref<?xi64>, i64, memref<?xi64>) -> i1
    func.return %r : i1
  }

  func.func @LyFrozenSet_IsSuperset(%lhs: memref<9xi64> {ly.ownership.object_header}, %rhs: memref<9xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.frozenset", ly.runtime.method = "issuperset"} {
    %result = func.call @LyFrozenSet_IsSubset(%rhs, %lhs) : (memref<9xi64>, memref<9xi64>) -> i1
    func.return %result : i1
  }

  func.func @LyFrozenSet_IsDisjoint(%lhs: memref<9xi64> {ly.ownership.object_header}, %rhs: memref<9xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.frozenset", ly.runtime.method = "isdisjoint"} {
    %length_slot = arith.constant 2 : index
    %llen = memref.load %lhs[%length_slot] : memref<9xi64>
    %li = func.call @__ly_frozenset_items(%lhs) : (memref<9xi64>) -> memref<?xi64>
    %rlen = memref.load %rhs[%length_slot] : memref<9xi64>
    %ri = func.call @__ly_frozenset_items(%rhs) : (memref<9xi64>) -> memref<?xi64>
    %r = func.call @__ly_set_disjoint_lens(%llen, %li, %rlen, %ri) : (i64, memref<?xi64>, i64, memref<?xi64>) -> i1
    func.return %r : i1
  }

  func.func @LyFrozenSet_LeBool(%lhs: memref<9xi64> {ly.ownership.object_header}, %rhs: memref<9xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.frozenset", ly.runtime.method = "__le__"} {
    %result = func.call @LyFrozenSet_IsSubset(%lhs, %rhs) : (memref<9xi64>, memref<9xi64>) -> i1
    func.return %result : i1
  }

  func.func @LyFrozenSet_LtBool(%lhs: memref<9xi64> {ly.ownership.object_header}, %rhs: memref<9xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.frozenset", ly.runtime.method = "__lt__"} {
    %length_slot = arith.constant 2 : index
    %sub = func.call @LyFrozenSet_IsSubset(%lhs, %rhs) : (memref<9xi64>, memref<9xi64>) -> i1
    %llen = memref.load %lhs[%length_slot] : memref<9xi64>
    %rlen = memref.load %rhs[%length_slot] : memref<9xi64>
    %proper = arith.cmpi ne, %llen, %rlen : i64
    %result = arith.andi %sub, %proper : i1
    func.return %result : i1
  }

  func.func @LyFrozenSet_GeBool(%lhs: memref<9xi64> {ly.ownership.object_header}, %rhs: memref<9xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.frozenset", ly.runtime.method = "__ge__"} {
    %result = func.call @LyFrozenSet_IsSubset(%rhs, %lhs) : (memref<9xi64>, memref<9xi64>) -> i1
    func.return %result : i1
  }

  func.func @LyFrozenSet_GtBool(%lhs: memref<9xi64> {ly.ownership.object_header}, %rhs: memref<9xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.frozenset", ly.runtime.method = "__gt__"} {
    %length_slot = arith.constant 2 : index
    %sub = func.call @LyFrozenSet_IsSubset(%rhs, %lhs) : (memref<9xi64>, memref<9xi64>) -> i1
    %llen = memref.load %lhs[%length_slot] : memref<9xi64>
    %rlen = memref.load %rhs[%length_slot] : memref<9xi64>
    %proper = arith.cmpi ne, %llen, %rlen : i64
    %result = arith.andi %sub, %proper : i1
    func.return %result : i1
  }

  func.func @LyFrozenSet_DecRef(%self: memref<9xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.frozenset", ly.runtime.deallocator} {
    %storage = memref.cast %self : memref<9xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    func.call @__ly_set_raw_release_payload(%storage) : (memref<?xi64>) -> ()
    memref.dealloc %self : memref<9xi64>
    cf.br ^done

  ^done:
    func.return
  }
}
