// `list` -- CPython's Objects/listobject.c (list.sort's run lengths are
// merge_compute_minrun's).
//
// Signature source (1:1 correspondence target):
//   https://github.com/python/typeshed/blob/main/stdlib/builtins.pyi

module attributes {
  ly.typing.manifest,
  ly.runtime.contracts = ["builtins.list"]
} {
  // ===== declared here, defined in another runtime file or built by the lowering =====
  func.func private @__ly_slice_unpack(%self: memref<5xi64>) -> (i64, i64, i64, i64)
  func.func private @__ly_unicode_from_valid_utf8(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]}
  func.func private @LyErr_NoMemory() attributes {ly.runtime.contract = "builtins.MemoryError"}
  func.func private @LyLong_FromI64(%value: i64 {ly.runtime.default_i64 = 0 : i64}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 1 : i64, ly.runtime.contract = "builtins.int", ly.runtime.initializer = "__new__"}
  func.func private @LyObject_ReleaseBoxedPayloadArraySlotRaw(%payload: memref<?xi64>, %logical_index: i64)
  func.func private @LyObject_ReleaseBoxedPayloadRaw(%box: memref<5xi64>)
  func.func private @LyObject_ReleaseStorageToZero(%storage: memref<?xi64>) -> i1 attributes {ly.runtime.contract = "builtins.object", ly.runtime.primitive = "release_to_zero"}
  func.func private @LyObject_RetainBoxedPayloadArraySlotRaw(%payload: memref<?xi64>, %logical_index: i64)
  func.func private @LyUnicode_Concat(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_bytes: memref<?xi8>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__add__"}
  func.func private @LyUnicode_DecRef(%header: memref<2xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.str", ly.runtime.deallocator}
  func.func private @LyUnicode_FromBytes(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 4 : i64, ly.runtime.contract = "builtins.str", ly.runtime.initializer = "__new__"}
  func.func private @__ly_alloc_count(%count: i64, %element_bytes: i64, %extra: i64) -> index
  func.func private @__ly_box_move_slot(%dst: memref<?xi64>, %d: i64, %src: memref<?xi64>, %s: i64)
  func.func private @__ly_box_word_count() -> i64
  func.func private @__ly_check_alloc_count(%count: i64, %element_bytes: i64, %extra: i64)
  func.func private @__ly_global_view_i64(%pointer: i64, %size: i64) -> memref<?xi64>
  func.func private @__ly_long_parts(%header: memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
  func.func private @__ly_raise_static_message(%class_id: i64, %message: memref<?xi8>, %length: i64)
  func.func private @__ly_repeat_overflows(%len: i64, %n: i64) -> i1
  func.func private @__ly_repr_boxed_or_default(%box_ptr: !llvm.ptr, %class_id: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.object", ly.runtime.primitive = "repr_boxed_or_default", ly.runtime.result_contract = "builtins.str"}
  memref.global "private" constant @__ly_repr_comma : memref<2xi8>
  memref.global "private" constant @__ly_repr_lbracket : memref<1xi8>
  memref.global "private" constant @__ly_repr_rbracket : memref<1xi8>
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
  func.func private @__ly_slice_adjust(%len: i64, %start_in: i64, %stop_in: i64, %step: i64, %mask: i64) -> (i64, i64)
  func.func private @__ly_slice_raise_extended_mismatch(%prefix: memref<?xi8>, %prefix_len: i64, %src_len: i64, %slice_len: i64)
  func.func private @__ly_slice_raise_zero_step()
  func.func private @__ly_slot_class(%word: i64) -> i64
  func.func private @__ly_slot_less(%items_ptr: !llvm.ptr, %a: i64, %b: i64) -> i1
  func.func private @free_raw_i64_ptr(%address: i64)
  func.func private @realloc_raw_i64_ptr(%address: i64, %bytes: i64) -> i64

  py.class @list attributes {
    base_names = ["MutableSequence"], ly.typing.params = ["T"],
    ly.runtime.contract = "builtins.list", ly.runtime.required,
    ly.runtime.required_deallocator,
    ly.runtime.required_initializers = ["__new__"],
    ly.runtime.required_methods = ["__len__"],
    ly.runtime.required_primitives = ["ensure_capacity"],
    ly.typing.structural_mutators = ["append", "extend", "insert", "__setslice__", "__delslice__"],
    ly.typing.base_args = [[!py.contract<"$T">]],
    method_names = ["__init__", "__init__", "append", "extend", "pop", "pop",
                    "insert", "remove", "clear", "__len__", "__iter__",
                    "__getitem__", "__getslice__", "__setslice__",
                    "__delslice__", "__setitem__", "__delitem__",
                    "__contains__", "__repr__", "sort", "reverse", "copy",
                    "count", "index", "__add__", "__mul__", "__eq__",
                    "__ne__", "__lt__", "__le__", "__gt__", "__ge__", "__getslice__", "__setslice__", "__delslice__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"builtins.list">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">, !py.protocol<"Iterable", [!py.contract<"$T">]>] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">, !py.contract<"$T">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">, !py.protocol<"Iterable", [!py.contract<"$T">]>] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">] -> [!py.contract<"$T">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">, !py.contract<"typing.SupportsIndex">] -> [!py.contract<"$T">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">, !py.contract<"typing.SupportsIndex">, !py.contract<"$T">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">, !py.contract<"$T">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">] -> [!py.protocol<"Iterator", [!py.contract<"$T">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">, !py.contract<"typing.SupportsIndex">] -> [!py.contract<"$T">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.list", [!py.contract<"$T">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.list", [!py.contract<"$T">]>] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">, !py.contract<"typing.SupportsIndex">, !py.contract<"$T">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">, !py.contract<"typing.SupportsIndex">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">] -> [!py.contract<"builtins.list", [!py.contract<"$T">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">, !py.contract<"$T">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">, !py.contract<"$T">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">, !py.contract<"builtins.list", [!py.contract<"$T">]>] -> [!py.contract<"builtins.list", [!py.contract<"$T">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.list", [!py.contract<"$T">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">, !py.contract<"builtins.list">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">, !py.contract<"builtins.list">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">, !py.contract<"builtins.list">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">, !py.contract<"builtins.list">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">, !py.contract<"builtins.slice">] -> [!py.contract<"builtins.list", [!py.contract<"$T">]>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">, !py.contract<"builtins.slice">, !py.contract<"builtins.list", [!py.contract<"$T">]>] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.list">, !py.contract<"builtins.slice">] -> [!py.literal<None>]>
    ],
    method_kinds = ["instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance", "instance", "instance", "instance"]
  } {}

  // ===== impls: list methods =====
  func.func private @__ly_swap_slots(%items: memref<?xi64>, %a: index, %b: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16_words = func.call @__ly_box_word_count() : () -> i64
    %c16 = arith.index_cast %c16_words : i64 to index
    %a_base = arith.muli %a, %c16 : index
    %b_base = arith.muli %b, %c16 : index
    scf.for %w = %c0 to %c16 step %c1 {
      %ai = arith.addi %a_base, %w : index
      %bi = arith.addi %b_base, %w : index
      %av = memref.load %items[%ai] : memref<?xi64>
      %bv = memref.load %items[%bi] : memref<?xi64>
      memref.store %bv, %items[%ai] : memref<?xi64>
      memref.store %av, %items[%bi] : memref<?xi64>
    }
    func.return
  }

  // Stable in-place insertion sort over boxed slots, ordered by
  // __ly_box_less (adjacent swaps only while strictly less, so equal
  // elements keep their relative order).
  // CPython's merge_compute_minrun (listobject.c): the low bits of n rolled
  // into the answer, so n/minrun is just under a power of two and the merge
  // passes stay balanced. 32..64 for anything worth merging.
  func.func private @__ly_sort_minrun(%n0: i64) -> i64 {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %c64 = arith.constant 64 : i64
    %walk:2 = scf.while (%n = %n0, %r = %zero) : (i64, i64) -> (i64, i64) {
      %big = arith.cmpi sge, %n, %c64 : i64
      scf.condition(%big) %n, %r : i64, i64
    } do {
    ^bb0(%n: i64, %r: i64):
      %bit = arith.andi %n, %one : i64
      %nr = arith.ori %r, %bit : i64
      %nn = arith.shrui %n, %one : i64
      scf.yield %nn, %nr : i64, i64
    }
    %minrun = arith.addi %walk#0, %walk#1 : i64
    func.return %minrun : i64
  }

  func.func private @__ly_reverse_slots(%items: memref<?xi64>, %lo: i64, %hi: i64) {
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %n = arith.subi %hi, %lo : i64
    %half = arith.divui %n, %two : i64
    %half_index = arith.index_cast %half : i64 to index
    %last = arith.subi %hi, %one : i64
    scf.for %k = %c0 to %half_index step %c1 {
      %kk = arith.index_cast %k : index to i64
      %a = arith.addi %lo, %kk : i64
      %b = arith.subi %last, %kk : i64
      %ai = arith.index_cast %a : i64 to index
      %bi = arith.index_cast %b : i64 to index
      func.call @__ly_swap_slots(%items, %ai, %bi) : (memref<?xi64>, index, index) -> ()
    }
    func.return
  }

  // Sort [lo, hi) in place: take the natural run at %lo -- reversing it when it
  // is strictly descending, which is what keeps a reversed input linear -- and
  // insertion-sort the remainder of the block into it. This is count_run plus
  // binary_sort, and the block is at most minrun long so the quadratic tail is
  // bounded by 64.
  func.func private @__ly_sort_block(%items: memref<?xi64>, %items_ptr: !llvm.ptr, %lo: i64, %hi: i64) {
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %false = arith.constant false
    %c1i = arith.constant 1 : index
    %n = arith.subi %hi, %lo : i64
    %enough = arith.cmpi sge, %n, %two : i64
    scf.if %enough {
      %second = arith.addi %lo, %one : i64
      %desc = func.call @__ly_slot_less(%items_ptr, %second, %lo) : (!llvm.ptr, i64, i64) -> i1
      %start2 = arith.addi %lo, %two : i64
      %end = scf.while (%k = %start2) : (i64) -> i64 {
        %inb = arith.cmpi slt, %k, %hi : i64
        %cont = scf.if %inb -> (i1) {
          %prev = arith.subi %k, %one : i64
          %lt = func.call @__ly_slot_less(%items_ptr, %k, %prev) : (!llvm.ptr, i64, i64) -> i1
          %same = arith.cmpi eq, %lt, %desc : i1
          scf.yield %same : i1
        } else {
          scf.yield %false : i1
        }
        scf.condition(%cont) %k : i64
      } do {
      ^bb0(%k: i64):
        %nk = arith.addi %k, %one : i64
        scf.yield %nk : i64
      }
      scf.if %desc {
        func.call @__ly_reverse_slots(%items, %lo, %end) : (memref<?xi64>, i64, i64) -> ()
      }
      %end_index = arith.index_cast %end : i64 to index
      %hi_index = arith.index_cast %hi : i64 to index
      scf.for %p = %end_index to %hi_index step %c1i {
        %pp = arith.index_cast %p : index to i64
        %fin = scf.while (%j = %pp) : (i64) -> i64 {
          %above = arith.cmpi sgt, %j, %lo : i64
          %swap = scf.if %above -> (i1) {
            %prev = arith.subi %j, %one : i64
            %lt = func.call @__ly_slot_less(%items_ptr, %j, %prev) : (!llvm.ptr, i64, i64) -> i1
            scf.yield %lt : i1
          } else {
            scf.yield %false : i1
          }
          scf.condition(%swap) %j : i64
        } do {
        ^bb0(%j: i64):
          %prev = arith.subi %j, %one : i64
          %ji = arith.index_cast %j : i64 to index
          %pi = arith.index_cast %prev : i64 to index
          func.call @__ly_swap_slots(%items, %ji, %pi) : (memref<?xi64>, index, index) -> ()
          scf.yield %prev : i64
        }
      }
    }
    func.return
  }

  // Merge the adjacent sorted ranges [lo, mid) and [mid, hi).
  //
  // ⭐ THE COMPARISONS COME FIRST AND THE MOVES SECOND, and that split is the
  // whole reason this is safe. `__ly_box_less` can raise -- it reaches
  // `__ly_cmp_raise_unorderable` and a user `__lt__` -- and a merge that
  // interleaved comparing with moving would be caught half way, with the array
  // holding some elements twice and others not at all: the list's destructor
  // would then release one box twice and leak another. Phase A only reads, so
  // an unwind out of it leaves the array exactly as it was, and Phase B calls
  // nothing that can raise.
  //
  // ⛔ What still escapes on that path is the scratch, which the caller
  // allocated. CPython avoids even that by emptying the list for the duration
  // of the sort and cleaning up in its error path; writing that here needs a
  // landing pad the manifest has no spelling for.
  func.func private @__ly_merge_slots(%items: memref<?xi64>, %items_ptr: !llvm.ptr, %scratch: memref<?xi64>, %decisions: memref<?xi8>, %lo: i64, %mid: i64, %hi: i64) {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero8 = arith.constant 0 : i8
    %one8 = arith.constant 1 : i8
    %prev = arith.subi %mid, %one : i64
    // Already in order end to end: CPython's same shortcut, and it is what
    // makes a sorted input cost one comparison per block per pass.
    %out_of_order = func.call @__ly_slot_less(%items_ptr, %mid, %prev) : (!llvm.ptr, i64, i64) -> i1
    scf.if %out_of_order {
      %walk:3 = scf.while (%i = %lo, %j = %mid, %k = %zero) : (i64, i64, i64) -> (i64, i64, i64) {
        %li = arith.cmpi slt, %i, %mid : i64
        %rj = arith.cmpi slt, %j, %hi : i64
        %both = arith.andi %li, %rj : i1
        scf.condition(%both) %i, %j, %k : i64, i64, i64
      } do {
      ^bb0(%i: i64, %j: i64, %k: i64):
        // STRICTLY less, so an equal pair takes the left element: that is what
        // makes the sort stable, and stability is observable.
        %take_right = func.call @__ly_slot_less(%items_ptr, %j, %i) : (!llvm.ptr, i64, i64) -> i1
        %kk = arith.index_cast %k : i64 to index
        %mark = arith.select %take_right, %one8, %zero8 : i8
        memref.store %mark, %decisions[%kk] : memref<?xi8>
        %i_next = arith.addi %i, %one : i64
        %j_next = arith.addi %j, %one : i64
        %ni = arith.select %take_right, %i, %i_next : i64
        %nj = arith.select %take_right, %j_next, %j : i64
        %nk = arith.addi %k, %one : i64
        scf.yield %ni, %nj, %nk : i64, i64, i64
      }
      %n1 = arith.subi %mid, %lo : i64
      %n1_index = arith.index_cast %n1 : i64 to index
      scf.for %t = %c0 to %n1_index step %c1 {
        %tt = arith.index_cast %t : index to i64
        %src = arith.addi %lo, %tt : i64
        func.call @__ly_box_move_slot(%scratch, %tt, %items, %src) : (memref<?xi64>, i64, memref<?xi64>, i64) -> ()
      }
      %ndec_index = arith.index_cast %walk#2 : i64 to index
      %after:3 = scf.for %d = %c0 to %ndec_index step %c1 iter_args(%i2 = %zero, %j2 = %mid, %k2 = %lo) -> (i64, i64, i64) {
        %mark = memref.load %decisions[%d] : memref<?xi8>
        %right = arith.cmpi ne, %mark, %zero8 : i8
        %next:3 = scf.if %right -> (i64, i64, i64) {
          func.call @__ly_box_move_slot(%items, %k2, %items, %j2) : (memref<?xi64>, i64, memref<?xi64>, i64) -> ()
          %nj = arith.addi %j2, %one : i64
          %nk = arith.addi %k2, %one : i64
          scf.yield %i2, %nj, %nk : i64, i64, i64
        } else {
          func.call @__ly_box_move_slot(%items, %k2, %scratch, %i2) : (memref<?xi64>, i64, memref<?xi64>, i64) -> ()
          %ni = arith.addi %i2, %one : i64
          %nk = arith.addi %k2, %one : i64
          scf.yield %ni, %j2, %nk : i64, i64, i64
        }
        scf.yield %next#0, %next#1, %next#2 : i64, i64, i64
      }
      // Whatever is left of the LEFT half. The right half's remainder is
      // already where it belongs -- the cursors meet at the same index.
      %i2_index = arith.index_cast %after#0 : i64 to index
      scf.for %t = %i2_index to %n1_index step %c1 {
        %tt = arith.index_cast %t : index to i64
        %delta = arith.subi %tt, %after#0 : i64
        %dst = arith.addi %after#2, %delta : i64
        func.call @__ly_box_move_slot(%items, %dst, %scratch, %tt) : (memref<?xi64>, i64, memref<?xi64>, i64) -> ()
      }
    }
    func.return
  }

  // ⭐ A STABLE MERGE SORT, where this was an insertion sort: `xs.sort()` on
  // 20,000 elements took 19.0 s against CPython 3.14's 43.5 ms, and doubling
  // the length quadrupled the time (4,000 -> 26 ms, 8,000 -> 101, 16,000 ->
  // 401). CPython's listsort is Timsort, and this is its shape without the
  // merge stack or galloping: minrun-sized blocks made sorted by natural run
  // plus insertion, then balanced bottom-up merge passes.
  //
  // Why the output needs nothing more than stability to agree with CPython: a
  // stable sort of a total order has exactly one answer, so Timsort and this
  // produce the same list. What Timsort's stack and galloping buy is speed on
  // shapes this does not exploit -- long pre-sorted runs that are not block
  // aligned, and merges where one side is far longer than the other.
  func.func private @__ly_sort_slots(%items: memref<?xi64>, %len: i64) {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %c16_i64 = func.call @__ly_box_word_count() : () -> i64
    %c0 = arith.constant 0 : index
    %items_idx = memref.extract_aligned_pointer_as_index %items : memref<?xi64> -> index
    %items_i64 = arith.index_cast %items_idx : index to i64
    %items_ptr = llvm.inttoptr %items_i64 : i64 to !llvm.ptr
    %sortable = arith.cmpi sge, %len, %two : i64
    scf.if %sortable {
      %minrun = func.call @__ly_sort_minrun(%len) : (i64) -> i64
      %len_index = arith.index_cast %len : i64 to index
      %minrun_index = arith.index_cast %minrun : i64 to index
      scf.for %lo = %c0 to %len_index step %minrun_index {
        %lo_i64 = arith.index_cast %lo : index to i64
        %want = arith.addi %lo_i64, %minrun : i64
        %past = arith.cmpi sgt, %want, %len : i64
        %hi = arith.select %past, %len, %want : i64
        func.call @__ly_sort_block(%items, %items_ptr, %lo_i64, %hi) : (memref<?xi64>, !llvm.ptr, i64, i64) -> ()
      }
      %needs_merge = arith.cmpi slt, %minrun, %len : i64
      scf.if %needs_merge {
        %scratch_words = arith.muli %len, %c16_i64 : i64
        %scratch_index = arith.index_cast %scratch_words : i64 to index
        %scratch = memref.alloc(%scratch_index) : memref<?xi64>
        %decisions = memref.alloc(%len_index) : memref<?xi8>
        %final = scf.while (%width = %minrun) : (i64) -> i64 {
          %more = arith.cmpi slt, %width, %len : i64
          scf.condition(%more) %width : i64
        } do {
        ^bb0(%width: i64):
          %step_i64 = arith.muli %width, %two : i64
          %step_index = arith.index_cast %step_i64 : i64 to index
          scf.for %lo = %c0 to %len_index step %step_index {
            %lo_i64 = arith.index_cast %lo : index to i64
            %mid_want = arith.addi %lo_i64, %width : i64
            %mid_past = arith.cmpi sgt, %mid_want, %len : i64
            %mid = arith.select %mid_past, %len, %mid_want : i64
            %hi_want = arith.addi %lo_i64, %step_i64 : i64
            %hi_past = arith.cmpi sgt, %hi_want, %len : i64
            %hi = arith.select %hi_past, %len, %hi_want : i64
            %has_right = arith.cmpi slt, %mid, %hi : i64
            scf.if %has_right {
              func.call @__ly_merge_slots(%items, %items_ptr, %scratch, %decisions, %lo_i64, %mid, %hi) : (memref<?xi64>, !llvm.ptr, memref<?xi64>, memref<?xi8>, i64, i64, i64) -> ()
            }
          }
          scf.yield %step_i64 : i64
        }
        %scratch_addr_index = memref.extract_aligned_pointer_as_index %scratch : memref<?xi64> -> index
        %scratch_addr = arith.index_cast %scratch_addr_index : index to i64
        %decisions_addr_index = memref.extract_aligned_pointer_as_index %decisions : memref<?xi8> -> index
        %decisions_addr = arith.index_cast %decisions_addr_index : index to i64
        func.call @free_raw_i64_ptr(%decisions_addr) : (i64) -> ()
        func.call @free_raw_i64_ptr(%scratch_addr) : (i64) -> ()
      }
    }
    func.return
  }

  func.func @LyList_Sort(%self: memref<5xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "builtins.list", ly.runtime.method = "sort"} {
    %length_slot = arith.constant 2 : index
    %len = memref.load %self[%length_slot] : memref<5xi64>
    %items = func.call @__ly_list_items(%self) : (memref<5xi64>) -> memref<?xi64>
    func.call @__ly_sort_slots(%items, %len) : (memref<?xi64>, i64) -> ()
    func.return
  }

  func.func @LyList_Reverse(%self: memref<5xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "builtins.list", ly.runtime.method = "reverse"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %length_slot = arith.constant 2 : index
    %len = memref.load %self[%length_slot] : memref<5xi64>
    %items = func.call @__ly_list_items(%self) : (memref<5xi64>) -> memref<?xi64>
    %len_index = arith.index_cast %len : i64 to index
    %half = arith.divui %len_index, %c2 : index
    scf.for %i = %c0 to %half step %c1 {
      %last = arith.subi %len_index, %c1 : index
      %mirror = arith.subi %last, %i : index
      func.call @__ly_swap_slots(%items, %i, %mirror) : (memref<?xi64>, index, index) -> ()
    }
    func.return
  }

  // The six list comparisons read length from handle word 2 and the items
  // array through the handle, then share tuple's element loops.
  func.func private @__ly_list_compare(%lhs: memref<5xi64>, %rhs: memref<5xi64>) -> i64 {
    %length_slot = arith.constant 2 : index
    %llen = memref.load %lhs[%length_slot] : memref<5xi64>
    %rlen = memref.load %rhs[%length_slot] : memref<5xi64>
    %li = func.call @__ly_list_items(%lhs) : (memref<5xi64>) -> memref<?xi64>
    %ri = func.call @__ly_list_items(%rhs) : (memref<5xi64>) -> memref<?xi64>
    %cmp = func.call @__ly_sequence_compare_lens(%llen, %li, %rlen, %ri) : (i64, memref<?xi64>, i64, memref<?xi64>) -> i64
    func.return %cmp : i64
  }

  func.func @LyList_EqBool(%lhs: memref<5xi64> {ly.ownership.object_header}, %rhs: memref<5xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.list", ly.runtime.method = "__eq__"} {
    %length_slot = arith.constant 2 : index
    %llen = memref.load %lhs[%length_slot] : memref<5xi64>
    %rlen = memref.load %rhs[%length_slot] : memref<5xi64>
    %li = func.call @__ly_list_items(%lhs) : (memref<5xi64>) -> memref<?xi64>
    %ri = func.call @__ly_list_items(%rhs) : (memref<5xi64>) -> memref<?xi64>
    %eq = func.call @__ly_sequence_equal_lens(%llen, %li, %rlen, %ri) : (i64, memref<?xi64>, i64, memref<?xi64>) -> i1
    func.return %eq : i1
  }

  func.func @LyList_NeBool(%lhs: memref<5xi64> {ly.ownership.object_header}, %rhs: memref<5xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.list", ly.runtime.method = "__ne__"} {
    %eq = func.call @LyList_EqBool(%lhs, %rhs) : (memref<5xi64>, memref<5xi64>) -> i1
    %true = arith.constant true
    %ne = arith.xori %eq, %true : i1
    func.return %ne : i1
  }

  func.func @LyList_LtBool(%lhs: memref<5xi64> {ly.ownership.object_header}, %rhs: memref<5xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.list", ly.runtime.method = "__lt__"} {
    %zero = arith.constant 0 : i64
    %cmp = func.call @__ly_list_compare(%lhs, %rhs) : (memref<5xi64>, memref<5xi64>) -> i64
    %lt = arith.cmpi slt, %cmp, %zero : i64
    func.return %lt : i1
  }

  func.func @LyList_LeBool(%lhs: memref<5xi64> {ly.ownership.object_header}, %rhs: memref<5xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.list", ly.runtime.method = "__le__"} {
    %one = arith.constant 1 : i64
    %cmp = func.call @__ly_list_compare(%lhs, %rhs) : (memref<5xi64>, memref<5xi64>) -> i64
    %le = arith.cmpi slt, %cmp, %one : i64
    func.return %le : i1
  }

  func.func @LyList_GtBool(%lhs: memref<5xi64> {ly.ownership.object_header}, %rhs: memref<5xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.list", ly.runtime.method = "__gt__"} {
    %zero = arith.constant 0 : i64
    %cmp = func.call @__ly_list_compare(%lhs, %rhs) : (memref<5xi64>, memref<5xi64>) -> i64
    %gt = arith.cmpi sgt, %cmp, %zero : i64
    func.return %gt : i1
  }

  func.func @LyList_GeBool(%lhs: memref<5xi64> {ly.ownership.object_header}, %rhs: memref<5xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.list", ly.runtime.method = "__ge__"} {
    %minus_one = arith.constant -1 : i64
    %cmp = func.call @__ly_list_compare(%lhs, %rhs) : (memref<5xi64>, memref<5xi64>) -> i64
    %ge = arith.cmpi sgt, %cmp, %minus_one : i64
    func.return %ge : i1
  }

  func.func @LyList_Copy(%self: memref<5xi64> {ly.ownership.object_header}) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.list", ly.runtime.method = "copy", ly.runtime.result_contract = "builtins.list"} {
    %length_slot = arith.constant 2 : index
    %len = memref.load %self[%length_slot] : memref<5xi64>
    %items = func.call @__ly_list_items(%self) : (memref<5xi64>) -> memref<?xi64>
    %copy = func.call @__ly_list_copy_alloc(%len, %items) : (i64, memref<?xi64>) -> memref<5xi64>
    func.return %copy : memref<5xi64>
  }

  func.func @LyList_Concat(%lhs: memref<5xi64> {ly.ownership.object_header}, %rhs: memref<5xi64> {ly.ownership.object_header}) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.list", ly.runtime.method = "__add__", ly.runtime.result_contract = "builtins.list"} {
    %length_slot = arith.constant 2 : index
    %llen = memref.load %lhs[%length_slot] : memref<5xi64>
    %rlen = memref.load %rhs[%length_slot] : memref<5xi64>
    %li = func.call @__ly_list_items(%lhs) : (memref<5xi64>) -> memref<?xi64>
    %ri = func.call @__ly_list_items(%rhs) : (memref<5xi64>) -> memref<?xi64>
    %result = func.call @__ly_list_concat_alloc(%llen, %li, %rlen, %ri) : (i64, memref<?xi64>, i64, memref<?xi64>) -> memref<5xi64>
    func.return %result : memref<5xi64>
  }

  func.func @LyList_Repeat(%self: memref<5xi64> {ly.ownership.object_header}, %nh: memref<2xi64> {ly.ownership.object_header}) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.list", ly.runtime.method = "__mul__", ly.runtime.result_contract = "builtins.list"} {
    %nm, %nd = func.call @__ly_long_parts(%nh) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %length_slot = arith.constant 2 : index
    %len = memref.load %self[%length_slot] : memref<5xi64>
    %items = func.call @__ly_list_items(%self) : (memref<5xi64>) -> memref<?xi64>
    %n = func.call @__ly_seq_repeat_count(%nm, %nd) : (memref<2xi64>, memref<?xi32>) -> i64
    %result = func.call @__ly_list_repeat_alloc(%len, %items, %n) : (i64, memref<?xi64>, i64) -> memref<5xi64>
    func.return %result : memref<5xi64>
  }

  // NOTE(wave15 integration): the iter track's rebind-form LyList_SetItemBox
  // (caller-normalized index, transfer/owned result triple) was dropped in
  // favor of the closure track's in-place setitem_box below - the in-place
  // form normalizes and bounds-checks inside the native and stays sound on
  // borrowed receivers (closure captures, parameters), which the rebind
  // convention cannot express.

  // list.clear: release every element, zero the slot words, len = 0.
  func.func @LyList_Clear(%self: memref<5xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "builtins.list", ly.runtime.method = "clear"} {
    %zero = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16_words = func.call @__ly_box_word_count() : () -> i64
    %c16 = arith.index_cast %c16_words : i64 to index
    %length_slot = arith.constant 2 : index
    %len = memref.load %self[%length_slot] : memref<5xi64>
    %items = func.call @__ly_list_items(%self) : (memref<5xi64>) -> memref<?xi64>
    %len_index = arith.index_cast %len : i64 to index
    scf.for %i = %c0 to %len_index step %c1 {
      %ii = arith.index_cast %i : index to i64
      func.call @LyObject_ReleaseBoxedPayloadArraySlotRaw(%items, %ii) : (memref<?xi64>, i64) -> ()
    }
    %total = arith.muli %len_index, %c16 : index
    scf.for %w = %c0 to %total step %c1 {
      memref.store %zero, %items[%w] : memref<?xi64>
    }
    memref.store %zero, %self[%length_slot] : memref<5xi64>
    func.return
  }

  // list.extend(other): copy other's slot words past the current end and retain
  // each copied slot. Void and non-transferring: ensure_capacity may reallocate
  // the items array, but it publishes the new base THROUGH the handle, so there
  // is nothing to hand back.
  //
  // Both views are derived AFTER the growth on purpose. `xs.extend(xs)` aliases
  // the two, and the pre-growth array is freed by the growth -- the three-lane
  // form read the source through a lane captured before it, which is a
  // use-after-free the one-lane form cannot spell.
  func.func @LyList_ExtendM(%self: memref<5xi64> {ly.ownership.object_header}, %other: memref<5xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "builtins.list", ly.runtime.method = "extend"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16_words = func.call @__ly_box_word_count() : () -> i64
    %c16 = arith.index_cast %c16_words : i64 to index
    %length_slot = arith.constant 2 : index
    %len = memref.load %self[%length_slot] : memref<5xi64>
    %olen = memref.load %other[%length_slot] : memref<5xi64>
    %required = arith.addi %len, %olen : i64
    func.call @LyList_EnsureCapacity(%self, %required) : (memref<5xi64>, i64) -> ()
    %items = func.call @__ly_list_items(%self) : (memref<5xi64>) -> memref<?xi64>
    %oitems = func.call @__ly_list_items(%other) : (memref<5xi64>) -> memref<?xi64>
    %olen_index = arith.index_cast %olen : i64 to index
    %len_index = arith.index_cast %len : i64 to index
    scf.for %i = %c0 to %olen_index step %c1 {
      %src_base = arith.muli %i, %c16 : index
      %dst_slot = arith.addi %len_index, %i : index
      %dst_base = arith.muli %dst_slot, %c16 : index
      scf.for %w = %c0 to %c16 step %c1 {
        %src_index = arith.addi %src_base, %w : index
        %dst_index = arith.addi %dst_base, %w : index
        %word = memref.load %oitems[%src_index] : memref<?xi64>
        memref.store %word, %items[%dst_index] : memref<?xi64>
      }
      %dst64 = arith.index_cast %dst_slot : index to i64
      func.call @LyObject_RetainBoxedPayloadArraySlotRaw(%items, %dst64) : (memref<?xi64>, i64) -> ()
    }
    memref.store %required, %self[%length_slot] : memref<5xi64>
    func.return
  }

  // Probe a box against the list's slots. Split out because five list entry
  // points want it and each would otherwise re-spell the handle reads.
  func.func private @__ly_list_find_box(%self: memref<5xi64>, %elem_box: memref<5xi64>) -> i64 {
    %length_slot = arith.constant 2 : index
    %box_idx = memref.extract_aligned_pointer_as_index %elem_box : memref<5xi64> -> index
    %box_i64 = arith.index_cast %box_idx : index to i64
    %box_ptr = llvm.inttoptr %box_i64 : i64 to !llvm.ptr
    %len = memref.load %self[%length_slot] : memref<5xi64>
    %items = func.call @__ly_list_items(%self) : (memref<5xi64>) -> memref<?xi64>
    %found = func.call @__ly_sequence_find_lens(%len, %items, %box_ptr) : (i64, memref<?xi64>, !llvm.ptr) -> i64
    func.return %found : i64
  }

  func.func @LyList_ContainsBox(%self: memref<5xi64> {ly.ownership.object_header}, %elem_box: memref<5xi64>) -> i1 attributes {ly.runtime.contract = "builtins.list", ly.runtime.primitive = "contains_box"} {
    %minus_one = arith.constant -1 : i64
    %found = func.call @__ly_list_find_box(%self, %elem_box) : (memref<5xi64>, memref<5xi64>) -> i64
    %result = arith.cmpi ne, %found, %minus_one : i64
    func.return %result : i1
  }

  func.func @LyList_CountBox(%self: memref<5xi64> {ly.ownership.object_header}, %elem_box: memref<5xi64>) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.list", ly.runtime.primitive = "count_box", ly.runtime.result_contract = "builtins.int"} {
    %length_slot = arith.constant 2 : index
    %box_idx = memref.extract_aligned_pointer_as_index %elem_box : memref<5xi64> -> index
    %box_i64 = arith.index_cast %box_idx : index to i64
    %box_ptr = llvm.inttoptr %box_i64 : i64 to !llvm.ptr
    %len = memref.load %self[%length_slot] : memref<5xi64>
    %items = func.call @__ly_list_items(%self) : (memref<5xi64>) -> memref<?xi64>
    %count = func.call @__ly_sequence_count_lens(%len, %items, %box_ptr) : (i64, memref<?xi64>, !llvm.ptr) -> i64
    %h = func.call @LyLong_FromI64(%count) : (i64) -> memref<2xi64>
    func.return %h : memref<2xi64>
  }

  // "list.index(x): x not in list"
  memref.global "private" constant @__ly_list_msg_index_missing : memref<28xi8> = dense<[108, 105, 115, 116, 46, 105, 110, 100, 101, 120, 40, 120, 41, 58, 32, 120, 32, 110, 111, 116, 32, 105, 110, 32, 108, 105, 115, 116]>

  func.func @LyList_IndexBox(%self: memref<5xi64> {ly.ownership.object_header}, %elem_box: memref<5xi64>) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.list", ly.runtime.primitive = "index_box", ly.runtime.result_contract = "builtins.int"} {
    %minus_one = arith.constant -1 : i64
    %found = func.call @__ly_list_find_box(%self, %elem_box) : (memref<5xi64>, memref<5xi64>) -> i64
    %missing = arith.cmpi eq, %found, %minus_one : i64
    scf.if %missing {
      // The message is a fixed string, not `repr(x) is not in list`: CPython
      // 3.14 no longer interpolates the probe here, and a fixed string also
      // keeps this path off the boxed-__repr__ dispatch.
      %value_error = arith.constant 53 : i64
      %msg_static = memref.get_global @__ly_list_msg_index_missing : memref<28xi8>
      %msg = memref.cast %msg_static : memref<28xi8> to memref<?xi8>
      %msg_len = arith.constant 28 : i64
      func.call @__ly_raise_static_message(%value_error, %msg, %msg_len) : (i64, memref<?xi8>, i64) -> ()
    }
    %h = func.call @LyLong_FromI64(%found) : (i64) -> memref<2xi64>
    func.return %h : memref<2xi64>
  }

  // "list.remove(x): x not in list"
  memref.global "private" constant @__ly_list_msg_remove_missing : memref<29xi8> = dense<[108, 105, 115, 116, 46, 114, 101, 109, 111, 118, 101, 40, 120, 41, 58, 32, 120, 32, 110, 111, 116, 32, 105, 110, 32, 108, 105, 115, 116]>

  func.func @LyList_RemoveBox(%self: memref<5xi64> {ly.ownership.object_header}, %elem_box: memref<5xi64>) attributes {ly.runtime.contract = "builtins.list", ly.runtime.primitive = "remove_box"} {
    %minus_one = arith.constant -1 : i64
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16_words = func.call @__ly_box_word_count() : () -> i64
    %c16 = arith.index_cast %c16_words : i64 to index
    %length_slot = arith.constant 2 : index
    %found = func.call @__ly_list_find_box(%self, %elem_box) : (memref<5xi64>, memref<5xi64>) -> i64
    %missing = arith.cmpi eq, %found, %minus_one : i64
    scf.if %missing {
      %value_error = arith.constant 53 : i64
      %msg_static = memref.get_global @__ly_list_msg_remove_missing : memref<29xi8>
      %msg = memref.cast %msg_static : memref<29xi8> to memref<?xi8>
      %msg_len = arith.constant 29 : i64
      func.call @__ly_raise_static_message(%value_error, %msg, %msg_len) : (i64, memref<?xi8>, i64) -> ()
    }
    %len = memref.load %self[%length_slot] : memref<5xi64>
    %items = func.call @__ly_list_items(%self) : (memref<5xi64>) -> memref<?xi64>
    func.call @LyObject_ReleaseBoxedPayloadArraySlotRaw(%items, %found) : (memref<?xi64>, i64) -> ()
    %slot_index = arith.index_cast %found : i64 to index
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
    memref.store %new_len, %self[%length_slot] : memref<5xi64>
    func.return
  }

  // "list assignment index out of range"
  memref.global "private" constant @__ly_list_msg_assign_range : memref<34xi8> = dense<[108, 105, 115, 116, 32, 97, 115, 115, 105, 103, 110, 109, 101, 110, 116, 32, 105, 110, 100, 101, 120, 32, 111, 117, 116, 32, 111, 102, 32, 114, 97, 110, 103, 101]>

  // Normalize a raw (possibly negative) index against the list length and
  // raise IndexError with CPython's assignment message when it falls outside
  // [0, len). Shared by the runtime-mode setitem/delitem entry points, which
  // exist because compile-time element evidence does not cross function
  // boundaries (closure captures, parameters): the payload state is the only
  // authority there.
  func.func private @__ly_list_normalize_assign_index(%len: i64, %raw_index: i64) -> i64 {
    %zero = arith.constant 0 : i64
    %is_neg = arith.cmpi slt, %raw_index, %zero : i64
    %adjusted = arith.addi %raw_index, %len : i64
    %normalized = arith.select %is_neg, %adjusted, %raw_index : i1, i64
    %lower_ok = arith.cmpi sge, %normalized, %zero : i64
    %upper_ok = arith.cmpi slt, %normalized, %len : i64
    %in_range = arith.andi %lower_ok, %upper_ok : i1
    scf.if %in_range {
    } else {
      %index_error = arith.constant 55 : i64
      %msg_static = memref.get_global @__ly_list_msg_assign_range : memref<34xi8>
      %msg = memref.cast %msg_static : memref<34xi8> to memref<?xi8>
      %msg_len = arith.constant 34 : i64
      func.call @__ly_raise_static_message(%index_error, %msg, %msg_len) : (i64, memref<?xi8>, i64) -> ()
    }
    func.return %normalized : i64
  }

  // ⭐ THE OUT-OF-RANGE PATH RELEASES THE VALUE BOX before it raises, which is
  // why this does not just call `__ly_list_normalize_assign_index`. The caller
  // retained the value for the slot and handed this function a box that owns that
  // reference; raising from inside the normalizer left it with nowhere to go:
  //
  //     xs: list[int] = []
  //     xs.append(1)
  //     try:
  //         xs[5] = 9
  //     except IndexError:
  //         pass
  //     # leaked 52 B, one allocation, per execution
  //
  // The delete path keeps the shared normalizer: it has no value box to own.
  func.func @LyList_SetItemBox(%self: memref<5xi64> {ly.ownership.object_header}, %raw_index: i64, %value_box: memref<5xi64>) attributes {ly.runtime.contract = "builtins.list", ly.runtime.primitive = "setitem_box"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16_words = func.call @__ly_box_word_count() : () -> i64
    %c16 = arith.index_cast %c16_words : i64 to index
    %length_slot = arith.constant 2 : index
    %len = memref.load %self[%length_slot] : memref<5xi64>
    %zero_i = arith.constant 0 : i64
    %is_neg = arith.cmpi slt, %raw_index, %zero_i : i64
    %adjusted = arith.addi %raw_index, %len : i64
    %normalized = arith.select %is_neg, %adjusted, %raw_index : i1, i64
    %lower_ok = arith.cmpi sge, %normalized, %zero_i : i64
    %upper_ok = arith.cmpi slt, %normalized, %len : i64
    %in_range = arith.andi %lower_ok, %upper_ok : i1
    scf.if %in_range {
    } else {
      func.call @LyObject_ReleaseBoxedPayloadRaw(%value_box) : (memref<5xi64>) -> ()
      %index_error = arith.constant 55 : i64
      %msg_static = memref.get_global @__ly_list_msg_assign_range : memref<34xi8>
      %msg = memref.cast %msg_static : memref<34xi8> to memref<?xi8>
      %msg_len = arith.constant 34 : i64
      func.call @__ly_raise_static_message(%index_error, %msg, %msg_len) : (i64, memref<?xi8>, i64) -> ()
    }
    %items = func.call @__ly_list_items(%self) : (memref<5xi64>) -> memref<?xi64>
    func.call @LyObject_ReleaseBoxedPayloadArraySlotRaw(%items, %normalized) : (memref<?xi64>, i64) -> ()
    %slot = arith.index_cast %normalized : i64 to index
    %base = arith.muli %slot, %c16 : index
    scf.for %w = %c0 to %c16 step %c1 {
      %word = memref.load %value_box[%w] : memref<5xi64>
      %dst = arith.addi %base, %w : index
      memref.store %word, %items[%dst] : memref<?xi64>
    }
    func.return
  }

  func.func @LyList_DelItemIndex(%self: memref<5xi64> {ly.ownership.object_header}, %raw_index: i64) attributes {ly.runtime.contract = "builtins.list", ly.runtime.primitive = "delitem_index"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16_words = func.call @__ly_box_word_count() : () -> i64
    %c16 = arith.index_cast %c16_words : i64 to index
    %length_slot = arith.constant 2 : index
    %len = memref.load %self[%length_slot] : memref<5xi64>
    %normalized = func.call @__ly_list_normalize_assign_index(%len, %raw_index) : (i64, i64) -> i64
    %items = func.call @__ly_list_items(%self) : (memref<5xi64>) -> memref<?xi64>
    func.call @LyObject_ReleaseBoxedPayloadArraySlotRaw(%items, %normalized) : (memref<?xi64>, i64) -> ()
    %slot_index = arith.index_cast %normalized : i64 to index
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
    memref.store %new_len, %self[%length_slot] : memref<5xi64>
    func.return
  }

  // "pop from empty list"
  memref.global "private" constant @__ly_list_msg_pop_empty : memref<19xi8> = dense<[112, 111, 112, 32, 102, 114, 111, 109, 32, 101, 109, 112, 116, 121, 32, 108, 105, 115, 116]>
  // "pop index out of range"
  memref.global "private" constant @__ly_list_msg_pop_range : memref<22xi8> = dense<[112, 111, 112, 32, 105, 110, 100, 101, 120, 32, 111, 117, 116, 32, 111, 102, 32, 114, 97, 110, 103, 101]>

  // list.pop support: remove the entry at `raw_index` WITHOUT releasing it; the
  // popped box is parked in the (now free) tail slot for the caller to read,
  // exactly like LyDict_PopSlot. Returns the park slot.
  //
  // Parking rather than returning the element by value: the element type is
  // `$T`, so a by-value signature would need one manifest overload per physical
  // lane count, while the box words are a single fixed-width shape.
  func.func @LyList_PopSlot(%self: memref<5xi64> {ly.ownership.object_header}, %raw_index: i64) -> i64 attributes {ly.runtime.contract = "builtins.list", ly.runtime.primitive = "pop_slot"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16_words = func.call @__ly_box_word_count() : () -> i64
    %c16 = arith.index_cast %c16_words : i64 to index
    %index_error = arith.constant 55 : i64
    %length_slot = arith.constant 2 : index
    %len = memref.load %self[%length_slot] : memref<5xi64>
    %items = func.call @__ly_list_items(%self) : (memref<5xi64>) -> memref<?xi64>
    %empty = arith.cmpi sle, %len, %zero : i64
    scf.if %empty {
      %msg_static = memref.get_global @__ly_list_msg_pop_empty : memref<19xi8>
      %msg = memref.cast %msg_static : memref<19xi8> to memref<?xi8>
      %msg_len = arith.constant 19 : i64
      func.call @__ly_raise_static_message(%index_error, %msg, %msg_len) : (i64, memref<?xi8>, i64) -> ()
    }
    %is_neg = arith.cmpi slt, %raw_index, %zero : i64
    %adjusted = arith.addi %raw_index, %len : i64
    %normalized = arith.select %is_neg, %adjusted, %raw_index : i1, i64
    %lower_ok = arith.cmpi sge, %normalized, %zero : i64
    %upper_ok = arith.cmpi slt, %normalized, %len : i64
    %in_range = arith.andi %lower_ok, %upper_ok : i1
    scf.if %in_range {
    } else {
      %msg_static = memref.get_global @__ly_list_msg_pop_range : memref<22xi8>
      %msg = memref.cast %msg_static : memref<22xi8> to memref<?xi8>
      %msg_len = arith.constant 22 : i64
      func.call @__ly_raise_static_message(%index_error, %msg, %msg_len) : (i64, memref<?xi8>, i64) -> ()
    }
    %new_len = arith.subi %len, %one : i64
    %park = arith.index_cast %new_len : i64 to index
    %park_base = arith.muli %park, %c16 : index
    // Stage through scratch: the park slot lies inside the shifted range when
    // the popped index is not the last, so the shift would overwrite it.
    %scratch = memref.alloca() : memref<5xi64>
    %slot_index = arith.index_cast %normalized : i64 to index
    %slot_base = arith.muli %slot_index, %c16 : index
    scf.for %w = %c0 to %c16 step %c1 {
      %src = arith.addi %slot_base, %w : index
      %word = memref.load %items[%src] : memref<?xi64>
      memref.store %word, %scratch[%w] : memref<5xi64>
    }
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
    scf.for %w = %c0 to %c16 step %c1 {
      %dst = arith.addi %park_base, %w : index
      %word = memref.load %scratch[%w] : memref<5xi64>
      memref.store %word, %items[%dst] : memref<?xi64>
    }
    memref.store %new_len, %self[%length_slot] : memref<5xi64>
    func.return %new_len : i64
  }

  // Release the reference parked at an items-array slot (list.pop's caller side
  // runs this AFTER retaining the popped value into its own binding).
  func.func @LyList_ReleaseParked(%items: memref<?xi64>, %slot: i64) attributes {ly.runtime.contract = "builtins.list", ly.runtime.primitive = "release_parked"} {
    func.call @LyObject_ReleaseBoxedPayloadArraySlotRaw(%items, %slot) : (memref<?xi64>, i64) -> ()
    func.return
  }

  // list.insert: shift the tail up and store the boxed value. The caller must
  // have grown the payload to len+1 through `ensure_capacity` first; this entry
  // point only rearranges an already-sized array so it stays usable on the
  // rebound triple. CPython clamps the index instead of raising.
  func.func @LyList_InsertBox(%self: memref<5xi64> {ly.ownership.object_header}, %raw_index: i64, %value_box: memref<5xi64>) attributes {ly.runtime.contract = "builtins.list", ly.runtime.primitive = "insert_box"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16_words = func.call @__ly_box_word_count() : () -> i64
    %c16 = arith.index_cast %c16_words : i64 to index
    %length_slot = arith.constant 2 : index
    %len = memref.load %self[%length_slot] : memref<5xi64>
    %items = func.call @__ly_list_items(%self) : (memref<5xi64>) -> memref<?xi64>
    %is_neg = arith.cmpi slt, %raw_index, %zero : i64
    %adjusted = arith.addi %raw_index, %len : i64
    %from_end = arith.select %is_neg, %adjusted, %raw_index : i1, i64
    %below = arith.cmpi slt, %from_end, %zero : i64
    %clamped_low = arith.select %below, %zero, %from_end : i1, i64
    %above = arith.cmpi sgt, %clamped_low, %len : i64
    %at = arith.select %above, %len, %clamped_low : i1, i64
    %at_index = arith.index_cast %at : i64 to index
    %len_index = arith.index_cast %len : i64 to index
    // scf.for only counts up, so walk the shifted span in reverse by index
    // arithmetic: src = len-1-j, dst = len-j.
    %span = arith.subi %len_index, %at_index : index
    scf.for %j = %c0 to %span step %c1 {
      %offset = arith.addi %j, %c1 : index
      %dst_entry = arith.subi %len_index, %j : index
      %src_entry = arith.subi %len_index, %offset : index
      %src_base = arith.muli %src_entry, %c16 : index
      %dst_base = arith.muli %dst_entry, %c16 : index
      scf.for %w = %c0 to %c16 step %c1 {
        %src = arith.addi %src_base, %w : index
        %dst = arith.addi %dst_base, %w : index
        %word = memref.load %items[%src] : memref<?xi64>
        memref.store %word, %items[%dst] : memref<?xi64>
      }
    }
    %at_base = arith.muli %at_index, %c16 : index
    scf.for %w = %c0 to %c16 step %c1 {
      %word = memref.load %value_box[%w] : memref<5xi64>
      %dst = arith.addi %at_base, %w : index
      memref.store %word, %items[%dst] : memref<?xi64>
    }
    %new_len = arith.addi %len, %one : i64
    memref.store %new_len, %self[%length_slot] : memref<5xi64>
    func.return
  }

  func.func private @LyList_Shape() -> memref<5xi64> attributes {ly.runtime.contract = "builtins.list", ly.runtime.shape}

  // ===== builtins.list: one entity, one root =====
  //
  // The handle is `memref<5xi64>`:
  //
  //   word 0  refcount            word 3  capacity
  //   word 1  class id (10)       word 4  items base address
  //   word 2  length
  //
  // Words 0-4 are the prefix of the layout in
  // Passes/Runtime/ABI/ContainerLayout.h that a sequence uses.
  //
  // Why the items array is an ADDRESS in the handle and not a value beside it:
  // a growth writes the new address THROUGH the handle, so every holder
  // observes it with no further action and a mutation has nothing to rename.
  // That is what lets ensure_capacity / extend / __setslice__ / __delslice__
  // be void and non-transferring (rfc/memory-safety-proof.md, `Interior`).
  // CPython's PyList_New takes exactly `size` slots; the over-allocation lives
  // in list_resize, not here. This used to take max(length, 64), which with a
  // 16-word element box is 8 KB for every list however short.
  func.func private @__ly_list_alloc(%length: i64) -> memref<5xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.list"], ly.ownership.owned_results = [0]} {
    %one = arith.constant 1 : i64
    %handle_words = func.call @__ly_box_word_count() : () -> i64
    %class_id = arith.constant 10 : i64
    %zero = arith.constant 0 : i64
    // Before the handle is made, so a refusal leaves nothing behind.
    %capacity = arith.maxsi %length, %zero : i64
    %word_bytes = arith.constant 8 : i64
    %slot_bytes = arith.muli %handle_words, %word_bytes : i64
    %capacity_index = func.call @__ly_alloc_count(%capacity, %slot_bytes, %zero) : (i64, i64, i64) -> index
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %length_slot = arith.constant 2 : index
    %capacity_slot = arith.constant 3 : index
    %items_slot = arith.constant 4 : index
    %handle_bytes = arith.constant 40 : index
    %handle_block = memref.alloc(%handle_bytes) {alignment = 16 : i64} : memref<?xi8>
    %handle_at = arith.constant 0 : index
    %self = memref.view %handle_block[%handle_at][] {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<?xi8> to memref<5xi64>
    %handle_words_index = arith.index_cast %handle_words : i64 to index
    %payload_words_index = arith.muli %capacity_index, %handle_words_index : index
    // Plain memref.alloc with no alignment attribute is a bare malloc, so the
    // aligned pointer IS the allocated pointer and free_raw_i64_ptr can
    // release it later (same convention as __ly_dict_alloc).
    %items = memref.alloc(%payload_words_index) : memref<?xi64>
    %items_index = memref.extract_aligned_pointer_as_index %items : memref<?xi64> -> index
    %items_word = arith.index_cast %items_index : index to i64

    memref.store %one, %self[%refcount_slot] : memref<5xi64>
    memref.store %class_id, %self[%layout_slot] : memref<5xi64>
    memref.store %length, %self[%length_slot] : memref<5xi64>
    memref.store %capacity, %self[%capacity_slot] : memref<5xi64>
    memref.store %items_word, %self[%items_slot] : memref<5xi64>
    func.return %self : memref<5xi64>
  }

  // Borrowed view of the items array, derived at the point of use. The view's
  // SSA name is not an identity: identity is the handle, so a view taken after
  // a growth and one taken before it name the same slot of the same entity.
  // Marked ly.runtime.interior_word so release placement pins the handle
  // across the view's uses; a plain private helper would leave the ownership
  // walk nothing to follow from the call.
  func.func private @__ly_list_items(%self: memref<5xi64>) -> memref<?xi64> attributes {ly.runtime.contract = "builtins.list", ly.runtime.interior_word, ly.runtime.primitive = "items_view"} {
    %capacity_slot = arith.constant 3 : index
    %items_slot = arith.constant 4 : index
    %handle_words = func.call @__ly_box_word_count() : () -> i64
    %capacity = memref.load %self[%capacity_slot] : memref<5xi64>
    %words = arith.muli %capacity, %handle_words : i64
    %base = memref.load %self[%items_slot] : memref<5xi64>
    %view = func.call @__ly_global_view_i64(%base, %words) : (i64, i64) -> memref<?xi64>
    func.return %view : memref<?xi64>
  }

  // The four allocating list producers. Each is the list-shaped half of a
  // `__ly_sequence_*` twin: identical fill loop, different destination.
  func.func private @__ly_list_copy_alloc(%src_len: i64, %src_items: memref<?xi64>) -> memref<5xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.list"], ly.ownership.owned_results = [0]} {
    %self = func.call @__ly_list_alloc(%src_len) : (i64) -> memref<5xi64>
    %items = func.call @__ly_list_items(%self) : (memref<5xi64>) -> memref<?xi64>
    func.call @__ly_seq_fill_copy(%items, %src_len, %src_items) : (memref<?xi64>, i64, memref<?xi64>) -> ()
    func.return %self : memref<5xi64>
  }

  func.func private @__ly_list_concat_alloc(%llen: i64, %li: memref<?xi64>, %rlen: i64, %ri: memref<?xi64>) -> memref<5xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.list"], ly.ownership.owned_results = [0]} {
    %total = arith.addi %llen, %rlen : i64
    %self = func.call @__ly_list_alloc(%total) : (i64) -> memref<5xi64>
    %items = func.call @__ly_list_items(%self) : (memref<5xi64>) -> memref<?xi64>
    func.call @__ly_seq_fill_concat(%items, %llen, %li, %rlen, %ri) : (memref<?xi64>, i64, memref<?xi64>, i64, memref<?xi64>) -> ()
    func.return %self : memref<5xi64>
  }

  func.func private @__ly_list_repeat_alloc(%len: i64, %li: memref<?xi64>, %n: i64) -> memref<5xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.list"], ly.ownership.owned_results = [0]} {
    // CPython's list_repeat: a length past the word, or past the allocator's
    // reach, is MemoryError.
    %overflows = func.call @__ly_repeat_overflows(%len, %n) : (i64, i64) -> i1
    scf.if %overflows {
      func.call @LyErr_NoMemory() : () -> ()
    }
    %total = arith.muli %len, %n : i64
    %handle_words = func.call @__ly_box_word_count() : () -> i64
    %word_bytes = arith.constant 8 : i64
    %slot_bytes = arith.muli %handle_words, %word_bytes : i64
    %no_prefix = arith.constant 0 : i64
    func.call @__ly_check_alloc_count(%total, %slot_bytes, %no_prefix) : (i64, i64, i64) -> ()
    %self = func.call @__ly_list_alloc(%total) : (i64) -> memref<5xi64>
    %items = func.call @__ly_list_items(%self) : (memref<5xi64>) -> memref<?xi64>
    func.call @__ly_seq_fill_repeat(%items, %len, %li, %n) : (memref<?xi64>, i64, memref<?xi64>, i64) -> ()
    func.return %self : memref<5xi64>
  }

  func.func private @__ly_list_slice_alloc(%count: i64, %start: i64, %step: i64, %src_items: memref<?xi64>) -> memref<5xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.list"], ly.ownership.owned_results = [0]} {
    %self = func.call @__ly_list_alloc(%count) : (i64) -> memref<5xi64>
    %items = func.call @__ly_list_items(%self) : (memref<5xi64>) -> memref<?xi64>
    func.call @__ly_seq_fill_slice(%items, %count, %start, %step, %src_items) : (memref<?xi64>, i64, i64, i64, memref<?xi64>) -> ()
    func.return %self : memref<5xi64>
  }

  func.func @LyList_FromLength(%length: i64 {ly.runtime.default_i64 = 0 : i64}) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 10 : i64, ly.runtime.contract = "builtins.list", ly.runtime.initializer = "__new__", ly.runtime.result_contract = "builtins.list"} {
    %self = func.call @__ly_list_alloc(%length) : (i64) -> memref<5xi64>
    func.return %self : memref<5xi64>
  }

  func.func @LyList_Len(%self: memref<5xi64> {ly.ownership.object_header}) -> i64 attributes {ly.runtime.contract = "builtins.list", ly.runtime.method = "__len__"} {
    %length_slot = arith.constant 2 : index
    %length = memref.load %self[%length_slot] : memref<5xi64>
    func.return %length : i64
  }

  func.func @LyList_GetSlice(%self: memref<5xi64> {ly.ownership.object_header}, %start_raw: i64 {ly.runtime.clip_i64}, %stop_raw: i64 {ly.runtime.clip_i64}, %step_raw: i64 {ly.runtime.clip_i64}, %mask: i64) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.list", ly.runtime.method = "__getslice__", ly.runtime.result_contract = "builtins.list"} {
    %length_slot = arith.constant 2 : index
    %len = memref.load %self[%length_slot] : memref<5xi64>
    %b:3 = func.call @__ly_seq_slice_bounds(%len, %start_raw, %stop_raw, %step_raw, %mask) : (i64, i64, i64, i64, i64) -> (i64, i64, i64)
    %items = func.call @__ly_list_items(%self) : (memref<5xi64>) -> memref<?xi64>
    %result = func.call @__ly_list_slice_alloc(%b#1, %b#0, %b#2, %items) : (i64, i64, i64, memref<?xi64>) -> memref<5xi64>
    func.return %result : memref<5xi64>
  }

  // xs[s] for a slice object: list_subscript's PySlice_Unpack, then the
  // slice `xs[a:b:c]` takes.
  func.func @LyList_SliceSubscript(%self: memref<5xi64> {ly.ownership.object_header}, %slice: memref<5xi64> {ly.ownership.object_header}) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.list", ly.runtime.method = "__getslice__", ly.runtime.result_contract = "builtins.list"} {
    %start, %stop, %step, %mask = func.call @__ly_slice_unpack(%slice) : (memref<5xi64>) -> (i64, i64, i64, i64)
    %result = func.call @LyList_GetSlice(%self, %start, %stop, %step, %mask) : (memref<5xi64>, i64, i64, i64, i64) -> memref<5xi64>
    func.return %result : memref<5xi64>
  }

  // list[i:j:k] = xs (CPython list_ass_subscript): step 1 splices with an
  // arbitrary-length replacement; any other step requires len(xs) equal to
  // the slice length and replaces per selected slot. Always rebuilds the
  // items array for the splice form -- the source may alias the target
  // (`a[1:3] = a`), and a fresh array with copy-then-release ordering stays
  // correct under overlap without direction analysis.
  // Void and non-transferring: the splice branch reallocates the items array
  // and publishes the new base and capacity THROUGH the handle, so the caller's
  // name for the list stays valid and there is no token to hand back.
  //
  // Both interior views are derived before the reallocation on purpose: the
  // splice reads the OLD array (and `a[1:3] = a` makes the source the same
  // array), so the reads must happen before the base word moves.
  // "attempt to assign sequence of size "
  memref.global "private" constant @__ly_list_msg_assign_prefix : memref<35xi8> = dense<[97, 116, 116, 101, 109, 112, 116, 32, 116, 111, 32, 97, 115, 115, 105, 103, 110, 32, 115, 101, 113, 117, 101, 110, 99, 101, 32, 111, 102, 32, 115, 105, 122, 101, 32]>

  func.func @LyList_SetSlice(%self: memref<5xi64> {ly.ownership.object_header}, %start_raw: i64 {ly.runtime.clip_i64}, %stop_raw: i64 {ly.runtime.clip_i64}, %step_raw: i64 {ly.runtime.clip_i64}, %mask: i64, %src: memref<5xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "builtins.list", ly.runtime.method = "__setslice__"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %handle_words = func.call @__ly_box_word_count() : () -> i64
    %length_slot = arith.constant 2 : index
    %capacity_slot = arith.constant 3 : index
    %items_slot = arith.constant 4 : index
    %step_zero = arith.cmpi eq, %step_raw, %zero : i64
    scf.if %step_zero {
      func.call @__ly_slice_raise_zero_step() : () -> ()
    }
    // The throw does not return; the substitute keeps the IR division-safe.
    %step = arith.select %step_zero, %one, %step_raw : i1, i64
    %len = memref.load %self[%length_slot] : memref<5xi64>
    %items = func.call @__ly_list_items(%self) : (memref<5xi64>) -> memref<?xi64>
    %src_items = func.call @__ly_list_items(%src) : (memref<5xi64>) -> memref<?xi64>
    %adj:2 = func.call @__ly_slice_adjust(%len, %start_raw, %stop_raw, %step, %mask) : (i64, i64, i64, i64, i64) -> (i64, i64)
    %src_len = memref.load %src[%length_slot] : memref<5xi64>
    %step_is_one = arith.cmpi eq, %step, %one : i64
    scf.if %step_is_one {
      // Splice: new = prefix ++ src ++ tail. Prefix/tail boxes MOVE (the old
      // array is dropped without releasing them); src boxes are copies and
      // retain; the replaced range releases from the OLD array after the
      // copies so self-assignment keeps every shared box alive.
      %removed = arith.subi %len, %adj#1 : i64
      %new_len = arith.addi %removed, %src_len : i64
      %min_capacity = arith.constant 64 : i64
      %below_min = arith.cmpi slt, %new_len, %min_capacity : i64
      %new_capacity = arith.select %below_min, %min_capacity, %new_len : i1, i64
      %new_words = arith.muli %new_capacity, %handle_words : i64
      %new_words_index = arith.index_cast %new_words : i64 to index
      %new_items = memref.alloc(%new_words_index) : memref<?xi64>
      %prefix_words64 = arith.muli %adj#0, %handle_words : i64
      %prefix_words = arith.index_cast %prefix_words64 : i64 to index
      scf.for %w = %c0 to %prefix_words step %c1 {
        %word = memref.load %items[%w] : memref<?xi64>
        memref.store %word, %new_items[%w] : memref<?xi64>
      }
      %src_count = arith.index_cast %src_len : i64 to index
      scf.for %k = %c0 to %src_count step %c1 {
        %k64 = arith.index_cast %k : index to i64
        %dst_slot64 = arith.addi %adj#0, %k64 : i64
        func.call @__ly_box_move_slot(%new_items, %dst_slot64, %src_items, %k64) : (memref<?xi64>, i64, memref<?xi64>, i64) -> ()
        func.call @LyObject_RetainBoxedPayloadArraySlotRaw(%new_items, %dst_slot64) : (memref<?xi64>, i64) -> ()
      }
      %tail_from = arith.addi %adj#0, %adj#1 : i64
      %tail_count64 = arith.subi %len, %tail_from : i64
      %tail_count = arith.index_cast %tail_count64 : i64 to index
      scf.for %k = %c0 to %tail_count step %c1 {
        %k64 = arith.index_cast %k : index to i64
        %src_slot = arith.addi %tail_from, %k64 : i64
        %dst_slot = arith.addi %adj#0, %src_len : i64
        %dst_slot_k = arith.addi %dst_slot, %k64 : i64
        func.call @__ly_box_move_slot(%new_items, %dst_slot_k, %items, %src_slot) : (memref<?xi64>, i64, memref<?xi64>, i64) -> ()
      }
      %replaced_count = arith.index_cast %adj#1 : i64 to index
      scf.for %k = %c0 to %replaced_count step %c1 {
        %k64 = arith.index_cast %k : index to i64
        %slot = arith.addi %adj#0, %k64 : i64
        func.call @LyObject_ReleaseBoxedPayloadArraySlotRaw(%items, %slot) : (memref<?xi64>, i64) -> ()
      }
      %old_items_word = memref.load %self[%items_slot] : memref<5xi64>
      %new_items_index = memref.extract_aligned_pointer_as_index %new_items : memref<?xi64> -> index
      %new_items_word = arith.index_cast %new_items_index : index to i64
      // Publish length, capacity and the new base together, then free the old
      // block: after these stores no view derived from the handle can reach the
      // old one, and before them no reader could reach the new one.
      memref.store %new_len, %self[%length_slot] : memref<5xi64>
      memref.store %new_capacity, %self[%capacity_slot] : memref<5xi64>
      memref.store %new_items_word, %self[%items_slot] : memref<5xi64>
      func.call @free_raw_i64_ptr(%old_items_word) : (i64) -> ()
    } else {
      // Extended slice: same-length per-slot replacement (CPython requires
      // len(xs) == slicelength and raises ValueError otherwise).
      %len_matches = arith.cmpi eq, %src_len, %adj#1 : i64
      scf.if %len_matches {
      } else {
        %sequence_prefix_static = memref.get_global @__ly_list_msg_assign_prefix : memref<35xi8>
        %sequence_prefix = memref.cast %sequence_prefix_static : memref<35xi8> to memref<?xi8>
        %sequence_prefix_len = arith.constant 35 : i64
        func.call @__ly_slice_raise_extended_mismatch(%sequence_prefix, %sequence_prefix_len, %src_len, %adj#1) : (memref<?xi8>, i64, i64, i64) -> ()
      }
      %count = arith.index_cast %adj#1 : i64 to index
      scf.for %k = %c0 to %count step %c1 {
        %k64 = arith.index_cast %k : index to i64
        %offset = arith.muli %k64, %step : i64
        %slot = arith.addi %adj#0, %offset : i64
        func.call @LyObject_ReleaseBoxedPayloadArraySlotRaw(%items, %slot) : (memref<?xi64>, i64) -> ()
        func.call @__ly_box_move_slot(%items, %slot, %src_items, %k64) : (memref<?xi64>, i64, memref<?xi64>, i64) -> ()
        func.call @LyObject_RetainBoxedPayloadArraySlotRaw(%items, %slot) : (memref<?xi64>, i64) -> ()
      }
    }
    func.return
  }

  // xs[s] = ys for a slice object (list_ass_subscript).
  func.func @LyList_SliceAssign(%self: memref<5xi64> {ly.ownership.object_header}, %slice: memref<5xi64> {ly.ownership.object_header}, %src: memref<5xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "builtins.list", ly.runtime.method = "__setslice__"} {
    %start, %stop, %step, %mask = func.call @__ly_slice_unpack(%slice) : (memref<5xi64>) -> (i64, i64, i64, i64)
    func.call @LyList_SetSlice(%self, %start, %stop, %step, %mask, %src) : (memref<5xi64>, i64, i64, i64, i64, memref<5xi64>) -> ()
    func.return
  }

  // del list[i:j:k] (CPython list_ass_subscript with NULL value): compact
  // the non-selected boxes into a fresh array, releasing the selected ones.
  // One general path covers step 1 and extended slices, either sign.
  func.func @LyList_DelSlice(%self: memref<5xi64> {ly.ownership.object_header}, %start_raw: i64 {ly.runtime.clip_i64}, %stop_raw: i64 {ly.runtime.clip_i64}, %step_raw: i64 {ly.runtime.clip_i64}, %mask: i64) attributes {ly.runtime.contract = "builtins.list", ly.runtime.method = "__delslice__"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %handle_words = func.call @__ly_box_word_count() : () -> i64
    %length_slot = arith.constant 2 : index
    %capacity_slot = arith.constant 3 : index
    %items_slot = arith.constant 4 : index
    %step_zero = arith.cmpi eq, %step_raw, %zero : i64
    scf.if %step_zero {
      func.call @__ly_slice_raise_zero_step() : () -> ()
    }
    // The throw does not return; the substitute keeps the IR division-safe.
    %step = arith.select %step_zero, %one, %step_raw : i1, i64
    %len = memref.load %self[%length_slot] : memref<5xi64>
    %items = func.call @__ly_list_items(%self) : (memref<5xi64>) -> memref<?xi64>
    %adj:2 = func.call @__ly_slice_adjust(%len, %start_raw, %stop_raw, %step, %mask) : (i64, i64, i64, i64, i64) -> (i64, i64)
    // Normalize the selected index set {start + k*step} to a low bound and
    // absolute stride so membership is one modulo test either direction.
    %neg_step = arith.cmpi slt, %step, %zero : i64
    %count_m1 = arith.subi %adj#1, %one : i64
    %span = arith.muli %count_m1, %step : i64
    %last = arith.addi %adj#0, %span : i64
    %lo = arith.select %neg_step, %last, %adj#0 : i1, i64
    %hi = arith.select %neg_step, %adj#0, %last : i1, i64
    %step_neg_abs = arith.subi %zero, %step : i64
    %step_abs = arith.select %neg_step, %step_neg_abs, %step : i1, i64
    %new_len = arith.subi %len, %adj#1 : i64
    %min_capacity = arith.constant 64 : i64
    %below_min = arith.cmpi slt, %new_len, %min_capacity : i64
    %new_capacity = arith.select %below_min, %min_capacity, %new_len : i1, i64
    %new_words = arith.muli %new_capacity, %handle_words : i64
    %new_words_index = arith.index_cast %new_words : i64 to index
    %new_items = memref.alloc(%new_words_index) : memref<?xi64>
    %len_index = arith.index_cast %len : i64 to index
    %has_selection = arith.cmpi sgt, %adj#1, %zero : i64
    %write0 = arith.constant 0 : i64
    %final_write = scf.for %i = %c0 to %len_index step %c1 iter_args(%write = %write0) -> (i64) {
      %i64v = arith.index_cast %i : index to i64
      %ge_lo = arith.cmpi sge, %i64v, %lo : i64
      %le_hi = arith.cmpi sle, %i64v, %hi : i64
      %in_range = arith.andi %ge_lo, %le_hi : i1
      %rel = arith.subi %i64v, %lo : i64
      %rem = arith.remsi %rel, %step_abs : i64
      %on_stride = arith.cmpi eq, %rem, %zero : i64
      %in_window = arith.andi %in_range, %on_stride : i1
      %selected_raw = arith.andi %in_window, %has_selection : i1
      %next = scf.if %selected_raw -> (i64) {
        func.call @LyObject_ReleaseBoxedPayloadArraySlotRaw(%items, %i64v) : (memref<?xi64>, i64) -> ()
        scf.yield %write : i64
      } else {
        func.call @__ly_box_move_slot(%new_items, %write, %items, %i64v) : (memref<?xi64>, i64, memref<?xi64>, i64) -> ()
        %bumped = arith.addi %write, %one : i64
        scf.yield %bumped : i64
      }
      scf.yield %next : i64
    }
    %old_items_word = memref.load %self[%items_slot] : memref<5xi64>
    %new_items_index = memref.extract_aligned_pointer_as_index %new_items : memref<?xi64> -> index
    %new_items_word = arith.index_cast %new_items_index : index to i64
    memref.store %final_write, %self[%length_slot] : memref<5xi64>
    memref.store %new_capacity, %self[%capacity_slot] : memref<5xi64>
    memref.store %new_items_word, %self[%items_slot] : memref<5xi64>
    func.call @free_raw_i64_ptr(%old_items_word) : (i64) -> ()
    func.return
  }

  // del xs[s] for a slice object (list_ass_subscript with no value).
  func.func @LyList_SliceDelete(%self: memref<5xi64> {ly.ownership.object_header}, %slice: memref<5xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "builtins.list", ly.runtime.method = "__delslice__"} {
    %start, %stop, %step, %mask = func.call @__ly_slice_unpack(%slice) : (memref<5xi64>) -> (i64, i64, i64, i64)
    func.call @LyList_DelSlice(%self, %start, %stop, %step, %mask) : (memref<5xi64>, i64, i64, i64, i64) -> ()
    func.return
  }

  // Grow the items array in place. Void and non-transferring for the same
  // reason as @LyDict_EnsureCapacity: the new base is written into the handle,
  // which every holder already names, so there is nothing to hand back and no
  // reference for a caller to re-acquire. The three-lane spelling had to
  // declare transfer_args = [0] + owned_results = [0] because the array VALUE
  // was part of the entity's identity.
  // CPython's list_resize growth: mild over-allocation padded to a multiple of
  // 4. The LOWERING computes the same expression (growCapacity in
  // Core/CollectionPayload.cpp) to decide when a store needs to grow at all, so
  // the two must agree exactly -- a capacity the lowering believes in and the
  // runtime did not allocate is a store past the array.
  func.func @LyList_EnsureCapacity(%self: memref<5xi64> {ly.ownership.object_header}, %required: i64) attributes {ly.runtime.contract = "builtins.list", ly.runtime.primitive = "ensure_capacity"} {
    %handle_words = func.call @__ly_box_word_count() : () -> i64
    %capacity_slot = arith.constant 3 : index
    %items_slot = arith.constant 4 : index
    %lower = arith.constant 0 : index
    %step = arith.constant 1 : index

    %capacity = memref.load %self[%capacity_slot] : memref<5xi64>
    %needs_grow = arith.cmpi slt, %capacity, %required : i64
    scf.if %needs_grow {
      // list_resize: (newsize + (newsize >> 3) + 6) & ~3
      %three = arith.constant 3 : i64
      %six = arith.constant 6 : i64
      %pad_mask = arith.constant -4 : i64
      %eighth = arith.shrsi %required, %three : i64
      %with_eighth = arith.addi %required, %eighth : i64
      %with_slack = arith.addi %with_eighth, %six : i64
      %new_capacity = arith.andi %with_slack, %pad_mask : i64
      // ⭐ REALLOC, not allocate-copy-free. CPython's list_resize hands the
      // block to PyMem_Realloc, which usually extends it where it lies; the
      // mild 1.125x growth above is only affordable because of that. Copying
      // instead made 20M appends take 5.0 s against 1.3 s for the doubling
      // this replaced -- the growth constant and the mechanism are one port,
      // not two.
      %word_bytes = arith.constant 8 : i64
      %slot_bytes = arith.muli %handle_words, %word_bytes : i64
      %none_before = arith.constant 0 : i64
      %checked = func.call @__ly_alloc_count(%new_capacity, %slot_bytes, %none_before) : (i64, i64, i64) -> index
      %old_items_word = memref.load %self[%items_slot] : memref<5xi64>
      %new_words = arith.muli %new_capacity, %handle_words : i64
      %new_bytes = arith.muli %new_words, %word_bytes : i64
      %new_items_word = func.call @realloc_raw_i64_ptr(%old_items_word, %new_bytes) : (i64, i64) -> i64
      // Publish capacity and the new base together; realloc already released
      // the old block if it moved.
      memref.store %new_capacity, %self[%capacity_slot] : memref<5xi64>
      memref.store %new_items_word, %self[%items_slot] : memref<5xi64>
    }
    func.return
  }

  func.func @LyList_DecRef(%self: memref<5xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.list", ly.runtime.deallocator} {
    %storage = memref.cast %self : memref<5xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    %length_slot = arith.constant 2 : index
    %items_slot = arith.constant 4 : index
    %lower = arith.constant 0 : index
    %step = arith.constant 1 : index
    %items = func.call @__ly_list_items(%self) : (memref<5xi64>) -> memref<?xi64>
    %length = memref.load %self[%length_slot] : memref<5xi64>
    %length_index = arith.index_cast %length : i64 to index
    scf.for %i = %lower to %length_index step %step {
      %logical_index = arith.index_cast %i : index to i64
      func.call @LyObject_ReleaseBoxedPayloadArraySlotRaw(%items, %logical_index) : (memref<?xi64>, i64) -> ()
    }
    %items_word = memref.load %self[%items_slot] : memref<5xi64>
    func.call @free_raw_i64_ptr(%items_word) : (i64) -> ()
    memref.dealloc %self : memref<5xi64>
    cf.br ^done

  ^done:
    func.return
  }

  // list.__repr__: `[e0, e1, ...]`, each element repr'd through the uniform
  // boxed-method hook. Intermediate strs are released explicitly (Concat
  // borrows its operands).
  func.func @LyList_Repr(%self: memref<5xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.list", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c16_i64 = func.call @__ly_box_word_count() : () -> i64
    %length_slot = arith.constant 2 : index
    %len = memref.load %self[%length_slot] : memref<5xi64>
    %items = func.call @__ly_list_items(%self) : (memref<5xi64>) -> memref<?xi64>
    %len_idx = arith.index_cast %len : i64 to index
    %items_idx = memref.extract_aligned_pointer_as_index %items : memref<?xi64> -> index
    %items_i64 = arith.index_cast %items_idx : index to i64
    %items_ptr = llvm.inttoptr %items_i64 : i64 to !llvm.ptr

    %open_ref = memref.get_global @__ly_repr_lbracket : memref<1xi8>
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

    %close_ref = memref.get_global @__ly_repr_rbracket : memref<1xi8>
    %close_dyn = memref.cast %close_ref : memref<1xi8> to memref<?xi8>
    %clh, %clb = func.call @__ly_unicode_from_valid_utf8(%close_dyn, %c0, %c1_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %out_h, %out_b = func.call @LyUnicode_Concat(%loop#0, %loop#1, %clh, %clb) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.call @LyUnicode_DecRef(%loop#0) : (memref<2xi64>) -> ()
    func.call @LyUnicode_DecRef(%clh) : (memref<2xi64>) -> ()
    func.return %out_h, %out_b : memref<2xi64>, memref<?xi8>
  }
}
