// What the sequence types share -- CPython's Objects/abstract.c: whether a
// repeat count fits (PyNumber_AsSsize_t with OverflowError, and the overflow
// test every repeat makes), and the copy / concat / repeat / slice fills and
// element-wise compare / find / count that list and tuple run over their
// slots.

module {
  // ===== declared here, defined in another runtime file or built by the lowering =====
  func.func private @LyObject_RetainBoxedPayloadArraySlotRaw(%payload: memref<?xi64>, %logical_index: i64)
  func.func private @__ly_box_equal(%lhs: !llvm.ptr, %rhs: !llvm.ptr) -> i1
  func.func private @__ly_box_order(%lhs: !llvm.ptr, %rhs: !llvm.ptr, %op: i64) -> i1
  func.func private @__ly_box_word_count() -> i64
  func.func private @__ly_handle_retain_raw(%entity: i64)
  func.func private @__ly_long_operand_view(%meta: memref<2xi64>, %digits: memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
  func.func private @__ly_long_view_as_i64(%meta: memref<2xi64>, %digits: memref<?xi32>) -> i64
  func.func private @__ly_long_view_fits_i64(%meta: memref<2xi64>, %digits: memref<?xi32>) -> i1
  func.func private @__ly_raise_static_message(%class_id: i64, %message: memref<?xi8>, %length: i64)
  func.func private @__ly_slice_adjust(%len: i64, %start_in: i64, %stop_in: i64, %step: i64, %mask: i64) -> (i64, i64)
  func.func private @__ly_slice_raise_zero_step()

  // CPython's PyNumber_AsSsize_t(n, OverflowError), what a sequence repeat
  // counts with: an int past the word, of either sign.
  memref.global "private" constant @__ly_msg_index_overflow : memref<44xi8> = dense<[99, 97, 110, 110, 111, 116, 32, 102, 105, 116, 32, 39, 105, 110, 116, 39, 32, 105, 110, 116, 111, 32, 97, 110, 32, 105, 110, 100, 101, 120, 45, 115, 105, 122, 101, 100, 32, 105, 110, 116, 101, 103, 101, 114]>
  func.func private @__ly_raise_index_overflow() {
    %class_id = arith.constant {ly.class_id_of = "builtins.OverflowError"} 104 : i64
    %length = arith.constant 44 : i64
    %message_static = memref.get_global @__ly_msg_index_overflow : memref<44xi8>
    %message = memref.cast %message_static : memref<44xi8> to memref<?xi8>
    func.call @__ly_raise_static_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  // Whether `len` copies `n` times overflow the word (`n` and `len` not
  // negative), CPython's `len && n > PY_SSIZE_T_MAX / len`.
  func.func private @__ly_repeat_overflows(%len: i64, %n: i64) -> i1 {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %max = arith.constant 9223372036854775807 : i64
    %empty = arith.cmpi eq, %len, %zero : i64
    %divisor = arith.select %empty, %one, %len : i64
    %most = arith.divui %max, %divisor : i64
    %over = arith.cmpi ugt, %n, %most : i64
    %nonempty = arith.cmpi ne, %len, %zero : i64
    %overflows = arith.andi %nonempty, %over : i1
    func.return %overflows : i1
  }

  // tuple.__eq__: element-wise recursive equality over the boxed payloads.
  // Element-wise sequence equality core, length by value for the same reason
  // as @__ly_sequence_compare_op.
  func.func private @__ly_sequence_equal_lens(%lhs_len: i64, %lhs_items: memref<?xi64>, %rhs_len: i64, %rhs_items: memref<?xi64>) -> i1 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16_i64 = func.call @__ly_box_word_count() : () -> i64
    %false = arith.constant false
    %true = arith.constant true
    %same_len = arith.cmpi eq, %lhs_len, %rhs_len : i64
    %result = scf.if %same_len -> (i1) {
      %len_index = arith.index_cast %lhs_len : i64 to index
      %lhs_idx = memref.extract_aligned_pointer_as_index %lhs_items : memref<?xi64> -> index
      %rhs_idx = memref.extract_aligned_pointer_as_index %rhs_items : memref<?xi64> -> index
      %lhs_i64 = arith.index_cast %lhs_idx : index to i64
      %rhs_i64 = arith.index_cast %rhs_idx : index to i64
      %lhs_ptr = llvm.inttoptr %lhs_i64 : i64 to !llvm.ptr
      %rhs_ptr = llvm.inttoptr %rhs_i64 : i64 to !llvm.ptr
      %all = scf.for %i = %c0 to %len_index step %c1 iter_args(%ok = %true) -> (i1) {
        %next = scf.if %ok -> (i1) {
          %i_i64 = arith.index_cast %i : index to i64
          %off = arith.muli %i_i64, %c16_i64 : i64
          %lbox = llvm.getelementptr %lhs_ptr[%off] : (!llvm.ptr, i64) -> !llvm.ptr, i64
          %rbox = llvm.getelementptr %rhs_ptr[%off] : (!llvm.ptr, i64) -> !llvm.ptr, i64
          %eq = func.call @__ly_box_equal(%lbox, %rbox) : (!llvm.ptr, !llvm.ptr) -> i1
          scf.yield %eq : i1
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

  // Lexicographic sequence ordering, tuplerichcompare / list_richcompare: the
  // first index whose items are not equal decides, by comparing THOSE items
  // with `op` itself (CPython's numbers: Py_LT 0, Py_LE 1, Py_GT 4, Py_GE 5);
  // with none, the lengths do.
  //
  // ⛔ Not a -1/0/1 answer that each operator reads: "the items differ and
  // the left is not less" is not "the left is greater" -- `[nan] > [1.0]`
  // answered True where CPython answers False -- and the refusal has to name
  // the operator the program wrote.
  //
  // Takes the LENGTH by value rather than reading a `meta` lane, because the
  // length lives in a lane for a lane-carrying sequence (tuple, set,
  // frozenset) and in handle word 2 for a handle-fronted one (list). One core
  // plus a lane-shaped wrapper serves both; duplicating the loop per
  // representation is how the two copies drift.
  func.func private @__ly_sequence_compare_op(%lhs_len: i64, %lhs_items: memref<?xi64>, %rhs_len: i64, %rhs_items: memref<?xi64>, %op: i64) -> i1 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16_i64 = func.call @__ly_box_word_count() : () -> i64
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %four = arith.constant 4 : i64
    %lhs_shorter = arith.cmpi slt, %lhs_len, %rhs_len : i64
    %common = arith.select %lhs_shorter, %lhs_len, %rhs_len : i1, i64
    %lhs_idx = memref.extract_aligned_pointer_as_index %lhs_items : memref<?xi64> -> index
    %rhs_idx = memref.extract_aligned_pointer_as_index %rhs_items : memref<?xi64> -> index
    %lhs_i64 = arith.index_cast %lhs_idx : index to i64
    %rhs_i64 = arith.index_cast %rhs_idx : index to i64
    %lhs_ptr = llvm.inttoptr %lhs_i64 : i64 to !llvm.ptr
    %rhs_ptr = llvm.inttoptr %rhs_i64 : i64 to !llvm.ptr
    // The first index whose items differ, or `common`.
    %differs = scf.while (%i = %zero) : (i64) -> i64 {
      %inside = arith.cmpi slt, %i, %common : i64
      %same = scf.if %inside -> (i1) {
        %off = arith.muli %i, %c16_i64 : i64
        %lbox = llvm.getelementptr %lhs_ptr[%off] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        %rbox = llvm.getelementptr %rhs_ptr[%off] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        %eq = func.call @__ly_box_equal(%lbox, %rbox) : (!llvm.ptr, !llvm.ptr) -> i1
        scf.yield %eq : i1
      } else {
        %no = arith.constant false
        scf.yield %no : i1
      }
      scf.condition(%same) %i : i64
    } do {
    ^bb0(%i: i64):
      %next = arith.addi %i, %one : i64
      scf.yield %next : i64
    }
    %decided = arith.cmpi slt, %differs, %common : i64
    %result = scf.if %decided -> (i1) {
      %off = arith.muli %differs, %c16_i64 : i64
      %lbox = llvm.getelementptr %lhs_ptr[%off] : (!llvm.ptr, i64) -> !llvm.ptr, i64
      %rbox = llvm.getelementptr %rhs_ptr[%off] : (!llvm.ptr, i64) -> !llvm.ptr, i64
      %ordered = func.call @__ly_box_order(%lbox, %rbox, %op) : (!llvm.ptr, !llvm.ptr, i64) -> i1
      scf.yield %ordered : i1
    } else {
      %lt = arith.cmpi slt, %lhs_len, %rhs_len : i64
      %le = arith.cmpi sle, %lhs_len, %rhs_len : i64
      %gt = arith.cmpi sgt, %lhs_len, %rhs_len : i64
      %ge = arith.cmpi sge, %lhs_len, %rhs_len : i64
      %greater_side = arith.cmpi uge, %op, %four : i64
      %equal_bit = arith.andi %op, %one : i64
      %or_equal = arith.cmpi ne, %equal_bit, %zero : i64
      %less_pick = arith.select %or_equal, %le, %lt : i1
      %greater_pick = arith.select %or_equal, %ge, %gt : i1
      %pick = arith.select %greater_side, %greater_pick, %less_pick : i1
      scf.yield %pick : i1
    }
    func.return %result : i1
  }

  // Copy `src_len` element boxes into an already-sized destination array,
  // retaining every copied payload reference. One loop rather than one per
  // container, so it serves both a lane-carrying destination (tuple/set/
  // frozenset, whose array is a lane) and a handle-fronted one (list, whose
  // array is reached through handle word 4). It was split out of the shared
  // @__ly_sequence_copy_alloc, which the per-container forks have since
  // replaced; this is the part of it that did not need forking.
  func.func private @__ly_seq_fill_copy(%dst_items: memref<?xi64>, %src_len: i64, %src_items: memref<?xi64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16_words = func.call @__ly_box_word_count() : () -> i64
    %c16 = arith.index_cast %c16_words : i64 to index
    %c2_slot = arith.constant 0 : index
    %len_index = arith.index_cast %src_len : i64 to index
    scf.for %i = %c0 to %len_index step %c1 {
      %base = arith.muli %i, %c16 : index
      scf.for %w = %c0 to %c16 step %c1 {
        %slot = arith.addi %base, %w : index
        %word = memref.load %src_items[%slot] : memref<?xi64>
        memref.store %word, %dst_items[%slot] : memref<?xi64>
      }
      %entity_slot = arith.addi %base, %c2_slot : index
      %entity = memref.load %dst_items[%entity_slot] : memref<?xi64>
      func.call @__ly_handle_retain_raw(%entity) : (i64) -> ()
    }
    func.return
  }

  // Shared box-copy concatenation for list/tuple `+`. Parameterized by class id
  // rather than duplicated per contract: both have the same (header, meta,
  // items) shape and the same 16-word element boxes, so only the id stamped
  // into the fresh header differs.
  // Fill an already-sized destination with lhs ++ rhs, retaining each copied
  // slot. One loop pair for both representations, as with @__ly_seq_fill_copy.
  func.func private @__ly_seq_fill_concat(%dst_items: memref<?xi64>, %llen: i64, %li: memref<?xi64>, %rlen: i64, %ri: memref<?xi64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16_words = func.call @__ly_box_word_count() : () -> i64
    %c16 = arith.index_cast %c16_words : i64 to index
    %c2_slot = arith.constant 0 : index
    %llen_index = arith.index_cast %llen : i64 to index
    %rlen_index = arith.index_cast %rlen : i64 to index
    scf.for %i = %c0 to %llen_index step %c1 {
      %base = arith.muli %i, %c16 : index
      scf.for %w = %c0 to %c16 step %c1 {
        %slot = arith.addi %base, %w : index
        %word = memref.load %li[%slot] : memref<?xi64>
        memref.store %word, %dst_items[%slot] : memref<?xi64>
      }
      %entity_slot = arith.addi %base, %c2_slot : index
      %entity = memref.load %dst_items[%entity_slot] : memref<?xi64>
      func.call @__ly_handle_retain_raw(%entity) : (i64) -> ()
    }
    scf.for %i = %c0 to %rlen_index step %c1 {
      %dst_entry = arith.addi %llen_index, %i : index
      %src_base = arith.muli %i, %c16 : index
      %dst_base = arith.muli %dst_entry, %c16 : index
      scf.for %w = %c0 to %c16 step %c1 {
        %src = arith.addi %src_base, %w : index
        %dst = arith.addi %dst_base, %w : index
        %word = memref.load %ri[%src] : memref<?xi64>
        memref.store %word, %dst_items[%dst] : memref<?xi64>
      }
      %entity_slot = arith.addi %dst_base, %c2_slot : index
      %entity = memref.load %dst_items[%entity_slot] : memref<?xi64>
      func.call @__ly_handle_retain_raw(%entity) : (i64) -> ()
    }
    func.return
  }

  // Shared box-copy repetition for list/tuple `*`, parameterized by class id
  // for the same reason as __ly_sequence_concat.
  // Fill an already-sized destination with `n` copies of the source slots.
  func.func private @__ly_seq_fill_repeat(%dst_items: memref<?xi64>, %len: i64, %li: memref<?xi64>, %n: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16_words = func.call @__ly_box_word_count() : () -> i64
    %c16 = arith.index_cast %c16_words : i64 to index
    %c2_slot = arith.constant 0 : index
    %len_index = arith.index_cast %len : i64 to index
    // No trips over nothing. ⛔ Not `n` of them: `[] * 2**62` looped 2**62
    // times, where CPython answers an empty repeat without reading `n`.
    %no_items = arith.constant 0 : i64
    %empty = arith.cmpi eq, %len, %no_items : i64
    %trips = arith.select %empty, %no_items, %n : i64
    %n_index = arith.index_cast %trips : i64 to index
    scf.for %rep = %c0 to %n_index step %c1 {
      %rep_entry = arith.muli %rep, %len_index : index
      scf.for %i = %c0 to %len_index step %c1 {
        %dst_entry = arith.addi %rep_entry, %i : index
        %src_base = arith.muli %i, %c16 : index
        %dst_base = arith.muli %dst_entry, %c16 : index
        scf.for %w = %c0 to %c16 step %c1 {
          %src = arith.addi %src_base, %w : index
          %dst = arith.addi %dst_base, %w : index
          %word = memref.load %li[%src] : memref<?xi64>
          memref.store %word, %dst_items[%dst] : memref<?xi64>
        }
        %entity_slot = arith.addi %dst_base, %c2_slot : index
        %entity = memref.load %dst_items[%entity_slot] : memref<?xi64>
        func.call @__ly_handle_retain_raw(%entity) : (i64) -> ()
      }
    }
    func.return
  }

  // Clamp a boxed int repetition count to a non-negative i64, as CPython's
  // sequence repeat does; past the word it raises, either sign.
  // ⛔ Not the count's low word, which is what it read: `[0] * (2**64 + 1)`
  // was a one-element list and `[0] * 2**63` an empty one.
  func.func private @__ly_seq_repeat_count(%nm: memref<2xi64>, %nd: memref<?xi32>) -> i64 {
    %zero = arith.constant 0 : i64
    %meta_view, %digits_view = func.call @__ly_long_operand_view(%nm, %nd) : (memref<2xi64>, memref<?xi32>) -> (memref<2xi64>, memref<?xi32>)
    %fits = func.call @__ly_long_view_fits_i64(%meta_view, %digits_view) : (memref<2xi64>, memref<?xi32>) -> i1
    %true = arith.constant true
    %past_word = arith.xori %fits, %true : i1
    scf.if %past_word {
      func.call @__ly_raise_index_overflow() : () -> ()
    }
    %n_raw = func.call @__ly_long_view_as_i64(%meta_view, %digits_view) : (memref<2xi64>, memref<?xi32>) -> i64
    %n_neg = arith.cmpi slt, %n_raw, %zero : i64
    %n = arith.select %n_neg, %zero, %n_raw : i1, i64
    func.return %n : i64
  }

  // Linear membership scan (identity-or-equality per slot; CPython's list
  // and tuple `in` do not hash).
  func.func private @__ly_sequence_find_lens(%len: i64, %items: memref<?xi64>, %probe: !llvm.ptr) -> i64 {
    %minus_one = arith.constant -1 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16_i64 = func.call @__ly_box_word_count() : () -> i64
    %len_index = arith.index_cast %len : i64 to index
    %items_idx = memref.extract_aligned_pointer_as_index %items : memref<?xi64> -> index
    %items_i64 = arith.index_cast %items_idx : index to i64
    %items_ptr = llvm.inttoptr %items_i64 : i64 to !llvm.ptr
    %found = scf.for %i = %c0 to %len_index step %c1 iter_args(%acc = %minus_one) -> (i64) {
      %not_yet = arith.cmpi eq, %acc, %minus_one : i64
      %next = scf.if %not_yet -> (i64) {
        %ii = arith.index_cast %i : index to i64
        %off = arith.muli %ii, %c16_i64 : i64
        %entry = llvm.getelementptr %items_ptr[%off] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        %eq = func.call @__ly_box_equal(%entry, %probe) : (!llvm.ptr, !llvm.ptr) -> i1
        %sel = arith.select %eq, %ii, %minus_one : i1, i64
        scf.yield %sel : i64
      } else {
        scf.yield %acc : i64
      }
      scf.yield %next : i64
    }
    func.return %found : i64
  }

  // Occurrence count over the 16-word element boxes, shared by list and tuple
  // (`__ly_sequence_find_equal`'s counting twin).
  func.func private @__ly_sequence_count_lens(%len: i64, %items: memref<?xi64>, %probe: !llvm.ptr) -> i64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16_i64 = func.call @__ly_box_word_count() : () -> i64
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %len_index = arith.index_cast %len : i64 to index
    %items_idx = memref.extract_aligned_pointer_as_index %items : memref<?xi64> -> index
    %items_i64 = arith.index_cast %items_idx : index to i64
    %items_ptr = llvm.inttoptr %items_i64 : i64 to !llvm.ptr
    %count = scf.for %i = %c0 to %len_index step %c1 iter_args(%acc = %zero) -> (i64) {
      %ii = arith.index_cast %i : index to i64
      %off = arith.muli %ii, %c16_i64 : i64
      %entry = llvm.getelementptr %items_ptr[%off] : (!llvm.ptr, i64) -> !llvm.ptr, i64
      %eq = func.call @__ly_box_equal(%entry, %probe) : (!llvm.ptr, !llvm.ptr) -> i1
      %inc = arith.addi %acc, %one : i64
      %next = arith.select %eq, %inc, %acc : i1, i64
      scf.yield %next : i64
    }
    func.return %count : i64
  }

  // Shared strided box copy for list/tuple slices: duplicate the selected
  // 16-word element boxes into a fresh sequence and retain each copy (two
  // handles now reference each boxed entity).
  // Fill an already-sized destination with the strided selection
  // {start + k*step | k < count} of the source slots, retaining each copy.
  func.func private @__ly_seq_fill_slice(%dst_items: memref<?xi64>, %count: i64, %start: i64, %step: i64, %src_items: memref<?xi64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16_words = func.call @__ly_box_word_count() : () -> i64
    %c16 = arith.index_cast %c16_words : i64 to index
    %handle_words = func.call @__ly_box_word_count() : () -> i64
    %count_index = arith.index_cast %count : i64 to index
    scf.for %k = %c0 to %count_index step %c1 {
      %k64 = arith.index_cast %k : index to i64
      %offset = arith.muli %k64, %step : i64
      %src_slot = arith.addi %start, %offset : i64
      %src_base64 = arith.muli %src_slot, %handle_words : i64
      %src_base = arith.index_cast %src_base64 : i64 to index
      %dst_base = arith.muli %k, %c16 : index
      scf.for %w = %c0 to %c16 step %c1 {
        %src_index = arith.addi %src_base, %w : index
        %dst_index = arith.addi %dst_base, %w : index
        %word = memref.load %src_items[%src_index] : memref<?xi64>
        memref.store %word, %dst_items[%dst_index] : memref<?xi64>
      }
      func.call @LyObject_RetainBoxedPayloadArraySlotRaw(%dst_items, %k64) : (memref<?xi64>, i64) -> ()
    }
    func.return
  }

  // Resolve a slice against a sequence length: raises on step 0, then returns
  // (start, count) with a division-safe substitute step.
  func.func private @__ly_seq_slice_bounds(%len: i64, %start_raw: i64, %stop_raw: i64, %step_raw: i64, %mask: i64) -> (i64, i64, i64) {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %step_zero = arith.cmpi eq, %step_raw, %zero : i64
    scf.if %step_zero {
      func.call @__ly_slice_raise_zero_step() : () -> ()
    }
    // The throw does not return; the substitute keeps the IR division-safe.
    %step = arith.select %step_zero, %one, %step_raw : i1, i64
    %adj:2 = func.call @__ly_slice_adjust(%len, %start_raw, %stop_raw, %step, %mask) : (i64, i64, i64, i64, i64) -> (i64, i64)
    func.return %adj#0, %adj#1, %step : i64, i64, i64
  }
}
