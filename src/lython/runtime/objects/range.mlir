// `range` and its iterator -- CPython's Objects/rangeobject.c.
//
// Signature source (1:1 correspondence target):
//   https://github.com/python/typeshed/blob/main/stdlib/builtins.pyi
//
// Deviations from CPython:
//   - The bounds are 64-bit: `range(2**70, 2**70 + 3)` raises OverflowError
//     where CPython builds the range, and so does a slice whose bounds or step
//     would leave the word (`range(0, 2**62, 2**61)[::4]`). A range longer
//     than INT64_MAX (`range(-sys.maxsize, sys.maxsize)`) indexes, tests
//     membership and iterates, but slicing it raises the OverflowError that
//     len() raises in CPython too, and so does a loop over it inside a
//     generator (`iter(r)`), which the emitter turns into an indexed loop.

module attributes {
  ly.typing.manifest,
  ly.runtime.contracts = ["builtins.range", "builtins.range_iterator"]
} {
  // ===== declared here, defined in another runtime file or built by the lowering =====
  func.func private @__ly_slice_unpack(%self: memref<5xi64>) -> (i64, i64, i64, i64)
  func.func private @__ly_unicode_from_valid_utf8(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]}
  func.func private @LyLong_FromI64(%value: i64 {ly.runtime.default_i64 = 0 : i64}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class, ly.runtime.contract = "builtins.int", ly.runtime.initializer = "__new__"}
  func.func private @LyObject_ReleaseStorageToZero(%storage: memref<?xi64>) -> i1 attributes {ly.runtime.contract = "builtins.object", ly.runtime.primitive = "release_to_zero"}
  func.func private @LyUnicode_Concat(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_bytes: memref<?xi8>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__add__"}
  func.func private @LyUnicode_DecRef(%header: memref<2xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.str", ly.runtime.deallocator}
  func.func private @LyUnicode_FromBytes(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class, ly.runtime.contract = "builtins.str", ly.runtime.initializer = "__new__"}
  func.func private @LyUnicode_FromI64(%value: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]}
  func.func private @Ly_IncRef(%header: memref<2xi64, strided<[1], offset: ?>> {ly.ownership.object_header}) attributes {ly.ownership.retain_args = [0], ly.runtime.primitive = "retain"}
  func.func private @__ly_long_parts(%header: memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
  func.func private @__ly_long_raise_too_large()
  func.func private @__ly_long_view_as_i64(%meta: memref<2xi64>, %digits: memref<?xi32>) -> i64
  func.func private @__ly_long_view_fits_i64(%meta: memref<2xi64>, %digits: memref<?xi32>) -> i1
  func.func private @__ly_raise_static_message(%class_word: i64, %message: memref<?xi8>, %length: i64)
  func.func private @__ly_slice_indices(%len: i64, %start_in: i64, %stop_in: i64, %step: i64, %mask: i64) -> (i64, i64)
  func.func private @__ly_slice_raise_zero_step()
  memref.global "private" constant @__ly_repr_comma : memref<2xi8>
  memref.global "private" constant @__ly_repr_range_open : memref<6xi8>
  memref.global "private" constant @__ly_repr_rparen : memref<1xi8>
  py.class @range attributes {base_names = ["Sequence", "Hashable"],
                             ly.typing.base_args = [[!py.contract<"builtins.int">], []],
                             ly.typing.final,
    field_names = ["start", "stop", "step"],
    field_contract_types = [
      !py.contract<"builtins.int">,
      !py.contract<"builtins.int">,
      !py.contract<"builtins.int">
    ],
    method_names = ["__new__", "__new__", "__new__", "__init__", "__init__",
                    "__init__", "__iter__", "__eq__", "__ne__", "__getslice__", "__getslice__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.range">>, !py.contract<"builtins.int">] -> [!py.self]>,
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.range">>, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.self]>,
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.range">>, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.self]>,
      !py.protocol<"Callable", [!py.contract<"builtins.range">, !py.contract<"builtins.int">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.range">, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.range">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.range">] -> [!py.contract<"builtins.range_iterator">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.range">, !py.contract<"builtins.range">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.range">, !py.contract<"builtins.range">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.range">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.contract<"builtins.range">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.range">, !py.contract<"builtins.slice">] -> [!py.contract<"builtins.range">]>
    ],
    method_kinds = ["classmethod", "classmethod", "classmethod", "instance",
                    "instance", "instance", "instance", "instance", "instance",
                    "instance", "instance"]
  } {}

  py.class @range_iterator attributes {
    base_names = ["Iterator"],
    ly.typing.base_args = [[!py.contract<"builtins.int">]],
    method_names = ["__iter__", "__next__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"builtins.range_iterator">] -> [!py.self]>,
      !py.protocol<"Callable", [!py.contract<"builtins.range_iterator">] -> [!py.contract<"builtins.int">]>
    ],
    method_kinds = ["instance", "instance"]
  } {}

  // range.__repr__: `range(stop)` is still spelled `range(0, 3)` -- CPython
  // prints the normalized triple and drops only a step of 1. The address form
  // this fell back to told the reader nothing.
  func.func @LyRange_Repr(%self: memref<5xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.range", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c6_i64 = arith.constant 6 : i64
    %one = arith.constant 1 : i64
    %start_slot = arith.constant 2 : index
    %stop_slot = arith.constant 3 : index
    %step_slot = arith.constant 4 : index
    %start = memref.load %self[%start_slot] : memref<5xi64>
    %stop = memref.load %self[%stop_slot] : memref<5xi64>
    %step = memref.load %self[%step_slot] : memref<5xi64>
    %open_ref = memref.get_global @__ly_repr_range_open : memref<6xi8>
    %open_dyn = memref.cast %open_ref : memref<6xi8> to memref<?xi8>
    %oh, %ob = func.call @__ly_unicode_from_valid_utf8(%open_dyn, %c0, %c6_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %sh0, %sb0 = func.call @LyUnicode_FromI64(%start) : (i64) -> (memref<2xi64>, memref<?xi8>)
    %a_h, %a_b = func.call @LyUnicode_Concat(%oh, %ob, %sh0, %sb0) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.call @LyUnicode_DecRef(%oh) : (memref<2xi64>) -> ()
    func.call @LyUnicode_DecRef(%sh0) : (memref<2xi64>) -> ()
    %comma_ref = memref.get_global @__ly_repr_comma : memref<2xi8>
    %comma_dyn = memref.cast %comma_ref : memref<2xi8> to memref<?xi8>
    %ch0, %cb0 = func.call @__ly_unicode_from_valid_utf8(%comma_dyn, %c0, %c2_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %b_h, %b_b = func.call @LyUnicode_Concat(%a_h, %a_b, %ch0, %cb0) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.call @LyUnicode_DecRef(%a_h) : (memref<2xi64>) -> ()
    func.call @LyUnicode_DecRef(%ch0) : (memref<2xi64>) -> ()
    %sh1, %sb1 = func.call @LyUnicode_FromI64(%stop) : (i64) -> (memref<2xi64>, memref<?xi8>)
    %c_h, %c_b = func.call @LyUnicode_Concat(%b_h, %b_b, %sh1, %sb1) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.call @LyUnicode_DecRef(%b_h) : (memref<2xi64>) -> ()
    func.call @LyUnicode_DecRef(%sh1) : (memref<2xi64>) -> ()
    %unit_step = arith.cmpi eq, %step, %one : i64
    %d:2 = scf.if %unit_step -> (memref<2xi64>, memref<?xi8>) {
      scf.yield %c_h, %c_b : memref<2xi64>, memref<?xi8>
    } else {
      %ch1, %cb1 = func.call @__ly_unicode_from_valid_utf8(%comma_dyn, %c0, %c2_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
      %e_h, %e_b = func.call @LyUnicode_Concat(%c_h, %c_b, %ch1, %cb1) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      func.call @LyUnicode_DecRef(%c_h) : (memref<2xi64>) -> ()
      func.call @LyUnicode_DecRef(%ch1) : (memref<2xi64>) -> ()
      %sh2, %sb2 = func.call @LyUnicode_FromI64(%step) : (i64) -> (memref<2xi64>, memref<?xi8>)
      %f_h, %f_b = func.call @LyUnicode_Concat(%e_h, %e_b, %sh2, %sb2) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      func.call @LyUnicode_DecRef(%e_h) : (memref<2xi64>) -> ()
      func.call @LyUnicode_DecRef(%sh2) : (memref<2xi64>) -> ()
      scf.yield %f_h, %f_b : memref<2xi64>, memref<?xi8>
    }
    %rp_ref = memref.get_global @__ly_repr_rparen : memref<1xi8>
    %rp_dyn = memref.cast %rp_ref : memref<1xi8> to memref<?xi8>
    %rph, %rpb = func.call @__ly_unicode_from_valid_utf8(%rp_dyn, %c0, %c1_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %out_h, %out_b = func.call @LyUnicode_Concat(%d#0, %d#1, %rph, %rpb) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    func.call @LyUnicode_DecRef(%d#0) : (memref<2xi64>) -> ()
    func.call @LyUnicode_DecRef(%rph) : (memref<2xi64>) -> ()
    func.return %out_h, %out_b : memref<2xi64>, memref<?xi8>
  }

  // ===== impls: range =====
  // Handle words: 0 refcount, 1 layout/destructor family id, 2 start (the
  // iterator's current), 3 stop, 4 step. Five is deliberately unlike either
  // lane width the two-lane form used (memref<2xi64> header, memref<3xi64>
  // state): verifyReceiverShape compares a prefix, so reusing 2 would have let
  // an unconverted method pass with its stale state lane outside the window.
  func.func private @LyRange_Shape() -> memref<5xi64> attributes {ly.runtime.contract = "builtins.range", ly.runtime.shape}
  func.func private @LyRangeIterator_Shape() -> memref<5xi64> attributes {ly.runtime.contract = "builtins.range_iterator", ly.runtime.shape}

  memref.global "private" constant @__ly_range_msg_zero_step : memref<30xi8> = dense<[114, 97, 110, 103, 101, 40, 41, 32, 97, 114, 103, 32, 51, 32, 109, 117, 115, 116, 32, 110, 111, 116, 32, 98, 101, 32, 122, 101, 114, 111]>

  func.func private @__ly_range_raise_zero_step() {
    %class_word = arith.constant {ly.class_of = "builtins.ValueError"} 53 : i64
    %length = arith.constant 30 : i64
    %message_static = memref.get_global @__ly_range_msg_zero_step : memref<30xi8>
    %message = memref.cast %message_static : memref<30xi8> to memref<?xi8>
    func.call @__ly_raise_static_message(%class_word, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  // range(stop), range(start, stop), range(start, stop, step): one
  // initializer per arity, as CPython's range_from_array switches on the
  // argument count.
  //
  // ⛔ Not one initializer with defaulted arguments, which is what this was: a
  // default is a value the caller can also pass, and INT64_MAX standing for
  // "absent" made `range(5, sys.maxsize)` the `range(5)` it is not, and
  // `range(0, 10, sys.maxsize)` a step-1 range of ten.
  func.func @LyRange_New(%stop: i64) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class, ly.runtime.contract = "builtins.range", ly.runtime.initializer = "__new__"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %self = func.call @__ly_range_alloc(%zero, %stop, %one) : (i64, i64, i64) -> memref<5xi64>
    func.return %self : memref<5xi64>
  }

  func.func @LyRange_NewStart(%start: i64, %stop: i64) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class, ly.runtime.contract = "builtins.range", ly.runtime.initializer = "__new__"} {
    %one = arith.constant 1 : i64
    %self = func.call @__ly_range_alloc(%start, %stop, %one) : (i64, i64, i64) -> memref<5xi64>
    func.return %self : memref<5xi64>
  }

  func.func @LyRange_NewStep(%start: i64, %stop: i64, %step: i64) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class, ly.runtime.contract = "builtins.range", ly.runtime.initializer = "__new__"} {
    %zero = arith.constant 0 : i64
    %step_zero = arith.cmpi eq, %step, %zero : i64
    cf.cond_br %step_zero, ^raise, ^make

  ^raise:
    func.call @__ly_range_raise_zero_step() : () -> ()
    cf.br ^make

  ^make:
    %self = func.call @__ly_range_alloc(%start, %stop, %step) : (i64, i64, i64) -> memref<5xi64>
    func.return %self : memref<5xi64>
  }

  func.func private @__ly_range_alloc(%start: i64, %stop: i64, %step: i64) -> memref<5xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.range"], ly.ownership.owned_results = [0]} {
    %one = arith.constant 1 : i64
    // One entity, one handle: the five words ARE the entity. The allocation is
    // still spelled as a byte block plus a view, rather than `memref.alloc() :
    // memref<5xi64>`, so the diff against the two-lane form changes only the
    // view's result type -- the owned-local marker keeps sitting on the same op
    // kind the ownership collector already reads it from.
    %block_bytes = arith.constant 40 : index
    %block = memref.alloc(%block_bytes) {alignment = 16 : i64} : memref<?xi8>
    %self_offset = arith.constant 0 : index
    %self = memref.view %block[%self_offset][] {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<?xi8> to memref<5xi64>
    %layout_range = arith.constant {ly.class_of = "builtins.range"} 3 : i64
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %start_slot = arith.constant 2 : index
    %stop_slot = arith.constant 3 : index
    %step_slot = arith.constant 4 : index
    memref.store %one, %self[%refcount_slot] : memref<5xi64>
    memref.store %layout_range, %self[%layout_slot] : memref<5xi64>
    memref.store %start, %self[%start_slot] : memref<5xi64>
    memref.store %stop, %self[%stop_slot] : memref<5xi64>
    memref.store %step, %self[%step_slot] : memref<5xi64>
    func.return %self : memref<5xi64>
  }

  func.func @LyRange_Init(%self: memref<5xi64> {ly.ownership.object_header}) attributes {ly.runtime.contract = "builtins.range", ly.runtime.method = "__init__"} {
    func.return
  }

  memref.global "private" constant @__ly_range_msg_index_out_of_range : memref<31xi8> = dense<[114, 97, 110, 103, 101, 32, 111, 98, 106, 101, 99, 116, 32, 105, 110, 100, 101, 120, 32, 111, 117, 116, 32, 111, 102, 32, 114, 97, 110, 103, 101]>

  func.func private @__ly_range_raise_index_error() {
    %class_word = arith.constant {ly.class_of = "builtins.IndexError"} 55 : i64
    %length = arith.constant 31 : i64
    %message_static = memref.get_global @__ly_range_msg_index_out_of_range : memref<31xi8>
    %message = memref.cast %message_static : memref<31xi8> to memref<?xi8>
    func.call @__ly_raise_static_message(%class_word, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  // CPython get_len_of_range (Objects/rangeobject.c): the count of steps from
  // start before passing stop, in UNSIGNED arithmetic -- `stop - 1 - start` is
  // at most 2 * INT64_MAX, which a signed word cannot hold and an unsigned one
  // can. The answer is a u64 in the word's bits: a range longer than
  // INT64_MAX (`range(-sys.maxsize, sys.maxsize)`) reads negative as a signed
  // length, and each caller says what that means for it.
  //
  // ⛔ Not signed, which is what this was: `range(-sys.maxsize, sys.maxsize,
  // sys.maxsize)` measured its span as a negative number, had length 0, and
  // `r[1]` was an IndexError for a range whose second element is 0.
  // Takes the entity, not a view of its state words: a helper that received a
  // view would have to be trusted to have been handed a live one, whereas the
  // handle is the operand release placement already follows.
  func.func private @__ly_range_length(%self: memref<5xi64>) -> i64 {
    %start_slot = arith.constant 2 : index
    %stop_slot = arith.constant 3 : index
    %step_slot = arith.constant 4 : index
    %start = memref.load %self[%start_slot] : memref<5xi64>
    %stop = memref.load %self[%stop_slot] : memref<5xi64>
    %step = memref.load %self[%step_slot] : memref<5xi64>
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %ascending = arith.cmpi sgt, %step, %zero : i64
    cf.cond_br %ascending, ^up, ^down

  ^up:
    %nonempty_up = arith.cmpi slt, %start, %stop : i64
    cf.cond_br %nonempty_up, ^count_up, ^empty

  ^count_up:
    %last_up = arith.subi %stop, %one : i64
    %span_up = arith.subi %last_up, %start : i64
    %steps_up = arith.divui %span_up, %step : i64
    %len_up = arith.addi %steps_up, %one : i64
    func.return %len_up : i64

  ^down:
    %nonempty_down = arith.cmpi sgt, %start, %stop : i64
    cf.cond_br %nonempty_down, ^count_down, ^empty

  ^count_down:
    %last_down = arith.subi %start, %one : i64
    %span_down = arith.subi %last_down, %stop : i64
    // 0 - INT64_MIN is 2**63 as an unsigned word, which is the magnitude.
    %step_magnitude = arith.subi %zero, %step : i64
    %steps_down = arith.divui %span_down, %step_magnitude : i64
    %len_down = arith.addi %steps_down, %one : i64
    func.return %len_down : i64

  ^empty:
    func.return %zero : i64
  }

  // "Python int too large to convert to C ssize_t": what len() of a range
  // longer than INT64_MAX raises in CPython, and what a 64-bit range raises
  // where it cannot answer.
  memref.global "private" constant @__ly_range_msg_too_long : memref<44xi8> = dense<[80, 121, 116, 104, 111, 110, 32, 105, 110, 116, 32, 116, 111, 111, 32, 108, 97, 114, 103, 101, 32, 116, 111, 32, 99, 111, 110, 118, 101, 114, 116, 32, 116, 111, 32, 67, 32, 115, 115, 105, 122, 101, 95, 116]>

  func.func private @__ly_range_raise_too_long() {
    %class_word = arith.constant {ly.class_of = "builtins.OverflowError"} 104 : i64
    %length = arith.constant 44 : i64
    %message_static = memref.get_global @__ly_range_msg_too_long : memref<44xi8>
    %message = memref.cast %message_static : memref<44xi8> to memref<?xi8>
    func.call @__ly_raise_static_message(%class_word, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  // The length a caller can use as a signed count, or the OverflowError
  // len() raises when there is none.
  func.func private @__ly_range_length_checked(%self: memref<5xi64>) -> i64 {
    %zero = arith.constant 0 : i64
    %length = func.call @__ly_range_length(%self) : (memref<5xi64>) -> i64
    %too_long = arith.cmpi slt, %length, %zero : i64
    cf.cond_br %too_long, ^raise, ^ok

  ^raise:
    func.call @__ly_range_raise_too_long() : () -> ()
    cf.br ^ok

  ^ok:
    func.return %length : i64
  }

  // range's own method_names declares only __new__/__init__/__iter__; __len__,
  // __getitem__ and __contains__ are promised by `base_names = ["Sequence"]`
  // instead, which is why they were declared-but-unimplemented without showing
  // up in a method_names sweep. len(r), r[i] and `v in r` all resolved through
  // the Sequence tower to a builtins.range method that did not exist.
  // ⭐ range.start / .stop / .step are the three words the object already
  // stores, boxed. Without them an attribute read reached the lowering's
  // "attr.get object type has no class schema" -- an internal sentence for
  // three attributes CPython answers with the numbers the constructor was
  // given. `which` is 0/1/2 and the caller passes a constant.
  func.func @LyRange_Field(%self: memref<5xi64> {ly.ownership.object_header}, %which: i64) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.range", ly.runtime.primitive = "field", ly.runtime.result_contract = "builtins.int"} {
    %two = arith.constant 2 : i64
    %slot_i64 = arith.addi %which, %two : i64
    %slot = arith.index_cast %slot_i64 : i64 to index
    %raw = memref.load %self[%slot] : memref<5xi64>
    %boxed = func.call @LyLong_FromI64(%raw) : (i64) -> memref<2xi64>
    func.return %boxed : memref<2xi64>
  }

  // compute_item's arithmetic, `start + i * step`, with whether the answer
  // is a word: a 64-bit range has no bigger int to hand back. The product is
  // taken in 128 bits and the start added with its carry, because the product
  // alone may leave the word where the sum does not -- the stop of
  // `range(-sys.maxsize, sys.maxsize, sys.maxsize)[1:]` is -M + 2 * M.
  func.func private @__ly_range_item_in_word(%start: i64, %index: i64, %step: i64) -> (i64, i1) {
    %c63 = arith.constant 63 : i64
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %low, %high = arith.mulsi_extended %index, %step : i64
    %start_high = arith.shrsi %start, %c63 : i64
    %sum = arith.addi %low, %start : i64
    %wrapped = arith.cmpi ult, %sum, %low : i64
    %carry = arith.select %wrapped, %one, %zero : i64
    %partial_high = arith.addi %high, %start_high : i64
    %sum_high = arith.addi %partial_high, %carry : i64
    %sum_sign = arith.shrsi %sum, %c63 : i64
    %fits = arith.cmpi eq, %sum_high, %sum_sign : i64
    func.return %sum, %fits : i64, i1
  }

  // range[start:stop:step] -- CPython compute_slice (Objects/rangeobject.c):
  // the slice's indices against the range's length (_PySlice_GetLongIndices),
  // each mapped through the range (compute_item), and the two steps
  // multiplied. The result is a range again, spelled with the clamped stop:
  // `range(10)[1:8:3]` is `range(1, 8, 3)`, not the `range(1, 10, 3)` that a
  // length would give.
  func.func @LyRange_GetSlice(%self: memref<5xi64> {ly.ownership.object_header}, %start_raw: i64 {ly.runtime.clip_i64}, %stop_raw: i64 {ly.runtime.clip_i64}, %step_raw: i64 {ly.runtime.clip_i64}, %mask: i64) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.range", ly.runtime.method = "__getslice__", ly.runtime.result_contract = "builtins.range"} {
    %zero = arith.constant 0 : i64
    %start_slot = arith.constant 2 : index
    %step_slot = arith.constant 4 : index
    %step_zero = arith.cmpi eq, %step_raw, %zero : i64
    cf.cond_br %step_zero, ^zero_step, ^slice

  ^zero_step:
    func.call @__ly_slice_raise_zero_step() : () -> ()
    cf.br ^slice

  ^slice:
    %len = func.call @__ly_range_length_checked(%self) : (memref<5xi64>) -> i64
    %first, %last = func.call @__ly_slice_indices(%len, %start_raw, %stop_raw, %step_raw, %mask) : (i64, i64, i64, i64, i64) -> (i64, i64)
    %range_start = memref.load %self[%start_slot] : memref<5xi64>
    %range_step = memref.load %self[%step_slot] : memref<5xi64>
    %sub_start, %start_fits = func.call @__ly_range_item_in_word(%range_start, %first, %range_step) : (i64, i64, i64) -> (i64, i1)
    %sub_stop, %stop_fits = func.call @__ly_range_item_in_word(%range_start, %last, %range_step) : (i64, i64, i64) -> (i64, i1)
    %sub_step, %step_fits = func.call @__ly_range_item_in_word(%zero, %step_raw, %range_step) : (i64, i64, i64) -> (i64, i1)
    %bounds_fit = arith.andi %start_fits, %stop_fits : i1
    %fits = arith.andi %bounds_fit, %step_fits : i1
    cf.cond_br %fits, ^make, ^too_large

  ^too_large:
    func.call @__ly_long_raise_too_large() : () -> ()
    cf.br ^make

  ^make:
    %result = func.call @__ly_range_alloc(%sub_start, %sub_stop, %sub_step) : (i64, i64, i64) -> memref<5xi64>
    func.return %result : memref<5xi64>
  }

  // r[s] for a slice object (range_subscript).
  func.func @LyRange_SliceSubscript(%self: memref<5xi64> {ly.ownership.object_header}, %slice: memref<5xi64> {ly.ownership.object_header}) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.range", ly.runtime.method = "__getslice__", ly.runtime.result_contract = "builtins.range"} {
    %start, %stop, %step, %mask = func.call @__ly_slice_unpack(%slice) : (memref<5xi64>) -> (i64, i64, i64, i64)
    %result = func.call @LyRange_GetSlice(%self, %start, %stop, %step, %mask) : (memref<5xi64>, i64, i64, i64, i64) -> memref<5xi64>
    func.return %result : memref<5xi64>
  }

  func.func @LyRange_Len(%self: memref<5xi64> {ly.ownership.object_header}) -> i64 attributes {ly.runtime.contract = "builtins.range", ly.runtime.method = "__len__"} {
    %length = func.call @__ly_range_length_checked(%self) : (memref<5xi64>) -> i64
    func.return %length : i64
  }

  // ⭐ TWO RANGES ARE EQUAL WHEN THEY PRODUCE THE SAME SEQUENCE, which is what
  // CPython's range_richcompare answers -- `range(3) == range(0, 3)` is True
  // and `range(0, 3, 7) == range(0, 3, 9)` is too, because a one-element range
  // never uses its step. With no __eq__ of its own, `range(3) == range(3)` fell
  // through to identity and printed False.
  func.func @LyRange_EqBool(%lhs: memref<5xi64> {ly.ownership.object_header}, %rhs: memref<5xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.range", ly.runtime.method = "__eq__"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %start_slot = arith.constant 2 : index
    %step_slot = arith.constant 4 : index
    %llen = func.call @__ly_range_length(%lhs) : (memref<5xi64>) -> i64
    %rlen = func.call @__ly_range_length(%rhs) : (memref<5xi64>) -> i64
    %same_len = arith.cmpi eq, %llen, %rlen : i64
    %answer = scf.if %same_len -> (i1) {
      %empty = arith.cmpi eq, %llen, %zero : i64
      %both_empty = scf.if %empty -> (i1) {
        %true_v = arith.constant true
        scf.yield %true_v : i1
      } else {
        %lstart = memref.load %lhs[%start_slot] : memref<5xi64>
        %rstart = memref.load %rhs[%start_slot] : memref<5xi64>
        %same_start = arith.cmpi eq, %lstart, %rstart : i64
        %rest = scf.if %same_start -> (i1) {
          %single = arith.cmpi eq, %llen, %one : i64
          %step_ok = scf.if %single -> (i1) {
            %true_v = arith.constant true
            scf.yield %true_v : i1
          } else {
            %lstep = memref.load %lhs[%step_slot] : memref<5xi64>
            %rstep = memref.load %rhs[%step_slot] : memref<5xi64>
            %same_step = arith.cmpi eq, %lstep, %rstep : i64
            scf.yield %same_step : i1
          }
          scf.yield %step_ok : i1
        } else {
          %false_v = arith.constant false
          scf.yield %false_v : i1
        }
        scf.yield %rest : i1
      }
      scf.yield %both_empty : i1
    } else {
      %false_v = arith.constant false
      scf.yield %false_v : i1
    }
    func.return %answer : i1
  }

  func.func @LyRange_NeBool(%lhs: memref<5xi64> {ly.ownership.object_header}, %rhs: memref<5xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.range", ly.runtime.method = "__ne__"} {
    %true_v = arith.constant true
    %eq = func.call @LyRange_EqBool(%lhs, %rhs) : (memref<5xi64>, memref<5xi64>) -> i1
    %ne = arith.xori %eq, %true_v : i1
    func.return %ne : i1
  }

  // CPython compute_range_item: a negative index counts from the end, and
  // the element is start + i * step. The comparisons are against the length as
  // an unsigned word (see __ly_range_length), and the element is computed in
  // wrapping arithmetic, which is exact because every element is a word.
  // An index past the word is past every range shorter than the word.
  func.func @LyRange_GetItem(%self: memref<5xi64> {ly.ownership.object_header}, %index_header: memref<2xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.range", ly.runtime.method = "__getitem__", ly.runtime.result_contract = "builtins.int"} {
    %zero = arith.constant 0 : i64
    %meta, %digits = func.call @__ly_long_parts(%index_header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %fits = func.call @__ly_long_view_fits_i64(%meta, %digits) : (memref<2xi64>, memref<?xi32>) -> i1
    %length = func.call @__ly_range_length(%self) : (memref<5xi64>) -> i64
    cf.cond_br %fits, ^word, ^wide

  ^wide:
    %long = arith.cmpi slt, %length, %zero : i64
    cf.cond_br %long, ^too_long, ^raise

  ^too_long:
    func.call @__ly_range_raise_too_long() : () -> ()
    cf.br ^raise

  ^word:
    %index = func.call @__ly_long_view_as_i64(%meta, %digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %negative = arith.cmpi slt, %index, %zero : i64
    // 0 - INT64_MIN wraps to 2**63, which is the magnitude as an unsigned word.
    %magnitude = arith.subi %zero, %index : i64
    %in_forward = arith.cmpi ult, %index, %length : i64
    %in_backward = arith.cmpi ule, %magnitude, %length : i64
    %in_range = arith.select %negative, %in_backward, %in_forward : i1
    %from_end = arith.addi %length, %index : i64
    %normalized = arith.select %negative, %from_end, %index : i64
    cf.cond_br %in_range, ^compute(%normalized : i64), ^raise

  ^raise:
    func.call @__ly_range_raise_index_error() : () -> ()
    cf.br ^compute(%zero : i64)

  ^compute(%position: i64):
    %start_slot = arith.constant 2 : index
    %step_slot = arith.constant 4 : index
    %start = memref.load %self[%start_slot] : memref<5xi64>
    %step = memref.load %self[%step_slot] : memref<5xi64>
    %offset = arith.muli %position, %step : i64
    %value = arith.addi %start, %offset : i64
    %h = func.call @LyLong_FromI64(%value) : (i64) -> memref<2xi64>
    func.return %h : memref<2xi64>
  }

  // CPython range_contains_long: arithmetic, not a scan -- membership is an
  // in-bounds test plus a stride test, so `v in range(n)` stays O(1).
  func.func @LyRange_Contains(%self: memref<5xi64> {ly.ownership.object_header}, %value_header: memref<2xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.range", ly.runtime.method = "__contains__"} {
    %false = arith.constant false
    %meta, %digits = func.call @__ly_long_parts(%value_header) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %fits = func.call @__ly_long_view_fits_i64(%meta, %digits) : (memref<2xi64>, memref<?xi32>) -> i1
    // An int past the word is in no 64-bit range.
    cf.cond_br %fits, ^word, ^outside

  ^outside:
    func.return %false : i1

  ^word:
    %value = func.call @__ly_long_view_as_i64(%meta, %digits) : (memref<2xi64>, memref<?xi32>) -> i64
    %start_slot = arith.constant 2 : index
    %stop_slot = arith.constant 3 : index
    %step_slot = arith.constant 4 : index
    %start = memref.load %self[%start_slot] : memref<5xi64>
    %stop = memref.load %self[%stop_slot] : memref<5xi64>
    %step = memref.load %self[%step_slot] : memref<5xi64>
    %zero = arith.constant 0 : i64
    %ascending = arith.cmpi sgt, %step, %zero : i64
    %ge_start = arith.cmpi sge, %value, %start : i64
    %lt_stop = arith.cmpi slt, %value, %stop : i64
    %in_up = arith.andi %ge_start, %lt_stop : i1
    %le_start = arith.cmpi sle, %value, %start : i64
    %gt_stop = arith.cmpi sgt, %value, %stop : i64
    %in_down = arith.andi %le_start, %gt_stop : i1
    %in_bounds = arith.select %ascending, %in_up, %in_down : i1
    cf.cond_br %in_bounds, ^stride, ^miss

  ^miss:
    func.return %false : i1

  ^stride:
    // The distance from start and the step's magnitude, as unsigned words:
    // inside the bounds the distance is at most 2 * INT64_MAX, which only an
    // unsigned word holds.
    %up_offset = arith.subi %value, %start : i64
    %down_offset = arith.subi %start, %value : i64
    %offset = arith.select %ascending, %up_offset, %down_offset : i64
    %negated_step = arith.subi %zero, %step : i64
    %step_magnitude = arith.select %ascending, %step, %negated_step : i64
    %remainder = arith.remui %offset, %step_magnitude : i64
    %aligned = arith.cmpi eq, %remainder, %zero : i64
    func.return %aligned : i1
  }

  // The iterator keeps how many elements are LEFT, not the stop, as CPython's
  // rangeiterobject does. ⛔ Not current-versus-stop, which is what this was:
  // the step after the last element can leave the word, and the wrapped value
  // compared as before the stop again -- `list(range(0, sys.maxsize,
  // sys.maxsize // 2 + 1))` never ended.
  func.func private @__ly_range_iterator_alloc(%current: i64, %remaining: i64, %step: i64) -> memref<5xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.range_iterator"], ly.ownership.owned_results = [0], ly.runtime.class, ly.runtime.contract = "builtins.range_iterator", ly.runtime.primitive = "alloc"} {
    // One entity, one handle; see LyRange_New for why the byte block and the
    // view are kept rather than allocating memref<5xi64> directly.
    %block_bytes = arith.constant 40 : index
    %block = memref.alloc(%block_bytes) {alignment = 16 : i64} : memref<?xi8>
    %self_offset = arith.constant 0 : index
    %self = memref.view %block[%self_offset][] {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<?xi8> to memref<5xi64>
    %one = arith.constant 1 : i64
    %layout_range_iterator = arith.constant {ly.class_of = "builtins.range_iterator"} 20 : i64
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %current_slot = arith.constant 2 : index
    %remaining_slot = arith.constant 3 : index
    %step_slot = arith.constant 4 : index
    memref.store %one, %self[%refcount_slot] : memref<5xi64>
    memref.store %layout_range_iterator, %self[%layout_slot] : memref<5xi64>
    memref.store %current, %self[%current_slot] : memref<5xi64>
    memref.store %remaining, %self[%remaining_slot] : memref<5xi64>
    memref.store %step, %self[%step_slot] : memref<5xi64>
    func.return %self : memref<5xi64>
  }

  func.func @LyRange_Iter(%self: memref<5xi64> {ly.ownership.object_header}) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.range", ly.runtime.method = "__iter__", ly.runtime.result_contract = "builtins.range_iterator"} {
    %start_slot = arith.constant 2 : index
    %step_slot = arith.constant 4 : index
    %start = memref.load %self[%start_slot] : memref<5xi64>
    %step = memref.load %self[%step_slot] : memref<5xi64>
    %length = func.call @__ly_range_length(%self) : (memref<5xi64>) -> i64
    %iter = func.call @__ly_range_iterator_alloc(%start, %length, %step) : (i64, i64, i64) -> memref<5xi64>
    func.return %iter : memref<5xi64>
  }

  func.func @LyRangeIterator_Iter(%self: memref<5xi64> {ly.ownership.object_header}) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.range_iterator", ly.runtime.method = "__iter__", ly.runtime.result_contract = "builtins.range_iterator"} {
    // Narrowing to the two refcount words needs a subview, not a memref.cast:
    // a cast may not change a static extent (5 -> 2), it may only make one
    // dynamic. Same spelling as lyrt.mlir's 4-word counters.
    %retain_offset = arith.constant 0 : index
    %refcount_view = memref.subview %self[%retain_offset] [2] [1] : memref<5xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%refcount_view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    func.return %self : memref<5xi64>
  }

  func.func @LyRangeIterator_Next(%self: memref<5xi64> {ly.ownership.object_header}) -> (memref<2xi64>, i1, memref<5xi64>) attributes {ly.ownership.owned_result_contracts = ["builtins.int", "builtins.range_iterator"], ly.ownership.owned_results = [0, 2], ly.runtime.contract = "builtins.range_iterator", ly.runtime.method = "__next__", ly.runtime.element_contract = "builtins.int", ly.runtime.next_contract = "builtins.range_iterator", ly.runtime.valid_result_index = 1 : i64} {
    %current_slot = arith.constant 2 : index
    %remaining_slot = arith.constant 3 : index
    %step_slot = arith.constant 4 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %current = memref.load %self[%current_slot] : memref<5xi64>
    %remaining = memref.load %self[%remaining_slot] : memref<5xi64>
    %step = memref.load %self[%step_slot] : memref<5xi64>
    %valid = arith.cmpi ne, %remaining, %zero : i64
    // The advance past the last element may wrap; it is never read.
    %advanced = arith.addi %current, %step : i64
    %next_current = arith.select %valid, %advanced, %current : i1, i64
    %counted = arith.subi %remaining, %one : i64
    %next_remaining = arith.select %valid, %counted, %remaining : i1, i64
    // The advance is now a store through the handle, so every holder of the
    // iterator observes it; the two-lane form wrote it through a state lane
    // that travelled beside the handle.
    memref.store %next_current, %self[%current_slot] : memref<5xi64>
    memref.store %next_remaining, %self[%remaining_slot] : memref<5xi64>
    %retain_offset = arith.constant 0 : index
    %refcount_view = memref.subview %self[%retain_offset] [2] [1] : memref<5xi64> to memref<2xi64, strided<[1], offset: ?>>
    func.call @Ly_IncRef(%refcount_view) : (memref<2xi64, strided<[1], offset: ?>>) -> ()
    %element_value = arith.select %valid, %current, %zero : i1, i64
    %element = func.call @LyLong_FromI64(%element_value) : (i64) -> memref<2xi64>
    func.return %element, %valid, %self : memref<2xi64>, i1, memref<5xi64>
  }

  func.func @LyRange_DecRef(%self: memref<5xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.range", ly.runtime.deallocator} {
    %storage = memref.cast %self : memref<5xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    memref.dealloc %self : memref<5xi64>
    cf.br ^done

  ^done:
    func.return
  }

  func.func @LyRangeIterator_DecRef(%self: memref<5xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.range_iterator", ly.runtime.deallocator} {
    %storage = memref.cast %self : memref<5xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    memref.dealloc %self : memref<5xi64>
    cf.br ^done

  ^done:
    func.return
  }
}
