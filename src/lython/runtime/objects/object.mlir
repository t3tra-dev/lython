// `object` and `NoneType`, and the generic object protocol every boxed value
// goes through -- CPython's Objects/object.c.
//
// Signature source (1:1 correspondence target):
//   https://github.com/python/typeshed/blob/main/stdlib/builtins.pyi (`object`)
//   https://github.com/python/typeshed/blob/main/stdlib/types.pyi (`NoneType`)
//
// A value whose static type is `object` or a union lives in a box: the class
// id and the payload's handle words, laid out once here (`__ly_box_*`). The
// box's methods dispatch on the class id through hooks the lowering generates
// per program (`__ly_*_boxed_by_contract`, Runtime/ABI/RuntimeABI.cpp), as
// PyObject_Repr / PyObject_RichCompare / PyObject_Hash dispatch through the
// type's slots; so do the errors those raise when nothing answers ("'<' not
// supported between instances", "unhashable type").
//
// ⛔ Not Objects/typeobject.c for `object` itself, where CPython defines it:
// `object`'s methods here ARE those dispatchers (`object.__repr__` of a box is
// the repr of its payload), so they stay with the protocol.

module attributes {
  ly.typing.manifest,
  ly.runtime.contracts = ["types.NoneType", "builtins.object"]
} {
  // ===== declared here, defined in another runtime file or built by the lowering =====
  func.func private @__ly_unicode_from_valid_utf8(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]}
  func.func private @LyFloat_SlotWordAsF64(%word: i64) -> f64 attributes {ly.runtime.contract = "builtins.float", ly.runtime.primitive = "slot_word_as_f64"}
  func.func private @LyLong_SlotWordAsI64(%word: i64) -> (i64, i1) attributes {ly.runtime.contract = "builtins.int", ly.runtime.primitive = "slot_word_as_i64"}
  func.func private @LyLong_TryAsI64(%header: memref<2xi64> {ly.ownership.object_header}) -> (i64, i1) attributes {ly.runtime.contract = "builtins.int", ly.runtime.primitive = "try_unbox.i64"}
  func.func private @LyUnicode_FromBytes(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 4 : i64, ly.runtime.contract = "builtins.str", ly.runtime.initializer = "__new__"}
  func.func private @__ly_float_immediate_fits(%bits: i64) -> i1
  func.func private @__ly_float_to_immediate(%bits: i64) -> i64
  func.func private @__ly_hash_fixup(%h: i64) -> i64
  func.func private @__ly_int_from_immediate(%word: i64) -> i64
  func.func private @__ly_int_immediate_fits(%value: i64) -> i1
  func.func private @__ly_int_to_immediate(%value: i64) -> i64
  func.func private @__ly_raise_static_message(%class_id: i64, %message: memref<?xi8>, %length: i64)
  func.func private @__ly_str_boxed_by_contract(%box: !llvm.ptr, %class_id: i64) -> (memref<2xi64>, memref<?xi8>, i1)
  func.func private @release_payload_slot_ptr(%slot: !llvm.ptr)
  py.class @object attributes {
    ly.typing.abstract,
    method_names = ["__init__", "__new__", "__repr__", "__str__", "__bool__",
                    "__eq__", "__ne__", "__hash__", "__getattribute__",
                    "__setattr__", "__delattr__", "__ly_is_none__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.contract<"builtins.object">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.object">>] -> [!py.self]>,
      !py.protocol<"Callable", [!py.contract<"builtins.object">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.object">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.object">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.object">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.object">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.object">, !py.contract<"builtins.str">] -> [!py.contract<"typing.Any">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.object">, !py.contract<"builtins.str">, !py.contract<"typing.Any">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.object">, !py.contract<"builtins.str">] -> [!py.literal<None>]>,
      // ⛔ NOT A CPython DUNDER, and it is on object because that is the only
      // class whose values can be None without the type saying so. `x is None`
      // is a compile-time fold everywhere else -- None is a singleton, so a
      // concrete type is never it -- and an ERASED object is the one place the
      // fold is a lie. The emitter reaches this by name; a program that writes
      // `x.__ly_is_none__()` gets the same answer, which is the price of
      // routing it through the ordinary method table instead of a private op.
      !py.protocol<"Callable", [!py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>
    ],
    method_kinds = ["instance", "classmethod", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance"]
  } {}

  // ABI shape declarations are manifest entries: they describe the physical
  // runtime bundle for contracts whose structural signatures live elsewhere.
  func.func private @LyNone_Shape() attributes {ly.runtime.contract = "types.NoneType", ly.runtime.shape}

  func.func private @LyObject_Shape() -> memref<5xi64> attributes {ly.runtime.contract = "builtins.object", ly.runtime.shape}

  func.func @LyObject_Init(%header: memref<5xi64> {ly.ownership.object_header}) attributes {ly.runtime.class_id = 0 : i64, ly.runtime.contract = "builtins.object", ly.runtime.method = "__init__"} {
    func.return
  }

  func.func private @LyObject_ReleaseBoxedPayloadRaw(%box: memref<5xi64>)

  // Liveness pin: a no-op call whose operand keeps a box alive past calls
  // that consumed only its raw pointer words (the same device the container
  // paths use with __len__).
  // py.keep_alive: a use of an object's storage and nothing else, so the
  // reference that keeps it alive is released after the call site.
  func.func @LyObject_KeepAlive(%storage: memref<?xi64>) attributes {ly.runtime.contract = "builtins.object", ly.runtime.primitive = "keep_alive"} {
    func.return
  }

  func.func @LyObject_Touch(%box: memref<5xi64>) attributes {ly.runtime.contract = "builtins.object", ly.runtime.primitive = "touch"} {
    func.return
  }

  // "<object object at 0x"
  memref.global "private" constant @__ly_object_repr_prefix : memref<20xi8> = dense<[60, 111, 98, 106, 101, 99, 116, 32, 111, 98, 106, 101, 99, 116, 32, 97, 116, 32, 48, 120]>
  // "None"
  memref.global "private" constant @__ly_object_repr_none : memref<4xi8> = dense<[78, 111, 110, 101]>

  // repr of an erased object box: dispatch the payload class through the
  // boxed-repr hook; the None handle prints "None"; anything unhandled gets
  // the CPython default `<object object at 0x...>` form.
  func.func @LyObject_BoxedRepr(%box: memref<5xi64>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.object", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %zero = arith.constant 0 : i64
    %class_id = memref.load %box[%c1] : memref<5xi64>
    %entity = memref.load %box[%c2] : memref<5xi64>
    %class_zero = arith.cmpi eq, %class_id, %zero : i64
    cf.cond_br %class_zero, ^zero_class, ^try_hook

  ^zero_class:
    %entity_zero = arith.cmpi eq, %entity, %zero : i64
    cf.cond_br %entity_zero, ^none, ^default

  ^none:
    %none_static = memref.get_global @__ly_object_repr_none : memref<4xi8>
    %none_bytes = memref.cast %none_static : memref<4xi8> to memref<?xi8>
    %none_len = arith.constant 4 : i64
    %nh, %nb = func.call @__ly_unicode_from_valid_utf8(%none_bytes, %c0, %none_len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    cf.br ^done(%nh, %nb : memref<2xi64>, memref<?xi8>)

  ^try_hook:
    %box_idx = memref.extract_aligned_pointer_as_index %box : memref<5xi64> -> index
    %box_i64 = arith.index_cast %box_idx : index to i64
    %box_base_ptr = llvm.inttoptr %box_i64 : i64 to !llvm.ptr
    // The box's word 2 reads as a slot (BoxLayout.h).
    %box_ptr = llvm.getelementptr %box_base_ptr[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %hooked:3 = func.call @__ly_repr_boxed_by_contract(%box_ptr, %class_id) : (!llvm.ptr, i64) -> (memref<2xi64>, memref<?xi8>, i1)
    cf.cond_br %hooked#2, ^done(%hooked#0, %hooked#1 : memref<2xi64>, memref<?xi8>), ^default

  ^default:
    // ⭐ The BOX carries the class id in the same word an instance header does,
    // so the erased path names the real class too. It used to pass the static
    // "<object object at 0x" prefix, which is why an instance handed to an
    // `object` parameter printed <object object ...> where CPython prints its
    // class -- the same defect the typed path had, read from the box instead of
    // the header.
    %header_sub = memref.subview %box[0] [2] [1] : memref<5xi64> to memref<2xi64, strided<[1]>>
    %header_view = memref.cast %header_sub : memref<2xi64, strided<[1]>> to memref<2xi64, strided<[1], offset: ?>>
    %dh, %db = func.call @LyObject_DefaultReprDynamic(%header_view) : (memref<2xi64, strided<[1], offset: ?>>) -> (memref<2xi64>, memref<?xi8>)
    cf.br ^done(%dh, %db : memref<2xi64>, memref<?xi8>)

  ^done(%rh: memref<2xi64>, %rb: memref<?xi8>):
    func.return %rh, %rb : memref<2xi64>, memref<?xi8>
  }

  // ⛔ THE ly.runtime.* ATTRIBUTES ARE LOAD-BEARING, not decoration:
  // `isRuntimeManifestFunction` keys on them, and a helper without one is
  // treated as USER code by the refcount pass -- which then inserts a release
  // for the hook result on the path that does not use it. The hook's miss
  // returns ub.poison, so that release freed garbage and the program aborted in
  // malloc. Every hand-written helper that returns an OWNED result needs one.
  func.func private @__ly_repr_boxed_or_default(%box_ptr: !llvm.ptr, %class_id: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.object", ly.runtime.primitive = "repr_boxed_or_default", ly.runtime.result_contract = "builtins.str"} {
    %h, %b, %ok = func.call @__ly_repr_boxed_by_contract(%box_ptr, %class_id) : (!llvm.ptr, i64) -> (memref<2xi64>, memref<?xi8>, i1)
    cf.cond_br %ok, ^hooked, ^fallback

  ^hooked:
    func.return %h, %b : memref<2xi64>, memref<?xi8>

  ^fallback:
    %addr = llvm.ptrtoint %box_ptr : !llvm.ptr to i64
    %dh, %db = func.call @__ly_default_repr_dynamic_from_addr(%addr, %class_id) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %dh, %db : memref<2xi64>, memref<?xi8>
  }

  // str() of an erased object box (print's conversion): classes with a
  // boxed-conforming __str__ dispatch through the str hook (str prints
  // unquoted); everything else falls back to the repr form, matching
  // CPython where str(x) == repr(x) for the remaining builtins.
  func.func @LyObject_BoxedStr(%box: memref<5xi64>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.object", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %class_id = memref.load %box[%c1] : memref<5xi64>
    %class_zero = arith.cmpi eq, %class_id, %zero : i64
    cf.cond_br %class_zero, ^fallback, ^try_hook

  ^try_hook:
    %box_idx = memref.extract_aligned_pointer_as_index %box : memref<5xi64> -> index
    %box_i64 = arith.index_cast %box_idx : index to i64
    %box_base_ptr = llvm.inttoptr %box_i64 : i64 to !llvm.ptr
    // The box's word 2 reads as a slot (BoxLayout.h).
    %box_ptr = llvm.getelementptr %box_base_ptr[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %hooked:3 = func.call @__ly_str_boxed_by_contract(%box_ptr, %class_id) : (!llvm.ptr, i64) -> (memref<2xi64>, memref<?xi8>, i1)
    cf.cond_br %hooked#2, ^done(%hooked#0, %hooked#1 : memref<2xi64>, memref<?xi8>), ^fallback

  ^fallback:
    %fh, %fb = func.call @LyObject_BoxedRepr(%box) : (memref<5xi64>) -> (memref<2xi64>, memref<?xi8>)
    cf.br ^done(%fh, %fb : memref<2xi64>, memref<?xi8>)

  ^done(%rh: memref<2xi64>, %rb: memref<?xi8>):
    func.return %rh, %rb : memref<2xi64>, memref<?xi8>
  }

  // object.__eq__ / __ne__ / __hash__: the defaults every class inherits.
  // They forward to the SAME boxed dispatchers the dict/set key paths use
  // (__ly_box_equal / __ly_box_hash) rather than open-coding an address
  // compare, so `a == b` and "a and b land in the same dict slot" cannot
  // disagree -- and a subclass that does define __eq__/__hash__ is still
  // reached, because those dispatchers consult the per-class-id hook before
  // falling back to identity (CPython's object.__eq__ / object.__hash__).
  // None inside an erased box: sixteen zero words, so no payload class and no
  // entity. Every other value in a box has at least one of the two.
  func.func @LyObject_IsNone(%box: memref<5xi64>) -> i1 attributes {ly.runtime.contract = "builtins.object", ly.runtime.method = "__ly_is_none__"} {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %zero = arith.constant 0 : i64
    %class_id = memref.load %box[%c1] : memref<5xi64>
    %entity = memref.load %box[%c2] : memref<5xi64>
    %no_class = arith.cmpi eq, %class_id, %zero : i64
    %no_entity = arith.cmpi eq, %entity, %zero : i64
    %is_none = arith.andi %no_class, %no_entity : i1
    func.return %is_none : i1
  }

  func.func @LyObject_BoxedEq(%lhs: memref<5xi64>, %rhs: memref<5xi64>) -> i1 attributes {ly.runtime.contract = "builtins.object", ly.runtime.method = "__eq__"} {
    %lhs_idx = memref.extract_aligned_pointer_as_index %lhs : memref<5xi64> -> index
    %lhs_word = arith.index_cast %lhs_idx : index to i64
    %lhs_base = llvm.inttoptr %lhs_word : i64 to !llvm.ptr
    %lhs_ptr = llvm.getelementptr %lhs_base[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %rhs_idx = memref.extract_aligned_pointer_as_index %rhs : memref<5xi64> -> index
    %rhs_word = arith.index_cast %rhs_idx : index to i64
    %rhs_base = llvm.inttoptr %rhs_word : i64 to !llvm.ptr
    %rhs_ptr = llvm.getelementptr %rhs_base[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %eq = func.call @__ly_box_equal(%lhs_ptr, %rhs_ptr) : (!llvm.ptr, !llvm.ptr) -> i1
    func.return %eq : i1
  }

  // Derived from __eq__, not open-coded: CPython's object.__ne__ is the
  // negation of whatever __eq__ resolved to, so deriving it here keeps the
  // pair from disagreeing when a subclass supplies only __eq__.
  func.func @LyObject_BoxedNe(%lhs: memref<5xi64>, %rhs: memref<5xi64>) -> i1 attributes {ly.runtime.contract = "builtins.object", ly.runtime.method = "__ne__"} {
    %true = arith.constant true
    %eq = func.call @LyObject_BoxedEq(%lhs, %rhs) : (memref<5xi64>, memref<5xi64>) -> i1
    %ne = arith.xori %eq, %true : i1
    func.return %ne : i1
  }

  func.func @LyObject_BoxedHash(%box: memref<5xi64>) -> i64 attributes {ly.runtime.contract = "builtins.object", ly.runtime.method = "__hash__"} {
    %idx = memref.extract_aligned_pointer_as_index %box : memref<5xi64> -> index
    %word = arith.index_cast %idx : index to i64
    %base = llvm.inttoptr %word : i64 to !llvm.ptr
    %ptr = llvm.getelementptr %base[2] : (!llvm.ptr) -> !llvm.ptr, i64
    %hashed = func.call @__ly_box_hash(%ptr) : (!llvm.ptr) -> i64
    func.return %hashed : i64
  }

  func.func @LyObject_DecRef(%box: memref<5xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.object", ly.runtime.deallocator} {
    %storage = memref.cast %box : memref<5xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    // The box's word 2 is the slot it holds (BoxLayout.h); a transient box
    // is a slot from word 0, which is what LyObject_ReleaseBoxedPayloadRaw
    // takes.
    %box_idx = memref.extract_aligned_pointer_as_index %box : memref<5xi64> -> index
    %box_word = arith.index_cast %box_idx : index to i64
    %box_base = llvm.inttoptr %box_word : i64 to !llvm.ptr
    %held_slot = llvm.getelementptr %box_base[2] : (!llvm.ptr) -> !llvm.ptr, i64
    func.call @release_payload_slot_ptr(%held_slot) : (!llvm.ptr) -> ()
    memref.dealloc %box : memref<5xi64>
    cf.br ^done

  ^done:
    func.return
  }

  func.func @LyObject_DefaultRepr(%header: memref<2xi64, strided<[1], offset: ?>> {ly.ownership.object_header}, %prefix: memref<?xi8>, %prefix_len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.object", ly.runtime.primitive = "default_repr", ly.runtime.result_contract = "builtins.str"} {
    %ptr_index = memref.extract_aligned_pointer_as_index %header : memref<2xi64, strided<[1], offset: ?>> -> index
    %ptr = arith.index_cast %ptr_index : index to i64
    %result_header, %result_bytes = func.call @__ly_default_repr_from_addr(%ptr, %prefix, %prefix_len) : (i64, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result_header, %result_bytes : memref<2xi64>, memref<?xi8>
  }

  // The address-keyed core of the default object repr, callable from paths
  // that only hold a raw box pointer (the dict missing-key raise).
  func.func private @__ly_default_repr_from_addr(%ptr: i64, %prefix: memref<?xi8>, %prefix_len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.object", ly.runtime.primitive = "default_repr_addr", ly.runtime.result_contract = "builtins.str"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %sixteen = arith.constant 16 : i64
    %max_digits_index = arith.constant 16 : index
    %lower = arith.constant 0 : index
    %step = arith.constant 1 : index
    %ascii_zero = arith.constant 48 : i64
    %ascii_a_minus_ten = arith.constant 87 : i64
    %ascii_gt = arith.constant 62 : i8
    %counted:2 = scf.for %i = %lower to %max_digits_index step %step iter_args(%n = %ptr, %digits = %zero) -> (i64, i64) {
      %active = arith.cmpi ne, %n, %zero : i64
      %next_n_active = arith.divui %n, %sixteen : i64
      %next_n = arith.select %active, %next_n_active, %n : i1, i64
      %incremented = arith.addi %digits, %one : i64
      %next_digits = arith.select %active, %incremented, %digits : i1, i64
      scf.yield %next_n, %next_digits : i64, i64
    }
    %ptr_is_zero = arith.cmpi eq, %ptr, %zero : i64
    %hex_digits = arith.select %ptr_is_zero, %one, %counted#1 : i1, i64
    %body_len = arith.addi %hex_digits, %one : i64
    %total_len = arith.addi %prefix_len, %body_len : i64
    %total_len_index = arith.index_cast %total_len : i64 to index
    %prefix_len_index = arith.index_cast %prefix_len : i64 to index
    %hex_digits_index = arith.index_cast %hex_digits : i64 to index
    %buffer = memref.alloca(%total_len_index) : memref<?xi8>

    scf.for %i = %lower to %prefix_len_index step %step {
      %byte = memref.load %prefix[%i] : memref<?xi8>
      memref.store %byte, %buffer[%i] : memref<?xi8>
    }

    scf.for %i = %lower to %hex_digits_index step %step iter_args(%n = %ptr) -> (i64) {
      %digit = arith.remui %n, %sixteen : i64
      %ten = arith.constant 10 : i64
      %is_decimal = arith.cmpi ult, %digit, %ten : i64
      %decimal_ch = arith.addi %digit, %ascii_zero : i64
      %alpha_ch = arith.addi %digit, %ascii_a_minus_ten : i64
      %ch_i64 = arith.select %is_decimal, %decimal_ch, %alpha_ch : i1, i64
      %ch = arith.trunci %ch_i64 : i64 to i8
      %one_index = arith.constant 1 : index
      %last_digit = arith.subi %hex_digits_index, %one_index : index
      %offset = arith.subi %last_digit, %i : index
      %dest = arith.addi %prefix_len_index, %offset : index
      memref.store %ch, %buffer[%dest] : memref<?xi8>
      %next = arith.divui %n, %sixteen : i64
      scf.yield %next : i64
    }

    %suffix_pos = arith.addi %prefix_len_index, %hex_digits_index : index
    memref.store %ascii_gt, %buffer[%suffix_pos] : memref<?xi8>
    %start = arith.constant 0 : index
    %result_header, %result_bytes = func.call @__ly_unicode_from_valid_utf8(%buffer, %start, %total_len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %result_header, %result_bytes : memref<2xi64>, memref<?xi8>
  }

  func.func @Ly_IncRef(%header: memref<2xi64, strided<[1], offset: ?>> {ly.ownership.object_header}) attributes {ly.ownership.retain_args = [0], ly.runtime.primitive = "retain"} {
    %slot = arith.constant 0 : index
    %zero = arith.constant 0 : i64
    %immortal = arith.constant 9223372036854775807 : i64
    // Tagged fast path: a header whose aligned pointer has either low bit set
    // is an immediate (`__ly_slot_word_is_immediate`). It owns no memory and
    // must not be dereferenced at all.
    %ptr_index = memref.extract_aligned_pointer_as_index %header : memref<2xi64, strided<[1], offset: ?>> -> index
    %ptr_bits = arith.index_cast %ptr_index : index to i64
    %tag_mask = arith.constant 3 : i64
    %tag_zero = arith.constant 0 : i64
    %tag_bit = arith.andi %ptr_bits, %tag_mask : i64
    %is_tagged = arith.cmpi ne, %tag_bit, %tag_zero : i64
    cf.cond_br %is_tagged, ^done, ^probe

  ^probe:
    // Immortal fast path: immortality is fixed at object creation and never
    // changes, so a pre-RMW acquire read is a stable witness. Skipping the
    // RMW keeps immortal headers write-free, which lets constant literals
    // live in read-only sections.
    %observed = memref.load %header[%slot] {ly.atomic.ordering = "acquire", ly.atomic.role = "object.refcount.load"} : memref<2xi64, strided<[1], offset: ?>>
    %pre_immortal = arith.cmpi eq, %observed, %immortal : i64
    cf.cond_br %pre_immortal, ^done, ^mutate

  ^mutate:
    %previous = memref.generic_atomic_rmw %header[%slot] : memref<2xi64, strided<[1], offset: ?>> {
    ^bb0(%current : i64):
      %body_zero = arith.constant 0 : i64
      %body_one = arith.constant 1 : i64
      %body_immortal = arith.constant 9223372036854775807 : i64
      %body_positive = arith.cmpi sgt, %current, %body_zero : i64
      %body_immortal_check = arith.cmpi eq, %current, %body_immortal : i64
      %body_incremented = arith.addi %current, %body_one : i64
      %body_positive_next = arith.select %body_positive, %body_incremented, %current : i1, i64
      %body_next = arith.select %body_immortal_check, %current, %body_positive_next : i1, i64
      memref.atomic_yield %body_next : i64
    } {ly.atomic.ordering = "acq_rel", ly.atomic.retain_premise = "entry-borrowed", ly.atomic.role = "object.refcount.retain"}
    %is_immortal = arith.cmpi eq, %previous, %immortal : i64
    cf.cond_br %is_immortal, ^done, ^check_positive

  ^check_positive:
    %positive = arith.cmpi sgt, %previous, %zero : i64
    cf.assert %positive, "Ly_IncRef observed non-positive refcount"
    cf.br ^done

  ^done:
    func.return
  }

  func.func @LyObject_ReleaseStorageToZero(%storage: memref<?xi64>) -> i1 attributes {ly.runtime.contract = "builtins.object", ly.runtime.primitive = "release_to_zero"} {
    %slot = arith.constant 0 : index
    %zero = arith.constant 0 : i64
    %immortal = arith.constant 9223372036854775807 : i64
    // Immortal fast path: see Ly_IncRef. Immortal storage is never written,
    // so constant literal objects stay in read-only storage.
    %observed = memref.load %storage[%slot] {ly.atomic.ordering = "acquire", ly.atomic.role = "object.refcount.load"} : memref<?xi64>
    %pre_immortal = arith.cmpi eq, %observed, %immortal : i64
    cf.cond_br %pre_immortal, ^done, ^mutate

  ^mutate:
    %previous = memref.generic_atomic_rmw %storage[%slot] : memref<?xi64> {
    ^bb0(%current : i64):
      %body_zero = arith.constant 0 : i64
      %body_one = arith.constant 1 : i64
      %body_immortal = arith.constant 9223372036854775807 : i64
      %body_positive = arith.cmpi sgt, %current, %body_zero : i64
      %body_immortal_check = arith.cmpi eq, %current, %body_immortal : i64
      %body_decremented = arith.subi %current, %body_one : i64
      %body_positive_next = arith.select %body_positive, %body_decremented, %current : i1, i64
      %body_next = arith.select %body_immortal_check, %current, %body_positive_next : i1, i64
      memref.atomic_yield %body_next : i64
    } {ly.atomic.ordering = "acq_rel", ly.atomic.role = "object.refcount.release"}
    %is_immortal = arith.cmpi eq, %previous, %immortal : i64
    cf.cond_br %is_immortal, ^done, ^check_positive

  ^check_positive:
    %positive = arith.cmpi sgt, %previous, %zero : i64
    cf.assert %positive, "Ly_DecRef observed non-positive refcount"
    %one = arith.constant 1 : i64
    %became_zero = arith.cmpi eq, %previous, %one : i64
    func.return %became_zero : i1

  ^done:
    %false = arith.constant false
    func.return %false : i1
  }

  // ⭐ A WORD PAST AN ENTITY'S DECLARED HANDLE. Several contracts are more than
  // one physical value, and a payload box can only hand back what the FIRST
  // one names -- so each of them records its other lanes in words its handle
  // does not span. `__ly_unicode_alloc` does it with the shape word,
  // `LyBaseException_New` with extended word 6, and these are the same access
  // for the contracts that need no accessor of their own.
  //
  // Why NOT widen the handle instead: a word no signature mentions is a word
  // no reader is told about. (The handle's width was also once the contract's
  // identity; a deallocator is chosen by contract name (`findDeallocatorForValueGroup`), not by width.)
  func.func private @__ly_entity_word_set(%ptr: i64, %slot: i64, %value: i64) {
    %base = llvm.inttoptr %ptr : i64 to !llvm.ptr
    %slot_ptr = llvm.getelementptr %base[%slot] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %value, %slot_ptr : i64, !llvm.ptr
    func.return
  }

  func.func private @__ly_entity_word_get(%ptr: i64, %slot: i64) -> i64 {
    %base = llvm.inttoptr %ptr : i64 to !llvm.ptr
    %slot_ptr = llvm.getelementptr %base[%slot] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %value = llvm.load %slot_ptr : !llvm.ptr -> i64
    func.return %value : i64
  }
  // Rebuild a rank-1 memref over the payload a raw pointer word addresses
  // (buildGlobalViewFunction: allocated == aligned, offset 0, stride 1). The
  // manifest's only route to a descriptor -- writing the insertvalue chain and
  // the closing cast inline is rejected at the runtime-lowering input, because
  // builtin.unrealized_conversion_cast is that pass's own marker vocabulary.
  // One declaration per element type so the func-level signature type-checks;
  // narrow to a static shape with memref.cast at the call site.
  func.func private @__ly_global_view_i64(%pointer: i64, %size: i64) -> memref<?xi64>
  func.func private @__ly_global_view_i32(%pointer: i64, %size: i64) -> memref<?xi32>
  func.func private @__ly_global_view_f64(%pointer: i64, %size: i64) -> memref<?xf64>
  func.func private @__ly_global_view_i8(%pointer: i64, %size: i64) -> memref<?xi8>

  // Per-program class-name table (synthesized by the lowering, one entry per
  // class the program declares, keyed by the id its instances carry in header
  // word 1). Null for an id the program does not know.
  func.func private @__ly_source_class_name(%class_id: i64) -> !llvm.ptr

  // ⭐ `type(v).__name__` FOR A VALUE WHOSE STATIC CLASS IS NOT ITS OWN. The
  // header's word 1 is the class id -- the word `isinstance` reads -- so the
  // dynamic name is a table lookup, and the only thing that has to happen here
  // is turning a NUL-terminated pointer into a str.
  //
  // ⛔ A null pointer is not an error: it is an id this program never declared
  // (a manifest object reaching here), and "object" is what CPython would print
  // for it.
  func.func @LyObject_ClassNameFromId(%class_id: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.object", ly.runtime.primitive = "class_name_from_id", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero_i8 = arith.constant 0 : i8
    %cap = arith.constant 64 : index
    %name_ptr = func.call @__ly_source_class_name(%class_id) : (i64) -> !llvm.ptr
    %null = llvm.mlir.zero : !llvm.ptr
    %is_null = llvm.icmp "eq" %name_ptr, %null : !llvm.ptr
    cf.cond_br %is_null, ^unknown, ^known

  ^unknown:
    %fallback = memref.get_global @__ly_class_name_object : memref<6xi8>
    %fallback_dyn = memref.cast %fallback : memref<6xi8> to memref<?xi8>
    %fallback_len = arith.constant 6 : i64
    %fh, %fb = func.call @__ly_unicode_from_valid_utf8(%fallback_dyn, %c0, %fallback_len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %fh, %fb : memref<2xi64>, memref<?xi8>

  ^known:
    %buffer = memref.alloca() : memref<64xi8>
    %true_scan = arith.constant true
    %dot = arith.constant 46 : i8
    // ⛔ The table entry is the class's QUALIFIED name ("__main__.A",
    // "lib.Base") because the repr needs the module; `__name__` is always the
    // leaf, so the scan also carries where the last '.' left off.
    %scan:3 = scf.while (%i = %c0, %go = %true_scan, %start = %c0) : (index, i1, index) -> (index, i1, index) {
      %in_bounds = arith.cmpi ult, %i, %cap : index
      %continue = arith.andi %in_bounds, %go : i1
      scf.condition(%continue) %i, %go, %start : index, i1, index
    } do {
    ^bb0(%i: index, %go: i1, %start: index):
      %i_i64 = arith.index_cast %i : index to i64
      %slot = llvm.getelementptr %name_ptr[%i_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %byte = llvm.load %slot : !llvm.ptr -> i8
      %is_nul = arith.cmpi eq, %byte, %zero_i8 : i8
      %true_x = arith.constant true
      %not_nul = arith.xori %is_nul, %true_x : i1
      scf.if %not_nul {
        memref.store %byte, %buffer[%i] : memref<64xi8>
      }
      %next = arith.addi %i, %c1 : index
      %kept = arith.select %is_nul, %i, %next : index
      %is_dot = arith.cmpi eq, %byte, %dot : i8
      %next_start = arith.select %is_dot, %next, %start : index
      scf.yield %kept, %not_nul, %next_start : index, i1, index
    }
    %leaf_len_index = arith.subi %scan#0, %scan#2 : index
    %name_len = arith.index_cast %leaf_len_index : index to i64
    %name_dyn = memref.cast %buffer : memref<64xi8> to memref<?xi8>
    %nh, %nb = func.call @__ly_unicode_from_valid_utf8(%name_dyn, %scan#2, %name_len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %nh, %nb : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_class_name_object : memref<6xi8> = dense<[111, 98, 106, 101, 99, 116]>
  memref.global "private" constant @__ly_repr_prefix_plain : memref<1xi8> = dense<[60]>
  memref.global "private" constant @__ly_repr_suffix : memref<13xi8> = dense<[32, 111, 98, 106, 101, 99, 116, 32, 97, 116, 32, 48, 120]>

  // ⭐ THE DEFAULT REPR OF A VALUE WHOSE STATIC CLASS IS NOT ITS OWN. The
  // prefix used to be baked in at compile time from the static contract, so
  // `x: A = B(); print(x)` printed `<__main__.A object at ...>` where CPython
  // prints B -- a wrong answer with nothing to diagnose, since the address
  // differs anyway and no differential can see the class name. The id in header
  // word 1 says which class it really is.
  //
  // ⛔ An unknown id prints `<object object at ...>`, which is what CPython
  // prints for a bare object(): the table has an entry for every class the
  // program declares, so a miss means the value is not one of them.
  func.func @LyObject_DefaultReprDynamic(%header: memref<2xi64, strided<[1], offset: ?>> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.object", ly.runtime.primitive = "default_repr_dynamic", ly.runtime.result_contract = "builtins.str"} {
    %c1_slot = arith.constant 1 : index
    %ptr_index_outer = memref.extract_aligned_pointer_as_index %header : memref<2xi64, strided<[1], offset: ?>> -> index
    %ptr_outer = arith.index_cast %ptr_index_outer : index to i64
    %class_id_outer = memref.load %header[%c1_slot] : memref<2xi64, strided<[1], offset: ?>>
    %h_outer, %b_outer = func.call @__ly_default_repr_dynamic_from_addr(%ptr_outer, %class_id_outer) : (i64, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %h_outer, %b_outer : memref<2xi64>, memref<?xi8>
  }

  // The address-keyed core, callable from paths that hold only a raw box
  // pointer (a container rendering its elements).
  func.func private @__ly_default_repr_dynamic_from_addr(%ptr: i64, %class_id: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.object", ly.runtime.primitive = "default_repr_dynamic_addr", ly.runtime.result_contract = "builtins.str"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    %c13 = arith.constant 13 : index
    %c64 = arith.constant 64 : index
    %zero_i8 = arith.constant 0 : i8
    %name_ptr = func.call @__ly_source_class_name(%class_id) : (i64) -> !llvm.ptr
    %buffer = memref.alloca() : memref<128xi8>
    %null = llvm.mlir.zero : !llvm.ptr
    %is_null = llvm.icmp "eq" %name_ptr, %null : !llvm.ptr
    cf.cond_br %is_null, ^unknown, ^known

  ^unknown:
    %plain = memref.get_global @__ly_repr_prefix_plain : memref<1xi8>
    %plain_byte = memref.load %plain[%c0] : memref<1xi8>
    memref.store %plain_byte, %buffer[%c0] : memref<128xi8>
    %object_name = memref.get_global @__ly_class_name_object : memref<6xi8>
    scf.for %i = %c0 to %c6 step %c1 {
      %byte = memref.load %object_name[%i] : memref<6xi8>
      %slot = arith.addi %i, %c1 : index
      memref.store %byte, %buffer[%slot] : memref<128xi8>
    }
    %unknown_len = arith.constant 7 : index
    cf.br ^suffix(%unknown_len : index)

  ^known:
    // ⛔ Only "<": the table entry is the qualified name, so pasting
    // "__main__." here printed "<__main__.lib.Base object at ...>" for a class
    // imported from another source module.
    %open = memref.get_global @__ly_repr_prefix_plain : memref<1xi8>
    %open_byte = memref.load %open[%c0] : memref<1xi8>
    memref.store %open_byte, %buffer[%c0] : memref<128xi8>
    %true_scan = arith.constant true
    %scan:2 = scf.while (%i = %c0, %go = %true_scan) : (index, i1) -> (index, i1) {
      %in_bounds = arith.cmpi ult, %i, %c64 : index
      %continue = arith.andi %in_bounds, %go : i1
      scf.condition(%continue) %i, %go : index, i1
    } do {
    ^bb0(%i: index, %go: i1):
      %i_i64 = arith.index_cast %i : index to i64
      %slot = llvm.getelementptr %name_ptr[%i_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %byte = llvm.load %slot : !llvm.ptr -> i8
      %is_nul = arith.cmpi eq, %byte, %zero_i8 : i8
      %true_x = arith.constant true
      %not_nul = arith.xori %is_nul, %true_x : i1
      scf.if %not_nul {
        %at = arith.addi %i, %c1 : index
        memref.store %byte, %buffer[%at] : memref<128xi8>
      }
      %next = arith.addi %i, %c1 : index
      %kept = arith.select %is_nul, %i, %next : index
      scf.yield %kept, %not_nul : index, i1
    }
    %known_len = arith.addi %scan#0, %c1 : index
    cf.br ^suffix(%known_len : index)

  ^suffix(%len: index):
    %tail = memref.get_global @__ly_repr_suffix : memref<13xi8>
    scf.for %i = %c0 to %c13 step %c1 {
      %byte = memref.load %tail[%i] : memref<13xi8>
      %slot = arith.addi %i, %len : index
      memref.store %byte, %buffer[%slot] : memref<128xi8>
    }
    %total_index = arith.addi %len, %c13 : index
    %total = arith.index_cast %total_index : index to i64
    %buffer_dyn = memref.cast %buffer : memref<128xi8> to memref<?xi8>
    %h, %b = func.call @__ly_default_repr_from_addr(%ptr, %buffer_dyn, %total) : (i64, memref<?xi8>, i64) -> (memref<2xi64>, memref<?xi8>)
    func.return %h, %b : memref<2xi64>, memref<?xi8>
  }

  // ===== immediates in a slot's entity word =====
  //
  // A slot (and a standalone `object` box) of class int or float may hold the
  // value itself in its entity word instead of the address of an object: bit 0
  // set says so, and every retain and release already skips such a word, so a
  // slot that holds one owns nothing. Which decoding applies is the class
  // word's: an int is `v << 1 | 1` for v in [-2^62, 2^62), a float is the
  // rotated encoding below. Values outside those ranges stay objects.
  //
  // ⛔ Not every int: a 64-bit value needs a bit the tag takes. CPython puts
  // every int in an object; here only the ones past 2^62 are (2^30 on a
  // 32-bit target, `__ly_addresses_are_word_wide`).
  func.func private @__ly_slot_word_is_immediate(%word: i64) -> i1 {
    %mask = arith.constant 3 : i64
    %zero = arith.constant 0 : i64
    %tag = arith.andi %word, %mask : i64
    %is = arith.cmpi ne, %tag, %zero : i64
    func.return %is : i1
  }

  // The class a slot word names: 0 for None (the word 0), int or float for an
  // immediate by its tag, and otherwise the class id every object keeps in
  // its header's word 1.
  func.func private @__ly_slot_class(%word: i64) -> i64 {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %tag = arith.andi %word, %three : i64
    %is_object = arith.cmpi eq, %tag, %zero : i64
    %is_null = arith.cmpi eq, %word, %zero : i64
    %int_tag = arith.andi %word, %one : i64
    %is_int = arith.cmpi ne, %int_tag, %zero : i64
    %immediate_class = arith.select %is_int, %one, %two : i1, i64
    %class = scf.if %is_object -> (i64) {
      %object_class = scf.if %is_null -> (i64) {
        scf.yield %zero : i64
      } else {
        %ptr = llvm.inttoptr %word : i64 to !llvm.ptr
        %class_gep = llvm.getelementptr %ptr[1] : (!llvm.ptr) -> !llvm.ptr, i64
        %loaded = llvm.load %class_gep : !llvm.ptr -> i64
        scf.yield %loaded : i64
      }
      scf.yield %object_class : i64
    } else {
      scf.yield %immediate_class : i64
    }
    func.return %class : i64
  }

  // What `is` compares for a value, given the entity word a box or a slot keeps
  // for it: the word a SLOT keeps for it. A slot keeps an int or a float that
  // has an immediate as that immediate, and anything else as its address; a
  // box keeps the address of the object it was made around. So one object
  // reads as its immediate in a list and as its address in a box, and `is`
  // between the two said False -- this names both by the slot's word.
  // ⛔ Not the value for every int and float: a float with no immediate (a NaN,
  // an infinity) keeps an object of its own in a slot too, and two of them
  // are two objects -- `nan in [nan * 1.0]` is False in CPython.
  func.func @LyObject_IdentityKey(%word: i64) -> i64 attributes {ly.runtime.contract = "builtins.object", ly.runtime.primitive = "identity_key"} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %three = arith.constant 3 : i64
    %immediate = func.call @__ly_slot_word_is_immediate(%word) : (i64) -> i1
    %is_null = arith.cmpi eq, %word, %zero : i64
    %plain = arith.ori %immediate, %is_null : i1
    %key = scf.if %plain -> (i64) {
      scf.yield %word : i64
    } else {
      %class = func.call @__ly_slot_class(%word) : (i64) -> i64
      %is_int = arith.cmpi eq, %class, %one : i64
      %is_float = arith.cmpi eq, %class, %two : i64
      %canonical = scf.if %is_int -> (i64) {
        %view = func.call @__ly_global_view_i64(%word, %two) : (i64, i64) -> memref<?xi64>
        %header = memref.cast %view : memref<?xi64> to memref<2xi64>
        %value, %fits = func.call @LyLong_TryAsI64(%header) : (memref<2xi64>) -> (i64, i1)
        %small = func.call @__ly_int_immediate_fits(%value) : (i64) -> i1
        %encodable = arith.andi %fits, %small : i1
        %int_key = scf.if %encodable -> (i64) {
          %w = func.call @__ly_int_to_immediate(%value) : (i64) -> i64
          scf.yield %w : i64
        } else {
          scf.yield %word : i64
        }
        scf.yield %int_key : i64
      } else {
        %float_key = scf.if %is_float -> (i64) {
          %view = func.call @__ly_global_view_i64(%word, %three) : (i64, i64) -> memref<?xi64>
          %bits_slot = arith.constant 2 : index
          %bits = memref.load %view[%bits_slot] : memref<?xi64>
          %fits = func.call @__ly_float_immediate_fits(%bits) : (i64) -> i1
          %fkey = scf.if %fits -> (i64) {
            %w = func.call @__ly_float_to_immediate(%bits) : (i64) -> i64
            scf.yield %w : i64
          } else {
            scf.yield %word : i64
          }
          scf.yield %fkey : i64
        } else {
          scf.yield %word : i64
        }
        scf.yield %float_key : i64
      }
      scf.yield %canonical : i64
    }
    func.return %key : i64
  }

  // True when an address is as wide as a slot word. On a 32-bit target
  // (wasm32, armv7) a word that becomes a view is cut to the address's width
  // on the way, so an immediate has to survive that: there an int is
  // immediate only within 31 bits and a float never is. Folds to a constant.
  // ⛔ Asked of a POINTER, not of `index`: wasm32 lowers `index` to 64 bits
  // and its pointers to 32, and it is the pointer the word passes through.
  func.func private @__ly_addresses_are_word_wide() -> i1 {
    %probe = arith.constant 1099511627776 : i64
    %as_ptr = llvm.inttoptr %probe : i64 to !llvm.ptr
    %back = llvm.ptrtoint %as_ptr : !llvm.ptr to i64
    %wide = arith.cmpi eq, %back, %probe : i64
    func.return %wide : i1
  }

  // The entity word a slot view was built from, read back off the view. On a
  // 32-bit target the view kept the low half, zero-extended; an immediate
  // there is a sign-extended 32-bit word, so the sign is put back.
  func.func private @__ly_slot_word_from_view_address(%address: i64) -> i64 {
    %wide = func.call @__ly_addresses_are_word_wide() : () -> i1
    %thirty_two = arith.constant 32 : i64
    %high = arith.shli %address, %thirty_two : i64
    %narrowed = arith.shrsi %high, %thirty_two : i64
    %one = arith.constant 1 : i64
    %tag = arith.andi %address, %one : i64
    %immediate = arith.cmpi eq, %tag, %one : i64
    %true = arith.constant true
    %narrow = arith.xori %wide, %true : i1
    %narrow_imm = arith.andi %immediate, %narrow : i1
    %word = arith.select %narrow_imm, %narrowed, %address : i1, i64
    func.return %word : i64
  }

  // Uniform per-element hash/eq dispatch, generated per program by the
  // lowering (class id -> the manifest __hash__ / __eq__); resolved at link.
  func.func private @__ly_hash_boxed_by_contract(%box: !llvm.ptr, %class_id: i64) -> (i64, i1)
  func.func private @__ly_eq_boxed_by_contract(%lhs: !llvm.ptr, %rhs: !llvm.ptr, %class_id: i64, %rhs_class_id: i64) -> (i1, i1)

  // "unhashable type: '<name>'"
  memref.global "private" constant @__ly_hash_msg_unhashable_list : memref<23xi8> = dense<[117, 110, 104, 97, 115, 104, 97, 98, 108, 101, 32, 116, 121, 112, 101, 58, 32, 39, 108, 105, 115, 116, 39]>
  memref.global "private" constant @__ly_hash_msg_unhashable_dict : memref<23xi8> = dense<[117, 110, 104, 97, 115, 104, 97, 98, 108, 101, 32, 116, 121, 112, 101, 58, 32, 39, 100, 105, 99, 116, 39]>
  memref.global "private" constant @__ly_hash_msg_unhashable_set : memref<22xi8> = dense<[117, 110, 104, 97, 115, 104, 97, 98, 108, 101, 32, 116, 121, 112, 101, 58, 32, 39, 115, 101, 116, 39]>

  func.func private @__ly_hash_raise_unhashable(%class_id: i64) {
    %type_error = arith.constant 52 : i64
    %c10 = arith.constant 10 : i64
    %c12 = arith.constant 12 : i64
    %is_list = arith.cmpi eq, %class_id, %c10 : i64
    %is_dict = arith.cmpi eq, %class_id, %c12 : i64
    scf.if %is_list {
      %msg_static = memref.get_global @__ly_hash_msg_unhashable_list : memref<23xi8>
      %msg = memref.cast %msg_static : memref<23xi8> to memref<?xi8>
      %len = arith.constant 23 : i64
      func.call @__ly_raise_static_message(%type_error, %msg, %len) : (i64, memref<?xi8>, i64) -> ()
    } else {
      scf.if %is_dict {
        %msg_static = memref.get_global @__ly_hash_msg_unhashable_dict : memref<23xi8>
        %msg = memref.cast %msg_static : memref<23xi8> to memref<?xi8>
        %len = arith.constant 23 : i64
        func.call @__ly_raise_static_message(%type_error, %msg, %len) : (i64, memref<?xi8>, i64) -> ()
      } else {
        %msg_static = memref.get_global @__ly_hash_msg_unhashable_set : memref<22xi8>
        %msg = memref.cast %msg_static : memref<22xi8> to memref<?xi8>
        %len = arith.constant 22 : i64
        func.call @__ly_raise_static_message(%type_error, %msg, %len) : (i64, memref<?xi8>, i64) -> ()
      }
    }
    func.return
  }

  // Hash of an arbitrary boxed value (16-word payload handle). Dispatches on
  // the class id: singletons inline, manifest/user `__hash__` through the
  // generated hook, identity hash for classes without `__hash__` (R6), and a
  // TypeError for the builtin mutable containers.
  // ===== the payload box's layout, stated once =====
  //
  // ⭐ EVERY BOX ACCESS COMES THROUGH THESE. The width and the word offsets
  // used to be literals in each function that touched a box -- 86 `slot * 16`
  // strides and some ninety word indices -- mixed in with unrelated 16s (hex
  // conversion, siphash, `__ly_long_parts`'s +16 byte offset). Narrowing the
  // box then meant classifying all of them by hand, and the type system checks
  // none of it: a store to the wrong word of a right-sized box verifies
  // cleanly and corrupts a refcount at run time. That is not hypothetical --
  // it is what a first attempt did, and `__ly_unicode_item_words` reading the
  // second lane's size at word 10 was one of the sites it missed.
  //
  // ⛔ These have to be INLINE or they are a regression: the JIT's default is
  // `-jit-opt=0`, where LLVM inlines nothing except what is marked, and a call
  // per box word would land in `__ly_slot_less` and the container copies.
  // `markBoxLayoutHelpersAlwaysInline` (driver/lib/LLVMFinalize.cpp) sets the
  // attribute, and LLVM's AlwaysInliner runs at every optimisation level.
  //
  // The layout itself is ABI/BoxLayout.h, and `ManifestBoxLayoutTest` checks
  // that these five agree with it.
  func.func private @__ly_box_word_count() -> i64 {
    %words = arith.constant 1 : i64
    func.return %words : i64
  }

  // A standalone `object` box: refcount, class id, entity, and two words the
  // box does not use (BoxLayout.h, kStandaloneBoxWords). A pointer to its
  // word 2 reads as a slot.
  func.func private @__ly_box_standalone_word_count() -> i64 {
    %words = arith.constant 5 : i64
    func.return %words : i64
  }

  func.func private @__ly_box_slot_base(%slot: i64) -> i64 {
    %words = func.call @__ly_box_word_count() : () -> i64
    %base = arith.muli %slot, %words : i64
    func.return %base : i64
  }

  func.func private @__ly_box_slot_base_index(%slot: index) -> index {
    %slot_i64 = arith.index_cast %slot : index to i64
    %base = func.call @__ly_box_slot_base(%slot_i64) : (i64) -> i64
    %base_index = arith.index_cast %base : i64 to index
    func.return %base_index : index
  }

  // The one address a box holds.
  func.func private @__ly_box_entity_word(%base: i64) -> i64 {
    %entity = arith.constant 0 : i64
    %word = arith.addi %base, %entity : i64
    func.return %word : i64
  }

  // ⭐ THE WHOLE BOX, because a box IS an entity and its bookkeeping now. Four
  // writers used to spell this out and each of them zeroed the lanes first;
  // there are no lanes to zero and no lane words that can disagree with the
  // block they described.
  // The box owns its entity exactly when the entity is an address: not 0
  // (None) and not an immediate (bit 0 set). There is no flag to disagree.
  // ⛔ %class_id is not stored: a slot's class is its entity's
  // (`__ly_slot_class`). The callers name it so they read as what they store.
  func.func private @__ly_box_store_entity(%items: memref<?xi64>, %slot: i64, %class_id: i64, %entity: i64) {
    %base_i64 = func.call @__ly_box_slot_base(%slot) : (i64) -> i64
    %entity_word = func.call @__ly_box_entity_word(%base_i64) : (i64) -> i64
    %entity_slot = arith.index_cast %entity_word : i64 to index
    memref.store %entity, %items[%entity_slot] : memref<?xi64>
    func.return
  }

  func.func private @__ly_box_hash(%box: !llvm.ptr) -> i64 {
    %zero = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %slot_word = llvm.load %box : !llvm.ptr -> i64
    %class_id = func.call @__ly_slot_class(%slot_word) : (i64) -> i64
    %is_none = arith.cmpi eq, %class_id, %zero : i64
    %result = scf.if %is_none -> (i64) {
      // hash(None): the CPython 3.12+ constant.
      %none_hash = arith.constant 4238894112 : i64
      scf.yield %none_hash : i64
    } else {
      %c10 = arith.constant 10 : i64
      %c12 = arith.constant 12 : i64
      %c21 = arith.constant 21 : i64
      %is_list = arith.cmpi eq, %class_id, %c10 : i64
      %is_dict = arith.cmpi eq, %class_id, %c12 : i64
      %is_set = arith.cmpi eq, %class_id, %c21 : i64
      %mut0 = arith.ori %is_list, %is_dict : i1
      %unhashable = arith.ori %mut0, %is_set : i1
      scf.if %unhashable {
        func.call @__ly_hash_raise_unhashable(%class_id) : (i64) -> ()
      }
      // ⭐ An immediate int hashes from its value, with no object made for
      // the hook to read: CPython's long_hash, v mod (2^61 - 1) with the sign
      // carried over, on a value that fits a word.
      %entity0 = llvm.load %box : !llvm.ptr -> i64
      %int_class = arith.constant 1 : i64
      %is_int = arith.cmpi eq, %class_id, %int_class : i64
      %is_immediate = func.call @__ly_slot_word_is_immediate(%entity0) : (i64) -> i1
      %int_immediate = arith.andi %is_int, %is_immediate : i1
      %h, %handled = scf.if %int_immediate -> (i64, i1) {
        %v = func.call @__ly_int_from_immediate(%entity0) : (i64) -> i64
        %modulus = arith.constant 2305843009213693951 : i64
        %negative = arith.cmpi slt, %v, %zero : i64
        %negated = arith.subi %zero, %v : i64
        %magnitude = arith.select %negative, %negated, %v : i1, i64
        %reduced = arith.remui %magnitude, %modulus : i64
        %neg_reduced = arith.subi %zero, %reduced : i64
        %signed = arith.select %negative, %neg_reduced, %reduced : i1, i64
        %true_h = arith.constant true
        scf.yield %signed, %true_h : i64, i1
      } else {
        %hh, %hd = func.call @__ly_hash_boxed_by_contract(%box, %class_id) : (!llvm.ptr, i64) -> (i64, i1)
        scf.yield %hh, %hd : i64, i1
      }
      %dispatched = scf.if %handled -> (i64) {
        %fixed = func.call @__ly_hash_fixup(%h) : (i64) -> i64
        scf.yield %fixed : i64
      } else {
        // Identity hash: the CPython pointer hash (rotate right by 4).
        %p = llvm.load %box : !llvm.ptr -> i64
        %c4 = arith.constant 4 : i64
        %c60 = arith.constant 60 : i64
        %lo = arith.shrui %p, %c4 : i64
        %hi = arith.shli %p, %c60 : i64
        %rot = arith.ori %lo, %hi : i64
        %fixed = func.call @__ly_hash_fixup(%rot) : (i64) -> i64
        scf.yield %fixed : i64
      }
      scf.yield %dispatched : i64
    }
    func.return %result : i64
  }

  // Raw bigint view words of a boxed int: (sign, digit count, digits ptr).
  // Reached through the ENTITY word rather than lane pointer words, for the
  // reason `__ly_boxed_float_value` gives: `builtins.int` is one lane (the
  // header), and meta/digits are interior at entity +16 and +32 of the same
  // block. Box words 5 and 6 were lanes 1 and 2, which no longer exist.
  //
  // An immediate entity has no digits to point at, so its three 30-bit digits
  // are written into %scratch (three i32s the caller owns, live as long as it
  // reads the view) and the view points there.
  func.func private @__ly_boxed_long_view(%box: !llvm.ptr, %scratch: memref<3xi32>) -> (i64, i64, i64) {
    %c1 = arith.constant 1 : i64
    %c2 = arith.constant 2 : i64
    %c4 = arith.constant 4 : i64
    %entity_word = llvm.load %box : !llvm.ptr -> i64
    %immediate = func.call @__ly_slot_word_is_immediate(%entity_word) : (i64) -> i1
    %sign, %count, %digits_word = scf.if %immediate -> (i64, i64, i64) {
      %value = func.call @__ly_int_from_immediate(%entity_word) : (i64) -> i64
      %zero = arith.constant 0 : i64
      %minus_one = arith.constant -1 : i64
      %three = arith.constant 3 : i64
      %negative = arith.cmpi slt, %value, %zero : i64
      %is_zero = arith.cmpi eq, %value, %zero : i64
      %signed = arith.select %negative, %minus_one, %c1 : i1, i64
      %sign_v = arith.select %is_zero, %zero, %signed : i1, i64
      // |v| < 2^62 for an immediate, so the negation cannot overflow.
      %negated = arith.subi %zero, %value : i64
      %magnitude = arith.select %negative, %negated, %value : i1, i64
      %mask = arith.constant 1073741823 : i64
      %thirty = arith.constant 30 : i64
      %sixty = arith.constant 60 : i64
      %d0 = arith.andi %magnitude, %mask : i64
      %s1 = arith.shrui %magnitude, %thirty : i64
      %d1 = arith.andi %s1, %mask : i64
      %d2 = arith.shrui %magnitude, %sixty : i64
      %i0 = arith.constant 0 : index
      %i1 = arith.constant 1 : index
      %i2 = arith.constant 2 : index
      %d0_32 = arith.trunci %d0 : i64 to i32
      %d1_32 = arith.trunci %d1 : i64 to i32
      %d2_32 = arith.trunci %d2 : i64 to i32
      memref.store %d0_32, %scratch[%i0] : memref<3xi32>
      memref.store %d1_32, %scratch[%i1] : memref<3xi32>
      memref.store %d2_32, %scratch[%i2] : memref<3xi32>
      %has1 = arith.cmpi ne, %d1, %zero : i64
      %has2 = arith.cmpi ne, %d2, %zero : i64
      %one_or_two = arith.select %has1, %c2, %c1 : i1, i64
      %nonzero_count = arith.select %has2, %three, %one_or_two : i1, i64
      %count_v = arith.select %is_zero, %zero, %nonzero_count : i1, i64
      %scratch_idx = memref.extract_aligned_pointer_as_index %scratch : memref<3xi32> -> index
      %scratch_word = arith.index_cast %scratch_idx : index to i64
      scf.yield %sign_v, %count_v, %scratch_word : i64, i64, i64
    } else {
      %entity = llvm.inttoptr %entity_word : i64 to !llvm.ptr
      %meta_ptr = llvm.getelementptr %entity[%c2] : (!llvm.ptr, i64) -> !llvm.ptr, i64
      %digits_ptr = llvm.getelementptr %entity[%c4] : (!llvm.ptr, i64) -> !llvm.ptr, i64
      %sign_o = llvm.load %meta_ptr : !llvm.ptr -> i64
      %count_gep = llvm.getelementptr %meta_ptr[%c1] : (!llvm.ptr, i64) -> !llvm.ptr, i64
      %count_o = llvm.load %count_gep : !llvm.ptr -> i64
      %digits_o = llvm.ptrtoint %digits_ptr : !llvm.ptr to i64
      scf.yield %sign_o, %count_o, %digits_o : i64, i64, i64
    }
    func.return %sign, %count, %digits_word : i64, i64, i64
  }

  // Exact equality of a raw bigint view against an f64 (CPython's mixed
  // int/float comparison semantics: exact, no rounding).
  func.func private @__ly_boxed_long_eq_f64(%sign: i64, %count: i64, %digits_word: i64, %value: f64) -> i1 {
    %zero = arith.constant 0 : i64
    %false = arith.constant false
    %bits = arith.bitcast %value : f64 to i64
    %c52 = arith.constant 52 : i64
    %c63 = arith.constant 63 : i64
    %exp_mask = arith.constant 2047 : i64
    %mant_mask = arith.constant 4503599627370495 : i64
    %implicit_bit = arith.constant 4503599627370496 : i64
    %exp_field = arith.shrui %bits, %c52 : i64
    %exp_bits = arith.andi %exp_field, %exp_mask : i64
    %mant_bits = arith.andi %bits, %mant_mask : i64
    %sign_bit = arith.shrui %bits, %c63 : i64
    // NaN / inf never equal an int.
    %is_special = arith.cmpi eq, %exp_bits, %exp_mask : i64
    %result = scf.if %is_special -> (i1) {
      scf.yield %false : i1
    } else {
      %mant_zero = arith.cmpi eq, %mant_bits, %zero : i64
      %exp_zero = arith.cmpi eq, %exp_bits, %zero : i64
      %is_zero = arith.andi %exp_zero, %mant_zero : i1
      %zero_case = scf.if %is_zero -> (i1) {
        %int_zero = arith.cmpi eq, %sign, %zero : i64
        scf.yield %int_zero : i1
      } else {
        // Sign must match a nonzero float.
        %float_neg = arith.cmpi ne, %sign_bit, %zero : i64
        %neg_one = arith.constant -1 : i64
        %one = arith.constant 1 : i64
        %expected_sign = arith.select %float_neg, %neg_one, %one : i1, i64
        %sign_matches = arith.cmpi eq, %sign, %expected_sign : i64
        %signed_case = scf.if %sign_matches -> (i1) {
          // Normalize to mant * 2^exp with mant integral (53 bits max).
          %bias = arith.constant 1075 : i64
          %subnormal = arith.cmpi eq, %exp_bits, %zero : i64
          %norm_mant = arith.ori %mant_bits, %implicit_bit : i64
          %mant0 = arith.select %subnormal, %mant_bits, %norm_mant : i1, i64
          %norm_exp = arith.subi %exp_bits, %bias : i64
          %sub_exp = arith.subi %one, %bias : i64
          %exp0 = arith.subi %norm_exp, %zero : i64
          %exp1 = arith.select %subnormal, %sub_exp, %exp0 : i1, i64
          // Strip trailing zero bits of the mantissa into the exponent so
          // (mant, exp) is canonical: mant odd.
          %canon:2 = scf.while (%m = %mant0, %e = %exp1) : (i64, i64) -> (i64, i64) {
            %one_bit = arith.andi %m, %one : i64
            %even = arith.cmpi eq, %one_bit, %zero : i64
            scf.condition(%even) %m, %e : i64, i64
          } do {
          ^bb0(%m: i64, %e: i64):
            %half = arith.shrui %m, %one : i64
            %inc = arith.addi %e, %one : i64
            scf.yield %half, %inc : i64, i64
          }
          // A negative exponent means a fractional value: never an int.
          %exp_neg = arith.cmpi slt, %canon#1, %zero : i64
          %int_case = scf.if %exp_neg -> (i1) {
            scf.yield %false : i1
          } else {
            // Compare mant << exp against the digit magnitude exactly:
            // walk the 30-bit limbs from most significant, matching the
            // corresponding float bits (the float value has at most 53
            // significant bits; everything below them must be zero).
            // bit_length(int) must equal bit_length(mant) + exp.
            %c30 = arith.constant 30 : i64
            %digits_ptr = llvm.inttoptr %digits_word : i64 to !llvm.ptr
            %count_minus_one = arith.subi %count, %one : i64
            %top_gep = llvm.getelementptr %digits_ptr[%count_minus_one] : (!llvm.ptr, i64) -> !llvm.ptr, i32
            %top_i32 = llvm.load %top_gep : !llvm.ptr -> i32
            %top = arith.extui %top_i32 : i32 to i64
            // bit width of the top limb.
            %top_bits = scf.while (%w = %zero) : (i64) -> i64 {
              %shifted = arith.shrui %top, %w : i64
              %nonzero = arith.cmpi ne, %shifted, %zero : i64
              scf.condition(%nonzero) %w : i64
            } do {
            ^bb0(%w: i64):
              %next = arith.addi %w, %one : i64
              scf.yield %next : i64
            }
            %lower_limbs = arith.subi %count, %one : i64
            %lower_bits = arith.muli %lower_limbs, %c30 : i64
            %int_bl = arith.addi %top_bits, %lower_bits : i64
            // bit width of mant.
            %mant_bl = scf.while (%w = %zero) : (i64) -> i64 {
              %shifted = arith.shrui %canon#0, %w : i64
              %nonzero = arith.cmpi ne, %shifted, %zero : i64
              scf.condition(%nonzero) %w : i64
            } do {
            ^bb0(%w: i64):
              %next = arith.addi %w, %one : i64
              scf.yield %next : i64
            }
            %float_bl = arith.addi %mant_bl, %canon#1 : i64
            %bl_matches = arith.cmpi eq, %int_bl, %float_bl : i64
            %bl_case = scf.if %bl_matches -> (i1) {
              // Walk limbs from most significant: each limb must equal the
              // matching 30-bit window of mant << exp. Window position of
              // limb i (0-based from least significant): bits [30i, 30i+30).
              // mant occupies bits [exp, exp+mant_bl).
              %true = arith.constant true
              %c0_index = arith.constant 0 : index
              %count_index = arith.index_cast %count : i64 to index
              %c1_index = arith.constant 1 : index
              %all_match = scf.for %li = %c0_index to %count_index step %c1_index iter_args(%ok = %true) -> (i1) {
                %li_i64 = arith.index_cast %li : index to i64
                %limb_gep = llvm.getelementptr %digits_ptr[%li_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i32
                %limb_i32 = llvm.load %limb_gep : !llvm.ptr -> i32
                %limb = arith.extui %limb_i32 : i32 to i64
                %win_lo = arith.muli %li_i64, %c30 : i64
                // expected = (mant << exp) >> win_lo, masked to 30 bits =
                // mant >> (win_lo - exp) when win_lo >= exp, else
                // mant << (exp - win_lo).
                %sh = arith.subi %win_lo, %canon#1 : i64
                %sh_neg = arith.cmpi slt, %sh, %zero : i64
                %c64 = arith.constant 64 : i64
                %sh_big = arith.cmpi sge, %sh, %c64 : i64
                %sh_safe = arith.select %sh_big, %c63, %sh : i1, i64
                %down = arith.shrui %canon#0, %sh_safe : i64
                %neg_sh = arith.subi %zero, %sh : i64
                %neg_big = arith.cmpi sge, %neg_sh, %c64 : i64
                %neg_safe = arith.select %neg_big, %c63, %neg_sh : i1, i64
                %up = arith.shli %canon#0, %neg_safe : i64
                %up_sel = arith.select %neg_big, %zero, %up : i1, i64
                %down_sel = arith.select %sh_big, %zero, %down : i1, i64
                %shifted = arith.select %sh_neg, %up_sel, %down_sel : i1, i64
                %window_mask = arith.constant 1073741823 : i64
                %expected = arith.andi %shifted, %window_mask : i64
                %limb_matches = arith.cmpi eq, %limb, %expected : i64
                %next = arith.andi %ok, %limb_matches : i1
                scf.yield %next : i1
              }
              scf.yield %all_match : i1
            } else {
              scf.yield %false : i1
            }
            scf.yield %bl_case : i1
          }
          scf.yield %int_case : i1
        } else {
          scf.yield %false : i1
        }
        scf.yield %signed_case : i1
      }
      scf.yield %zero_case : i1
    }
    func.return %result : i1
  }

  // Equality of two arbitrary boxed values with CPython's dict/set probe
  // semantics: identity implies equality (NaN keys), then the numeric tower
  // compares across int/bool/float, then same-class `__eq__` through the
  // generated hook. Distinct classes outside the tower compare unequal.
  func.func private @__ly_box_equal(%lhs: !llvm.ptr, %rhs: !llvm.ptr) -> i1 {
    %zero = arith.constant 0 : i64
    %true = arith.constant true
    %false = arith.constant false
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %lhs_ptr = llvm.load %lhs : !llvm.ptr -> i64
    %rhs_ptr = llvm.load %rhs : !llvm.ptr -> i64
    %lhs_class = func.call @__ly_slot_class(%lhs_ptr) : (i64) -> i64
    %rhs_class = func.call @__ly_slot_class(%rhs_ptr) : (i64) -> i64
    %same_ptr = arith.cmpi eq, %lhs_ptr, %rhs_ptr : i64
    %ptr_nonzero = arith.cmpi ne, %lhs_ptr, %zero : i64
    %same_class = arith.cmpi eq, %lhs_class, %rhs_class : i64
    %identical0 = arith.andi %same_ptr, %ptr_nonzero : i1
    %identical = arith.andi %identical0, %same_class : i1
    %result = scf.if %identical -> (i1) {
      scf.yield %true : i1
    } else {
      %lhs_none = arith.cmpi eq, %lhs_class, %zero : i64
      %rhs_none = arith.cmpi eq, %rhs_class, %zero : i64
      %either_none = arith.ori %lhs_none, %rhs_none : i1
      %outer = scf.if %either_none -> (i1) {
        %both_none = arith.andi %lhs_none, %rhs_none : i1
        scf.yield %both_none : i1
      } else {
        %int_class = arith.constant 1 : i64
        %float_class = arith.constant 2 : i64
        %bool_class = arith.constant 22 : i64
        %lhs_int = arith.cmpi eq, %lhs_class, %int_class : i64
        %lhs_float = arith.cmpi eq, %lhs_class, %float_class : i64
        %lhs_bool = arith.cmpi eq, %lhs_class, %bool_class : i64
        %rhs_int = arith.cmpi eq, %rhs_class, %int_class : i64
        %rhs_float = arith.cmpi eq, %rhs_class, %float_class : i64
        %rhs_bool = arith.cmpi eq, %rhs_class, %bool_class : i64
        %lhs_num0 = arith.ori %lhs_int, %lhs_float : i1
        %lhs_num = arith.ori %lhs_num0, %lhs_bool : i1
        %rhs_num0 = arith.ori %rhs_int, %rhs_float : i1
        %rhs_num = arith.ori %rhs_num0, %rhs_bool : i1
        %both_num = arith.andi %lhs_num, %rhs_num : i1
        %mixed_class = arith.cmpi ne, %lhs_class, %rhs_class : i64
        %numeric_mixed = arith.andi %both_num, %mixed_class : i1
        %same = arith.cmpi eq, %lhs_class, %rhs_class : i64
        // ⭐ Two ints or two floats of which one is an immediate compare by
        // value, with no object made for the hook to read. Equal immediates
        // never get here (the identity test above answers them).
        %lhs_imm = func.call @__ly_slot_word_is_immediate(%lhs_ptr) : (i64) -> i1
        %rhs_imm = func.call @__ly_slot_word_is_immediate(%rhs_ptr) : (i64) -> i1
        %any_imm = arith.ori %lhs_imm, %rhs_imm : i1
        %both_int = arith.andi %lhs_int, %rhs_int : i1
        %both_float = arith.andi %lhs_float, %rhs_float : i1
        %int_by_value = arith.andi %both_int, %any_imm : i1
        %float_by_value = arith.andi %both_float, %any_imm : i1
        %by_value = arith.ori %int_by_value, %float_by_value : i1
        %num_result = scf.if %by_value -> (i1) {
          %r = scf.if %int_by_value -> (i1) {
            %lv, %lfits = func.call @LyLong_SlotWordAsI64(%lhs_ptr) : (i64) -> (i64, i1)
            %rv, %rfits = func.call @LyLong_SlotWordAsI64(%rhs_ptr) : (i64) -> (i64, i1)
            %fit = arith.andi %lfits, %rfits : i1
            %veq = arith.cmpi eq, %lv, %rv : i64
            %ieq = arith.andi %fit, %veq : i1
            scf.yield %ieq : i1
          } else {
            %lf = func.call @LyFloat_SlotWordAsF64(%lhs_ptr) : (i64) -> f64
            %rf = func.call @LyFloat_SlotWordAsF64(%rhs_ptr) : (i64) -> f64
            %feq = arith.cmpf oeq, %lf, %rf : f64
            scf.yield %feq : i1
          }
          scf.yield %r : i1
        } else {
        %num_result_inner = scf.if %numeric_mixed -> (i1) {
          %r = func.call @__ly_box_equal_numeric(%lhs, %lhs_class, %rhs, %rhs_class) : (!llvm.ptr, i64, !llvm.ptr, i64) -> i1
          scf.yield %r : i1
        } else {
          // ⭐ A SUBCLASS SHARES ITS BASE'S IMPLEMENTATION, so "same class id"
          // was the wrong gate: `P(1) in [Q(1)]` compared a P against a Q,
          // found the ids unequal, and answered False without asking anything.
          // The hook decides now -- it accepts a right-hand class that resolves
          // the same implementation -- and `%handled` is the whole answer.
          %eq, %handled = func.call @__ly_eq_boxed_by_contract(%lhs, %rhs, %lhs_class, %rhs_class) : (!llvm.ptr, !llvm.ptr, i64, i64) -> (i1, i1)
          %same_result = arith.andi %eq, %handled : i1
          scf.yield %same_result : i1
        }
        scf.yield %num_result_inner : i1
        }
        scf.yield %num_result : i1
      }
      scf.yield %outer : i1
    }
    func.return %result : i1
  }

  // Boxed bool as an i64 value (0/1) via the singleton's value word.
  func.func private @__ly_boxed_bool_value(%box: !llvm.ptr) -> i64 {
    %c2_i64 = arith.constant 2 : i64
    %entity_word = llvm.load %box : !llvm.ptr -> i64
    %entity = llvm.inttoptr %entity_word : i64 to !llvm.ptr
    %value_gep = llvm.getelementptr %entity[%c2_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %value = llvm.load %value_gep : !llvm.ptr -> i64
    func.return %value : i64
  }

  // Boxed float payload: box word 2 is the entity pointer, and the double's bit
  // pattern is entity word 2 (LyFloat_Shape). Reached through the ENTITY word
  // rather than a lane pointer word so the accessor does not encode how many
  // lanes the contract expands to -- it used to read box pointer word 5, i.e.
  // lane 1, which one-laning float left uninitialised.
  func.func private @__ly_boxed_float_value(%box: !llvm.ptr) -> f64 {
    %c2_i64 = arith.constant 2 : i64
    %entity_word = llvm.load %box : !llvm.ptr -> i64
    %value = func.call @LyFloat_SlotWordAsF64(%entity_word) : (i64) -> f64
    func.return %value : f64
  }

  // Mixed-class numeric equality across int/bool/float boxes.
  func.func private @__ly_box_equal_numeric(%lhs: !llvm.ptr, %lhs_class: i64, %rhs: !llvm.ptr, %rhs_class: i64) -> i1 {
    %long_scratch = memref.alloca() : memref<3xi32>
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %false = arith.constant false
    %int_class = arith.constant 1 : i64
    %float_class = arith.constant 2 : i64
    %bool_class = arith.constant 22 : i64
    %lhs_is_float = arith.cmpi eq, %lhs_class, %float_class : i64
    %rhs_is_float = arith.cmpi eq, %rhs_class, %float_class : i64
    %either_float = arith.ori %lhs_is_float, %rhs_is_float : i1
    %lhs_is_bool0 = arith.cmpi eq, %lhs_class, %bool_class : i64
    %rhs_is_bool0 = arith.cmpi eq, %rhs_class, %bool_class : i64
    %both_bool = arith.andi %lhs_is_bool0, %rhs_is_bool0 : i1
    %result = scf.if %both_bool -> (i1) {
      %lv = func.call @__ly_boxed_bool_value(%lhs) : (!llvm.ptr) -> i64
      %rv = func.call @__ly_boxed_bool_value(%rhs) : (!llvm.ptr) -> i64
      %beq = arith.cmpi eq, %lv, %rv : i64
      scf.yield %beq : i1
    } else {
    %inner_result = scf.if %either_float -> (i1) {
      // Exactly one side is a float here (float/float is same-class).
      %fv = scf.if %lhs_is_float -> (f64) {
        %v = func.call @__ly_boxed_float_value(%lhs) : (!llvm.ptr) -> f64
        scf.yield %v : f64
      } else {
        %v = func.call @__ly_boxed_float_value(%rhs) : (!llvm.ptr) -> f64
        scf.yield %v : f64
      }
      %other = arith.select %lhs_is_float, %rhs, %lhs : i1, !llvm.ptr
      %other_class = arith.select %lhs_is_float, %rhs_class, %lhs_class : i1, i64
      %other_is_bool = arith.cmpi eq, %other_class, %bool_class : i64
      %cmp = scf.if %other_is_bool -> (i1) {
        %bv = func.call @__ly_boxed_bool_value(%other) : (!llvm.ptr) -> i64
        %bf = arith.sitofp %bv : i64 to f64
        %eq = arith.cmpf oeq, %bf, %fv : f64
        scf.yield %eq : i1
      } else {
        %sign, %count, %digits = func.call @__ly_boxed_long_view(%other, %long_scratch) : (!llvm.ptr, memref<3xi32>) -> (i64, i64, i64)
        %eq = func.call @__ly_boxed_long_eq_f64(%sign, %count, %digits, %fv) : (i64, i64, i64, f64) -> i1
        scf.yield %eq : i1
      }
      scf.yield %cmp : i1
    } else {
      // int/bool (mixed): bool value 0 -> int zero; 1 -> int one.
      %lhs_is_bool = arith.cmpi eq, %lhs_class, %bool_class : i64
      %bool_box = arith.select %lhs_is_bool, %lhs, %rhs : i1, !llvm.ptr
      %int_box = arith.select %lhs_is_bool, %rhs, %lhs : i1, !llvm.ptr
      %bv = func.call @__ly_boxed_bool_value(%bool_box) : (!llvm.ptr) -> i64
      %sign, %count, %digits = func.call @__ly_boxed_long_view(%int_box, %long_scratch) : (!llvm.ptr, memref<3xi32>) -> (i64, i64, i64)
      %bool_false = arith.cmpi eq, %bv, %zero : i64
      %cmp = scf.if %bool_false -> (i1) {
        %int_zero = arith.cmpi eq, %sign, %zero : i64
        scf.yield %int_zero : i1
      } else {
        %sign_one = arith.cmpi eq, %sign, %one : i64
        %count_one = arith.cmpi eq, %count, %one : i64
        %digits_ptr = llvm.inttoptr %digits : i64 to !llvm.ptr
        %d0_i32 = llvm.load %digits_ptr : !llvm.ptr -> i32
        %d0 = arith.extui %d0_i32 : i32 to i64
        %d0_one = arith.cmpi eq, %d0, %one : i64
        %m0 = arith.andi %sign_one, %count_one : i1
        %m1 = arith.andi %m0, %d0_one : i1
        scf.yield %m1 : i1
      }
      scf.yield %cmp : i1
    }
    scf.yield %inner_result : i1
    }
    func.return %result : i1
  }

  func.func private @__ly_lt_boxed_by_contract(%lhs: !llvm.ptr, %rhs: !llvm.ptr, %class_id: i64, %rhs_class_id: i64) -> (i1, i1)

  // "'<' not supported between operand types"
  memref.global "private" constant @__ly_cmp_msg_unorderable : memref<39xi8> = dense<[39, 60, 39, 32, 110, 111, 116, 32, 115, 117, 112, 112, 111, 114, 116, 101, 100, 32, 98, 101, 116, 119, 101, 101, 110, 32, 111, 112, 101, 114, 97, 110, 100, 32, 116, 121, 112, 101, 115]>

  func.func private @__ly_cmp_raise_unorderable() {
    %type_error = arith.constant 52 : i64
    %msg_static = memref.get_global @__ly_cmp_msg_unorderable : memref<39xi8>
    %msg = memref.cast %msg_static : memref<39xi8> to memref<?xi8>
    %len = arith.constant 39 : i64
    func.call @__ly_raise_static_message(%type_error, %msg, %len) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  // Boxed int as f64 (30-bit limb accumulation; values beyond 2^53 round,
  // matching a float(int) conversion for comparison purposes).
  func.func private @__ly_boxed_num_as_f64(%box: !llvm.ptr, %class_id: i64) -> f64 {
    %long_scratch = memref.alloca() : memref<3xi32>
    %float_class = arith.constant 2 : i64
    %bool_class = arith.constant 22 : i64
    %is_float = arith.cmpi eq, %class_id, %float_class : i64
    %result = scf.if %is_float -> (f64) {
      %v = func.call @__ly_boxed_float_value(%box) : (!llvm.ptr) -> f64
      scf.yield %v : f64
    } else {
      %is_bool = arith.cmpi eq, %class_id, %bool_class : i64
      %num = scf.if %is_bool -> (f64) {
        %bv = func.call @__ly_boxed_bool_value(%box) : (!llvm.ptr) -> i64
        %bf = arith.sitofp %bv : i64 to f64
        scf.yield %bf : f64
      } else {
        %sign, %count, %digits_word = func.call @__ly_boxed_long_view(%box, %long_scratch) : (!llvm.ptr, memref<3xi32>) -> (i64, i64, i64)
        %zero = arith.constant 0 : i64
        %one = arith.constant 1 : i64
        %limb_scale = arith.constant 1073741824.0 : f64
        %digits_ptr = llvm.inttoptr %digits_word : i64 to !llvm.ptr
        %count_index = arith.index_cast %count : i64 to index
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %zero_f = arith.constant 0.0 : f64
        %mag = scf.for %i = %c0 to %count_index step %c1 iter_args(%acc = %zero_f) -> (f64) {
          %ii = arith.index_cast %i : index to i64
          %rev = arith.subi %count, %ii : i64
          %rev_index = arith.subi %rev, %one : i64
          %limb_gep = llvm.getelementptr %digits_ptr[%rev_index] : (!llvm.ptr, i64) -> !llvm.ptr, i32
          %limb_i32 = llvm.load %limb_gep : !llvm.ptr -> i32
          %limb = arith.extui %limb_i32 : i32 to i64
          %limb_f = arith.uitofp %limb : i64 to f64
          %scaled = arith.mulf %acc, %limb_scale : f64
          %next = arith.addf %scaled, %limb_f : f64
          scf.yield %next : f64
        }
        %neg = arith.cmpi slt, %sign, %zero : i64
        %negated = arith.negf %mag : f64
        %signed = arith.select %neg, %negated, %mag : i1, f64
        scf.yield %signed : f64
      }
      scf.yield %num : f64
    }
    func.return %result : f64
  }

  // Strict less-than over two boxed values: numeric tower across classes,
  // same-class `__lt__` through the generated hook, TypeError otherwise
  // (CPython rejects cross-type ordering).
  func.func private @__ly_box_less(%lhs: !llvm.ptr, %rhs: !llvm.ptr) -> i1 {
    %false = arith.constant false
    %c1_i64 = arith.constant 1 : i64
    %lhs_word = llvm.load %lhs : !llvm.ptr -> i64
    %rhs_word = llvm.load %rhs : !llvm.ptr -> i64
    %lhs_class = func.call @__ly_slot_class(%lhs_word) : (i64) -> i64
    %rhs_class = func.call @__ly_slot_class(%rhs_word) : (i64) -> i64
    %int_class = arith.constant 1 : i64
    %float_class = arith.constant 2 : i64
    %bool_class = arith.constant 22 : i64
    %lhs_int = arith.cmpi eq, %lhs_class, %int_class : i64
    %lhs_float = arith.cmpi eq, %lhs_class, %float_class : i64
    %lhs_bool = arith.cmpi eq, %lhs_class, %bool_class : i64
    %rhs_int = arith.cmpi eq, %rhs_class, %int_class : i64
    %rhs_float = arith.cmpi eq, %rhs_class, %float_class : i64
    %rhs_bool = arith.cmpi eq, %rhs_class, %bool_class : i64
    %lhs_num0 = arith.ori %lhs_int, %lhs_float : i1
    %lhs_num = arith.ori %lhs_num0, %lhs_bool : i1
    %rhs_num0 = arith.ori %rhs_int, %rhs_float : i1
    %rhs_num = arith.ori %rhs_num0, %rhs_bool : i1
    %both_num = arith.andi %lhs_num, %rhs_num : i1
    %same = arith.cmpi eq, %lhs_class, %rhs_class : i64
    %mixed_num = arith.cmpi ne, %lhs_class, %rhs_class : i64
    // bool/bool pairs also take the numeric path: bool has no __lt__ hook
    // entry of its own (CPython orders bools as ints).
    %both_bool = arith.andi %lhs_bool, %rhs_bool : i1
    %mixed_or_bool = arith.ori %mixed_num, %both_bool : i1
    %numeric_mixed = arith.andi %both_num, %mixed_or_bool : i1
    %result = scf.if %numeric_mixed -> (i1) {
      // Equal values first (exact), then the f64 ordering: only values that
      // differ by less than one f64 ulp beyond 2^53 can misorder here.
      %eq = func.call @__ly_box_equal_numeric(%lhs, %lhs_class, %rhs, %rhs_class) : (!llvm.ptr, i64, !llvm.ptr, i64) -> i1
      %ordered = scf.if %eq -> (i1) {
        scf.yield %false : i1
      } else {
        %lf = func.call @__ly_boxed_num_as_f64(%lhs, %lhs_class) : (!llvm.ptr, i64) -> f64
        %rf = func.call @__ly_boxed_num_as_f64(%rhs, %rhs_class) : (!llvm.ptr, i64) -> f64
        %lt = arith.cmpf olt, %lf, %rf : f64
        scf.yield %lt : i1
      }
      scf.yield %ordered : i1
    } else {
      // The same rule as the equality hook: a subclass resolves its base's
      // `__lt__`, so the two ids need not be equal for the callee's lanes to be
      // there. `sorted([Q(2), P(1)])` raised TypeError where CPython sorts.
      %lt, %handled = func.call @__ly_lt_boxed_by_contract(%lhs, %rhs, %lhs_class, %rhs_class) : (!llvm.ptr, !llvm.ptr, i64, i64) -> (i1, i1)
      scf.if %handled {
      } else {
        func.call @__ly_cmp_raise_unorderable() : () -> ()
      }
      %inner = arith.select %handled, %lt, %false : i1
      scf.yield %inner : i1
    }
    func.return %result : i1
  }

  // One 16-word element from %src[%s] to %dst[%d]. Raw words: a slot move is
  // not a reference change, so nothing here retains or releases.
  func.func private @__ly_box_move_slot(%dst: memref<?xi64>, %d: i64, %src: memref<?xi64>, %s: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %words = func.call @__ly_box_word_count() : () -> i64
    %words_index = arith.index_cast %words : i64 to index
    %db = func.call @__ly_box_slot_base(%d) : (i64) -> i64
    %sb = func.call @__ly_box_slot_base(%s) : (i64) -> i64
    %dbi = arith.index_cast %db : i64 to index
    %sbi = arith.index_cast %sb : i64 to index
    scf.for %w = %c0 to %words_index step %c1 {
      %di = arith.addi %dbi, %w : index
      %si = arith.addi %sbi, %w : index
      %v = memref.load %src[%si] : memref<?xi64>
      memref.store %v, %dst[%di] : memref<?xi64>
    }
    func.return
  }

  func.func private @__ly_slot_less(%items_ptr: !llvm.ptr, %a: i64, %b: i64) -> i1 {
    %c16 = func.call @__ly_box_word_count() : () -> i64
    %ao = arith.muli %a, %c16 : i64
    %bo = arith.muli %b, %c16 : i64
    %ap = llvm.getelementptr %items_ptr[%ao] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %bp = llvm.getelementptr %items_ptr[%bo] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %less = func.call @__ly_box_less(%ap, %bp) : (!llvm.ptr, !llvm.ptr) -> i1
    func.return %less : i1
  }

  func.func private @LyObject_ReleaseBoxedPayloadArraySlotRaw(%payload: memref<?xi64>, %logical_index: i64)
  func.func private @LyObject_RetainBoxedPayloadArraySlotRaw(%payload: memref<?xi64>, %logical_index: i64)

  // Retain an arbitrary payload entity through its raw address (slot word 2).
  // Mirrors Ly_IncRef's null/tagged/immortal rules; erased slots have no
  // memref descriptor to hand the ordinary retain.
  func.func private @__ly_handle_retain_raw(%entity: i64) {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %immortal = arith.constant 9223372036854775807 : i64
    %is_null = arith.cmpi eq, %entity, %zero : i64
    %tag_mask = arith.constant 3 : i64
    %tag = arith.andi %entity, %tag_mask : i64
    %is_tagged = arith.cmpi ne, %tag, %zero : i64
    %skip = arith.ori %is_null, %is_tagged : i1
    scf.if %skip {
    } else {
      %ptr = llvm.inttoptr %entity : i64 to !llvm.ptr
      %observed = llvm.load %ptr atomic acquire {alignment = 8 : i64} : !llvm.ptr -> i64
      %is_immortal = arith.cmpi eq, %observed, %immortal : i64
      scf.if %is_immortal {
      } else {
        %prev = llvm.atomicrmw add %ptr, %one acq_rel : !llvm.ptr, i64
        %positive = arith.cmpi sgt, %prev, %zero : i64
        cf.assert %positive, "__ly_handle_retain_raw observed non-positive refcount"
      }
    }
    func.return
  }

  // Box the canonical payload handle stored at a collection slot into a
  // fresh owned `builtins.object` box (the erased read lane): the box adopts
  // one new reference to the payload entity. An invalid slot (exhausted
  // iteration) yields the all-zero None handle without touching the array.
  func.func @LyObject_FromSlot(%items: memref<?xi64>, %slot: i64, %valid: i1) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.object", ly.runtime.primitive = "from_slot", ly.runtime.result_contract = "builtins.object"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %box = memref.alloc() {alignment = 16 : i64, ly.ownership.object_header, ly.ownership.owned_local_object} : memref<5xi64>
    %all_words_i64 = func.call @__ly_box_standalone_word_count() : () -> i64
    %all_words = arith.index_cast %all_words_i64 : i64 to index
    scf.for %w = %c0 to %all_words step %c1 {
      memref.store %zero, %box[%w] : memref<5xi64>
    }
    %refcount_slot = arith.constant 0 : index
    memref.store %one, %box[%refcount_slot] : memref<5xi64>
    scf.if %valid {
      %base_i64 = func.call @__ly_box_slot_base(%slot) : (i64) -> i64
      %entity_word = func.call @__ly_box_entity_word(%base_i64) : (i64) -> i64
      %entity_index = arith.index_cast %entity_word : i64 to index
      %entity = memref.load %items[%entity_index] : memref<?xi64>
      %class_id = func.call @__ly_slot_class(%entity) : (i64) -> i64
      %class_slot = arith.constant 1 : index
      %box_entity_slot = arith.constant 2 : index
      memref.store %class_id, %box[%class_slot] : memref<5xi64>
      memref.store %entity, %box[%box_entity_slot] : memref<5xi64>
      // Skips a null or immediate entity, which is exactly what the box then
      // does not own (`release_payload_slot_ptr` asks the same question).
      func.call @__ly_handle_retain_raw(%entity) : (i64) -> ()
    }
    func.return %box : memref<5xi64>
  }

  // A standalone box of the value in the slot at %slot_word (an address),
  // with a reference of its own: for a slot that is not in an items array,
  // like an exception's field block.
  func.func @LyObject_FromSlotPtr(%slot_word: i64) -> memref<5xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.object", ly.runtime.primitive = "from_slot_ptr", ly.runtime.result_contract = "builtins.object"} {
    %one = arith.constant 1 : i64
    %view = func.call @__ly_global_view_i64(%slot_word, %one) : (i64, i64) -> memref<?xi64>
    %zero = arith.constant 0 : i64
    %true = arith.constant true
    %box = func.call @LyObject_FromSlot(%view, %zero, %true) : (memref<?xi64>, i64, i1) -> memref<5xi64>
    func.return %box : memref<5xi64>
  }

  // Uniform per-element repr dispatch, generated per program by the lowering
  // (class id -> the manifest __repr__); resolved at link. Returns an owned str
  // (header, bytes) plus a handled flag.
  func.func private @__ly_repr_boxed_by_contract(%box: !llvm.ptr, %class_id: i64) -> (memref<2xi64>, memref<?xi8>, i1) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]}

  memref.global "private" constant @__ly_repr_lbracket : memref<1xi8> = dense<91>
  memref.global "private" constant @__ly_repr_rbracket : memref<1xi8> = dense<93>
  memref.global "private" constant @__ly_repr_comma : memref<2xi8> = dense<[44, 32]>
  memref.global "private" constant @__ly_repr_lparen : memref<1xi8> = dense<40>
  memref.global "private" constant @__ly_repr_rparen : memref<1xi8> = dense<41>
  memref.global "private" constant @__ly_repr_lbrace : memref<1xi8> = dense<123>
  memref.global "private" constant @__ly_repr_rbrace : memref<1xi8> = dense<125>
  memref.global "private" constant @__ly_repr_colon : memref<2xi8> = dense<[58, 32]>
  memref.global "private" constant @__ly_repr_set_empty : memref<5xi8> = dense<[115, 101, 116, 40, 41]>
  memref.global "private" constant @__ly_repr_frozenset_open : memref<11xi8> = dense<[102, 114, 111, 122, 101, 110, 115, 101, 116, 40, 123]>
  memref.global "private" constant @__ly_repr_frozenset_empty : memref<11xi8> = dense<[102, 114, 111, 122, 101, 110, 115, 101, 116, 40, 41]>
  memref.global "private" constant @__ly_repr_range_open : memref<6xi8> = dense<[114, 97, 110, 103, 101, 40]>
}
