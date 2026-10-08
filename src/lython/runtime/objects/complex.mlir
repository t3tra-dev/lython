// `complex` -- CPython's Objects/complexobject.c (_Py_c_quot, _Py_c_pow and
// the rest of the arithmetic line for line).
//
// Signature source (1:1 correspondence target):
//   https://github.com/python/typeshed/blob/main/stdlib/builtins.pyi

module attributes {
  ly.typing.manifest,
  ly.runtime.contracts = ["builtins.complex"]
} {
  // ===== declared here, defined in another runtime file or built by the lowering =====
  func.func private @__ly_unicode_from_valid_utf8(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]}
  func.func private @LyFloat_AsF64(%header: memref<3xi64> {ly.ownership.object_header}) -> f64 attributes {ly.runtime.contract = "builtins.float", ly.runtime.method = "__float__", ly.runtime.primitive = "unbox.f64"}
  func.func private @LyFloat_DecRef(%header: memref<3xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.float", ly.runtime.deallocator}
  func.func private @LyFloat_FromF64(%value: f64 {ly.runtime.default_f64 = 0.0 : f64}) -> memref<3xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id, ly.runtime.contract = "builtins.float", ly.runtime.initializer = "__new__"}
  func.func private @LyFloat_Repr(%header: memref<3xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.float", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"}
  func.func private @LyObject_ReleaseStorageToZero(%storage: memref<?xi64>) -> i1 attributes {ly.runtime.contract = "builtins.object", ly.runtime.primitive = "release_to_zero"}
  func.func private @LyUnicode_Concat(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_bytes: memref<?xi8>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__add__"}
  func.func private @LyUnicode_DecRef(%header: memref<2xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.str", ly.runtime.deallocator}
  func.func private @LyUnicode_FromBytes(%bytes: memref<?xi8>, %start: index, %len: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id, ly.runtime.contract = "builtins.str", ly.runtime.initializer = "__new__"}
  func.func private @__ly_float_hash_bits(%bits: i64, %ident: i64) -> i64
  func.func private @__ly_hash_fixup(%h: i64) -> i64
  func.func private @__ly_long_cmp_f64(%meta_raw: memref<2xi64>, %digits_raw: memref<?xi32>, %d: f64) -> i64
  func.func private @__ly_long_parts(%header: memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
  func.func private @__ly_raise_static_message(%class_id: i64, %message: memref<?xi8>, %length: i64)

  py.class @complex attributes {
    base_names = ["object"], ly.typing.final,
    // ⭐ The three __new__ arities are what makes the NAME callable: the
    // runtime has had LyComplex_FromParts (two f64 with defaults) all along, so
    // `1 + 2j` ran while `complex(1, 2)` was "unresolved name 'complex'" and,
    // once the name bound, "builtins.complex does not provide manifest method
    // '__init__'". Declared the way range declares its three, because the
    // arities are what the call site resolves against.
    method_names = ["__new__", "__new__", "__new__", "__new__",
                    "__new__", "__new__", "__new__", "__init__",
                    "__init__", "__init__", "__init__", "__init__",
                    "__init__", "__init__", "__add__", "__sub__",
                    "__mul__", "__truediv__", "__pow__", "__add__",
                    "__sub__", "__mul__", "__truediv__", "__pow__",
                    "__radd__", "__rsub__", "__rmul__", "__rtruediv__",
                    "__rpow__", "__neg__", "__pos__", "__eq__",
                    "__ne__", "__hash__", "__repr__", "__str__",
                    "__abs__", "conjugate", "__bool__"],
    method_contracts = [
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.complex">>] -> [!py.self]>,
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.complex">>, !py.contract<"builtins.float">] -> [!py.self]>,
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.complex">>, !py.contract<"builtins.int">] -> [!py.self]>,
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.complex">>, !py.contract<"builtins.float">, !py.contract<"builtins.float">] -> [!py.self]>,
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.complex">>, !py.contract<"builtins.float">, !py.contract<"builtins.int">] -> [!py.self]>,
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.complex">>, !py.contract<"builtins.int">, !py.contract<"builtins.float">] -> [!py.self]>,
      !py.protocol<"Callable", [!py.type<!py.contract<"builtins.complex">>, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.self]>,
      !py.protocol<"Callable", [!py.contract<"builtins.complex">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.complex">, !py.contract<"builtins.float">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.complex">, !py.contract<"builtins.int">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.complex">, !py.contract<"builtins.float">, !py.contract<"builtins.float">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.complex">, !py.contract<"builtins.float">, !py.contract<"builtins.int">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.complex">, !py.contract<"builtins.int">, !py.contract<"builtins.float">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.complex">, !py.contract<"builtins.int">, !py.contract<"builtins.int">] -> [!py.literal<None>]>,
      !py.protocol<"Callable", [!py.contract<"builtins.complex">, !py.contract<"builtins.complex">] -> [!py.contract<"builtins.complex">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.complex">, !py.contract<"builtins.complex">] -> [!py.contract<"builtins.complex">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.complex">, !py.contract<"builtins.complex">] -> [!py.contract<"builtins.complex">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.complex">, !py.contract<"builtins.complex">] -> [!py.contract<"builtins.complex">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.complex">, !py.contract<"builtins.complex">] -> [!py.contract<"builtins.complex">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.complex">, !py.contract<"builtins.float">] -> [!py.contract<"builtins.complex">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.complex">, !py.contract<"builtins.float">] -> [!py.contract<"builtins.complex">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.complex">, !py.contract<"builtins.float">] -> [!py.contract<"builtins.complex">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.complex">, !py.contract<"builtins.float">] -> [!py.contract<"builtins.complex">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.complex">, !py.contract<"builtins.float">] -> [!py.contract<"builtins.complex">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.complex">, !py.contract<"builtins.float">] -> [!py.contract<"builtins.complex">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.complex">, !py.contract<"builtins.float">] -> [!py.contract<"builtins.complex">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.complex">, !py.contract<"builtins.float">] -> [!py.contract<"builtins.complex">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.complex">, !py.contract<"builtins.float">] -> [!py.contract<"builtins.complex">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.complex">, !py.contract<"builtins.float">] -> [!py.contract<"builtins.complex">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.complex">] -> [!py.contract<"builtins.complex">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.complex">] -> [!py.contract<"builtins.complex">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.complex">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.complex">, !py.contract<"builtins.object">] -> [!py.contract<"builtins.bool">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.complex">] -> [!py.contract<"builtins.int">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.complex">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.complex">] -> [!py.contract<"builtins.str">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.complex">] -> [!py.contract<"builtins.float">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.complex">] -> [!py.contract<"builtins.complex">]>,
      !py.protocol<"Callable", [!py.contract<"builtins.complex">] -> [!py.contract<"builtins.bool">]>
    ],
    method_kinds = ["classmethod", "classmethod", "classmethod", "classmethod",
                    "classmethod", "classmethod", "classmethod", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance", "instance",
                    "instance", "instance", "instance"]
  } {}

  // ===== impls: complex (R6 value type, one 7-word handle) =====

  // Handle words: 0 refcount, 1 layout/destructor family id, 2 real bits,
  // 3 imag bits, 4-6 unused. See LyFloat_Shape for why the width differs from
  // every lane width the two-lane form used.
  //
  func.func private @LyComplex_Shape() -> memref<4xi64> attributes {ly.runtime.contract = "builtins.complex", ly.runtime.shape}

  func.func @LyComplex_FromParts(%real: f64 {ly.runtime.default_f64 = 0.0 : f64}, %imag: f64 {ly.runtime.default_f64 = 0.0 : f64}) -> memref<4xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id, ly.runtime.contract = "builtins.complex", ly.runtime.initializer = "__new__"} {
    // One entity, one handle: words 2 and 3 carry the two doubles as their bit
    // patterns. See LyFloat_FromF64 for why the bits and not an f64-typed lane.
    %block_bytes = arith.constant 32 : index
    %block = memref.alloc(%block_bytes) {alignment = 16 : i64} : memref<?xi8>
    %self_offset = arith.constant 0 : index
    %header = memref.view %block[%self_offset][] {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<?xi8> to memref<4xi64>
    %one = arith.constant 1 : i64
    %layout_complex = arith.constant {ly.class_id_of = "builtins.complex"} 13 : i64
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %real_slot = arith.constant 2 : index
    %imag_slot = arith.constant 3 : index
    %real_bits = arith.bitcast %real : f64 to i64
    %imag_bits = arith.bitcast %imag : f64 to i64
    memref.store %one, %header[%refcount_slot] : memref<4xi64>
    memref.store %layout_complex, %header[%layout_slot] : memref<4xi64>
    memref.store %real_bits, %header[%real_slot] : memref<4xi64>
    memref.store %imag_bits, %header[%imag_slot] : memref<4xi64>
    func.return %header : memref<4xi64>
  }

  // ⛔ An empty __init__ is not a placeholder: LyComplex_FromParts is the
  // __new__ and it builds the whole value, but the constructor path calls both
  // and the MRO's next __init__ provider is builtins.object's, whose input is a
  // boxed object -- "cannot pass concrete object builtins.complex as
  // builtins.object". Declaring complex's own is what stops the lookup here.
  func.func @LyComplex_Init(%self: memref<4xi64> {ly.ownership.object_header}, %real: f64 {ly.runtime.default_f64 = 0.0 : f64}, %imag: f64 {ly.runtime.default_f64 = 0.0 : f64}) attributes {ly.runtime.contract = "builtins.complex", ly.runtime.method = "__init__"} {
    func.return
  }

  func.func @LyComplex_DecRef(%header: memref<4xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.complex", ly.runtime.deallocator} {
    %storage = memref.cast %header : memref<4xi64> to memref<?xi64>
    %became_zero = func.call @LyObject_ReleaseStorageToZero(%storage) : (memref<?xi64>) -> i1
    cf.cond_br %became_zero, ^dealloc, ^done

  ^dealloc:
    memref.dealloc %header : memref<4xi64>
    cf.br ^done

  ^done:
    func.return
  }

  func.func @LyComplex_Add(%lhs_header: memref<4xi64> {ly.ownership.object_header}, %rhs_header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.complex"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.complex", ly.runtime.method = "__add__"} {
    %re_slot = arith.constant 2 : index
    %im_slot = arith.constant 3 : index
    %a_bits = memref.load %lhs_header[%re_slot] : memref<4xi64>
    %a = arith.bitcast %a_bits : i64 to f64
    %b_bits = memref.load %lhs_header[%im_slot] : memref<4xi64>
    %b = arith.bitcast %b_bits : i64 to f64
    %c_bits = memref.load %rhs_header[%re_slot] : memref<4xi64>
    %c = arith.bitcast %c_bits : i64 to f64
    %d_bits = memref.load %rhs_header[%im_slot] : memref<4xi64>
    %d = arith.bitcast %d_bits : i64 to f64
    %re = arith.addf %a, %c : f64
    %im = arith.addf %b, %d : f64
    %header = func.call @LyComplex_FromParts(%re, %im) : (f64, f64) -> memref<4xi64>
    func.return %header : memref<4xi64>
  }

  func.func @LyComplex_Sub(%lhs_header: memref<4xi64> {ly.ownership.object_header}, %rhs_header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.complex"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.complex", ly.runtime.method = "__sub__"} {
    %re_slot = arith.constant 2 : index
    %im_slot = arith.constant 3 : index
    %a_bits = memref.load %lhs_header[%re_slot] : memref<4xi64>
    %a = arith.bitcast %a_bits : i64 to f64
    %b_bits = memref.load %lhs_header[%im_slot] : memref<4xi64>
    %b = arith.bitcast %b_bits : i64 to f64
    %c_bits = memref.load %rhs_header[%re_slot] : memref<4xi64>
    %c = arith.bitcast %c_bits : i64 to f64
    %d_bits = memref.load %rhs_header[%im_slot] : memref<4xi64>
    %d = arith.bitcast %d_bits : i64 to f64
    %re = arith.subf %a, %c : f64
    %im = arith.subf %b, %d : f64
    %header = func.call @LyComplex_FromParts(%re, %im) : (f64, f64) -> memref<4xi64>
    func.return %header : memref<4xi64>
  }
  // ===== complex arithmetic: CPython's Objects/complexobject.c, line for line
  //
  // Every rounding is CPython's on the same target. Where CPython's C source
  // writes `x*y + z` in ONE expression, clang (-ffp-contract=on, its default)
  // emits llvm.fmuladd, which the backend fuses where the target has FMA
  // (aarch64) and leaves as a multiply and an add where it has not (baseline
  // x86-64, wasm32). The same intrinsic is written here at exactly those
  // places, so the last ulp agrees with the CPython built for the target.
  //
  // ⛔ NOT the complex dialect's lowering (convert-complex-to-standard): its
  // division and multiplication are different formulas, and against CPython
  // 3.14 on 4000 random operands they disagreed in 58 quotients and 67
  // products. ⛔ NOT fmuladd at the products of `_Py_c_prod`: CPython 3.14
  // declares `ac`, `bd`, `ad`, `bc` separately, which clang does not fuse, and
  // the 3.14 binary's products are the unfused ones.

  func.func private @hypot(%x: f64, %y: f64) -> f64

  func.func private @__ly_complex_parts(%header: memref<4xi64> {ly.ownership.object_header}) -> (f64, f64) {
    %re_slot = arith.constant 2 : index
    %im_slot = arith.constant 3 : index
    %re_bits = memref.load %header[%re_slot] : memref<4xi64>
    %re = arith.bitcast %re_bits : i64 to f64
    %im_bits = memref.load %header[%im_slot] : memref<4xi64>
    %im = arith.bitcast %im_bits : i64 to f64
    func.return %re, %im : f64, f64
  }

  func.func private @__ly_f64_isinf(%x: f64) -> i1 {
    %inf = arith.constant 0x7FF0000000000000 : f64
    %abs = math.absf %x : f64
    %is = arith.cmpf oeq, %abs, %inf : f64
    func.return %is : i1
  }

  func.func private @__ly_f64_isfinite(%x: f64) -> i1 {
    %inf = arith.constant 0x7FF0000000000000 : f64
    %abs = math.absf %x : f64
    %is = arith.cmpf olt, %abs, %inf : f64
    func.return %is : i1
  }

  // copysign(isinf(x) ? 1.0 : 0.0, x): an infinity "boxed" to a unit.
  func.func private @__ly_f64_box_inf(%x: f64) -> f64 {
    %one = arith.constant 1.0 : f64
    %zero = arith.constant 0.0 : f64
    %inf = func.call @__ly_f64_isinf(%x) : (f64) -> i1
    %unit = arith.select %inf, %one, %zero : f64
    %boxed = math.copysign %unit, %x : f64
    func.return %boxed : f64
  }

  // isnan(x) ? copysign(0.0, x) : x, when `when` holds.
  func.func private @__ly_f64_nan_to_zero(%x: f64, %when: i1) -> f64 {
    %zero = arith.constant 0.0 : f64
    %nan = arith.cmpf uno, %x, %x : f64
    %replace = arith.andi %nan, %when : i1
    %signed_zero = math.copysign %zero, %x : f64
    %out = arith.select %replace, %signed_zero, %x : f64
    func.return %out : f64
  }

  // _Py_c_prod.
  func.func private @__ly_c_prod(%a: f64, %b: f64, %c: f64, %d: f64) -> (f64, f64) {
    %ac = arith.mulf %a, %c : f64
    %bd = arith.mulf %b, %d : f64
    %ad = arith.mulf %a, %d : f64
    %bc = arith.mulf %b, %c : f64
    %re = arith.subf %ac, %bd : f64
    %im = arith.addf %ad, %bc : f64
    %re_nan = arith.cmpf uno, %re, %re : f64
    %im_nan = arith.cmpf uno, %im, %im : f64
    %both_nan = arith.andi %re_nan, %im_nan : i1
    %r:2 = scf.if %both_nan -> (f64, f64) {
      %rr:2 = func.call @__ly_c_prod_recover(%a, %b, %c, %d, %ac, %bd, %ad, %bc, %re, %im) : (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64) -> (f64, f64)
      scf.yield %rr#0, %rr#1 : f64, f64
    } else {
      scf.yield %re, %im : f64, f64
    }
    func.return %r#0, %r#1 : f64, f64
  }

  // The C11 Annex G.5.1 recovery `_Py_c_prod` runs when both parts are NaN.
  func.func private @__ly_c_prod_recover(%a0: f64, %b0: f64, %c0: f64, %d0: f64, %ac: f64, %bd: f64, %ad: f64, %bc: f64, %re: f64, %im: f64) -> (f64, f64) {
    %true = arith.constant true
    %false = arith.constant false
    %inf = arith.constant 0x7FF0000000000000 : f64
    // z is infinite.
    %a_inf = func.call @__ly_f64_isinf(%a0) : (f64) -> i1
    %b_inf = func.call @__ly_f64_isinf(%b0) : (f64) -> i1
    %z_inf = arith.ori %a_inf, %b_inf : i1
    %a_box = func.call @__ly_f64_box_inf(%a0) : (f64) -> f64
    %b_box = func.call @__ly_f64_box_inf(%b0) : (f64) -> f64
    %a1 = arith.select %z_inf, %a_box, %a0 : f64
    %b1 = arith.select %z_inf, %b_box, %b0 : f64
    %c1 = func.call @__ly_f64_nan_to_zero(%c0, %z_inf) : (f64, i1) -> f64
    %d1 = func.call @__ly_f64_nan_to_zero(%d0, %z_inf) : (f64, i1) -> f64
    // w is infinite.
    %c_inf = func.call @__ly_f64_isinf(%c1) : (f64) -> i1
    %d_inf = func.call @__ly_f64_isinf(%d1) : (f64) -> i1
    %w_inf = arith.ori %c_inf, %d_inf : i1
    %c_box = func.call @__ly_f64_box_inf(%c1) : (f64) -> f64
    %d_box = func.call @__ly_f64_box_inf(%d1) : (f64) -> f64
    %c2 = arith.select %w_inf, %c_box, %c1 : f64
    %d2 = arith.select %w_inf, %d_box, %d1 : f64
    %a2 = func.call @__ly_f64_nan_to_zero(%a1, %w_inf) : (f64, i1) -> f64
    %b2 = func.call @__ly_f64_nan_to_zero(%b1, %w_inf) : (f64, i1) -> f64
    // Overflow in a product.
    %recalc_inf = arith.ori %z_inf, %w_inf : i1
    %ac_inf = func.call @__ly_f64_isinf(%ac) : (f64) -> i1
    %bd_inf = func.call @__ly_f64_isinf(%bd) : (f64) -> i1
    %ad_inf = func.call @__ly_f64_isinf(%ad) : (f64) -> i1
    %bc_inf = func.call @__ly_f64_isinf(%bc) : (f64) -> i1
    %p0 = arith.ori %ac_inf, %bd_inf : i1
    %p1 = arith.ori %ad_inf, %bc_inf : i1
    %any_product_inf = arith.ori %p0, %p1 : i1
    %no_recalc = arith.xori %recalc_inf, %true : i1
    %overflowed = arith.andi %no_recalc, %any_product_inf : i1
    %a3 = func.call @__ly_f64_nan_to_zero(%a2, %overflowed) : (f64, i1) -> f64
    %b3 = func.call @__ly_f64_nan_to_zero(%b2, %overflowed) : (f64, i1) -> f64
    %c3 = func.call @__ly_f64_nan_to_zero(%c2, %overflowed) : (f64, i1) -> f64
    %d3 = func.call @__ly_f64_nan_to_zero(%d2, %overflowed) : (f64, i1) -> f64
    %recalc = arith.ori %recalc_inf, %overflowed : i1
    // INFINITY*(a*c - b*d), INFINITY*(a*d + b*c): one expression each.
    %bd3 = arith.mulf %b3, %d3 : f64
    %neg_bd3 = arith.negf %bd3 : f64
    %re_sum = llvm.intr.fmuladd(%a3, %c3, %neg_bd3) : (f64, f64, f64) -> f64
    %bc3 = arith.mulf %b3, %c3 : f64
    %im_sum = llvm.intr.fmuladd(%a3, %d3, %bc3) : (f64, f64, f64) -> f64
    %re_rec = arith.mulf %inf, %re_sum : f64
    %im_rec = arith.mulf %inf, %im_sum : f64
    %re_out = arith.select %recalc, %re_rec, %re : f64
    %im_out = arith.select %recalc, %im_rec, %im : f64
    func.return %re_out, %im_out : f64, f64
  }

  // _Py_c_quot. The third result is CPython's errno == EDOM.
  func.func private @__ly_c_quot(%a: f64, %b: f64, %c: f64, %d: f64) -> (f64, f64, i1) {
    %zero = arith.constant 0.0 : f64
    %nan = arith.constant 0x7FF8000000000000 : f64
    %false = arith.constant false
    %true = arith.constant true
    // `b.real < 0 ? -b.real : b.real`: -0.0 stays -0.0 and NaN stays NaN.
    %c_neg = arith.cmpf olt, %c, %zero : f64
    %c_flip = arith.negf %c : f64
    %abs_c = arith.select %c_neg, %c_flip, %c : f64
    %d_neg = arith.cmpf olt, %d, %zero : f64
    %d_flip = arith.negf %d : f64
    %abs_d = arith.select %d_neg, %d_flip, %d : f64
    %real_major = arith.cmpf oge, %abs_c, %abs_d : f64
    %r:3 = scf.if %real_major -> (f64, f64, i1) {
      %c_zero = arith.cmpf oeq, %abs_c, %zero : f64
      %q:3 = scf.if %c_zero -> (f64, f64, i1) {
        scf.yield %zero, %zero, %true : f64, f64, i1
      } else {
        %ratio = arith.divf %d, %c : f64
        %denom = llvm.intr.fmuladd(%d, %ratio, %c) : (f64, f64, f64) -> f64
        %re_num = llvm.intr.fmuladd(%b, %ratio, %a) : (f64, f64, f64) -> f64
        %neg_a = arith.negf %a : f64
        %im_num = llvm.intr.fmuladd(%neg_a, %ratio, %b) : (f64, f64, f64) -> f64
        %re = arith.divf %re_num, %denom : f64
        %im = arith.divf %im_num, %denom : f64
        scf.yield %re, %im, %false : f64, f64, i1
      }
      scf.yield %q#0, %q#1, %q#2 : f64, f64, i1
    } else {
      %imag_major = arith.cmpf oge, %abs_d, %abs_c : f64
      %q:3 = scf.if %imag_major -> (f64, f64, i1) {
        %ratio = arith.divf %c, %d : f64
        %denom = llvm.intr.fmuladd(%c, %ratio, %d) : (f64, f64, f64) -> f64
        %re_num = llvm.intr.fmuladd(%a, %ratio, %b) : (f64, f64, f64) -> f64
        %neg_a = arith.negf %a : f64
        %im_num = llvm.intr.fmuladd(%b, %ratio, %neg_a) : (f64, f64, f64) -> f64
        %re = arith.divf %re_num, %denom : f64
        %im = arith.divf %im_num, %denom : f64
        scf.yield %re, %im, %false : f64, f64, i1
      } else {
        scf.yield %nan, %nan, %false : f64, f64, i1
      }
      scf.yield %q#0, %q#1, %q#2 : f64, f64, i1
    }
    %re_nan = arith.cmpf uno, %r#0, %r#0 : f64
    %im_nan = arith.cmpf uno, %r#1, %r#1 : f64
    %both_nan = arith.andi %re_nan, %im_nan : i1
    %out:2 = scf.if %both_nan -> (f64, f64) {
      %rr:2 = func.call @__ly_c_quot_recover(%a, %b, %c, %d, %abs_c, %abs_d, %r#0, %r#1) : (f64, f64, f64, f64, f64, f64, f64, f64) -> (f64, f64)
      scf.yield %rr#0, %rr#1 : f64, f64
    } else {
      scf.yield %r#0, %r#1 : f64, f64
    }
    func.return %out#0, %out#1, %r#2 : f64, f64, i1
  }

  // The C11 Annex G.5.2 recovery `_Py_c_quot` runs when both parts are NaN.
  func.func private @__ly_c_quot_recover(%a: f64, %b: f64, %c: f64, %d: f64, %abs_c: f64, %abs_d: f64, %re: f64, %im: f64) -> (f64, f64) {
    %inf = arith.constant 0x7FF0000000000000 : f64
    %zero = arith.constant 0.0 : f64
    %a_inf = func.call @__ly_f64_isinf(%a) : (f64) -> i1
    %b_inf = func.call @__ly_f64_isinf(%b) : (f64) -> i1
    %num_inf = arith.ori %a_inf, %b_inf : i1
    %c_fin = func.call @__ly_f64_isfinite(%c) : (f64) -> i1
    %d_fin = func.call @__ly_f64_isfinite(%d) : (f64) -> i1
    %den_fin = arith.andi %c_fin, %d_fin : i1
    %first = arith.andi %num_inf, %den_fin : i1
    %out:2 = scf.if %first -> (f64, f64) {
      %x = func.call @__ly_f64_box_inf(%a) : (f64) -> f64
      %y = func.call @__ly_f64_box_inf(%b) : (f64) -> f64
      // INFINITY * (x*b.real + y*b.imag), INFINITY * (y*b.real - x*b.imag).
      %yd = arith.mulf %y, %d : f64
      %re_sum = llvm.intr.fmuladd(%x, %c, %yd) : (f64, f64, f64) -> f64
      %xd = arith.mulf %x, %d : f64
      %neg_xd = arith.negf %xd : f64
      %im_sum = llvm.intr.fmuladd(%y, %c, %neg_xd) : (f64, f64, f64) -> f64
      %re1 = arith.mulf %inf, %re_sum : f64
      %im1 = arith.mulf %inf, %im_sum : f64
      scf.yield %re1, %im1 : f64, f64
    } else {
      %abs_c_inf = func.call @__ly_f64_isinf(%abs_c) : (f64) -> i1
      %abs_d_inf = func.call @__ly_f64_isinf(%abs_d) : (f64) -> i1
      %den_inf = arith.ori %abs_c_inf, %abs_d_inf : i1
      %a_fin = func.call @__ly_f64_isfinite(%a) : (f64) -> i1
      %b_fin = func.call @__ly_f64_isfinite(%b) : (f64) -> i1
      %num_fin = arith.andi %a_fin, %b_fin : i1
      %second = arith.andi %den_inf, %num_fin : i1
      %o:2 = scf.if %second -> (f64, f64) {
        %x = func.call @__ly_f64_box_inf(%c) : (f64) -> f64
        %y = func.call @__ly_f64_box_inf(%d) : (f64) -> f64
        // 0.0 * (a.real*x + a.imag*y), 0.0 * (a.imag*x - a.real*y).
        %by = arith.mulf %b, %y : f64
        %re_sum = llvm.intr.fmuladd(%a, %x, %by) : (f64, f64, f64) -> f64
        %ay = arith.mulf %a, %y : f64
        %neg_ay = arith.negf %ay : f64
        %im_sum = llvm.intr.fmuladd(%b, %x, %neg_ay) : (f64, f64, f64) -> f64
        %re2 = arith.mulf %zero, %re_sum : f64
        %im2 = arith.mulf %zero, %im_sum : f64
        scf.yield %re2, %im2 : f64, f64
      } else {
        scf.yield %re, %im : f64, f64
      }
      scf.yield %o#0, %o#1 : f64, f64
    }
    func.return %out#0, %out#1 : f64, f64
  }

  // _Py_rc_quot: a real numerator over a complex denominator.
  func.func private @__ly_rc_quot(%a: f64, %c: f64, %d: f64) -> (f64, f64, i1) {
    %zero = arith.constant 0.0 : f64
    %nan = arith.constant 0x7FF8000000000000 : f64
    %false = arith.constant false
    %true = arith.constant true
    %c_neg = arith.cmpf olt, %c, %zero : f64
    %c_flip = arith.negf %c : f64
    %abs_c = arith.select %c_neg, %c_flip, %c : f64
    %d_neg = arith.cmpf olt, %d, %zero : f64
    %d_flip = arith.negf %d : f64
    %abs_d = arith.select %d_neg, %d_flip, %d : f64
    %neg_a = arith.negf %a : f64
    %real_major = arith.cmpf oge, %abs_c, %abs_d : f64
    %r:3 = scf.if %real_major -> (f64, f64, i1) {
      %c_zero = arith.cmpf oeq, %abs_c, %zero : f64
      %q:3 = scf.if %c_zero -> (f64, f64, i1) {
        scf.yield %zero, %zero, %true : f64, f64, i1
      } else {
        %ratio = arith.divf %d, %c : f64
        %denom = llvm.intr.fmuladd(%d, %ratio, %c) : (f64, f64, f64) -> f64
        %re = arith.divf %a, %denom : f64
        %im_num = arith.mulf %neg_a, %ratio : f64
        %im = arith.divf %im_num, %denom : f64
        scf.yield %re, %im, %false : f64, f64, i1
      }
      scf.yield %q#0, %q#1, %q#2 : f64, f64, i1
    } else {
      %imag_major = arith.cmpf oge, %abs_d, %abs_c : f64
      %q:3 = scf.if %imag_major -> (f64, f64, i1) {
        %ratio = arith.divf %c, %d : f64
        %denom = llvm.intr.fmuladd(%c, %ratio, %d) : (f64, f64, f64) -> f64
        %re_num = arith.mulf %a, %ratio : f64
        %re = arith.divf %re_num, %denom : f64
        %im = arith.divf %neg_a, %denom : f64
        scf.yield %re, %im, %false : f64, f64, i1
      } else {
        scf.yield %nan, %nan, %false : f64, f64, i1
      }
      scf.yield %q#0, %q#1, %q#2 : f64, f64, i1
    }
    %re_nan = arith.cmpf uno, %r#0, %r#0 : f64
    %im_nan = arith.cmpf uno, %r#1, %r#1 : f64
    %both_nan = arith.andi %re_nan, %im_nan : i1
    %a_fin = func.call @__ly_f64_isfinite(%a) : (f64) -> i1
    %abs_c_inf = func.call @__ly_f64_isinf(%abs_c) : (f64) -> i1
    %abs_d_inf = func.call @__ly_f64_isinf(%abs_d) : (f64) -> i1
    %den_inf = arith.ori %abs_c_inf, %abs_d_inf : i1
    %nan_fin = arith.andi %both_nan, %a_fin : i1
    %recover = arith.andi %nan_fin, %den_inf : i1
    %x = func.call @__ly_f64_box_inf(%c) : (f64) -> f64
    %y = func.call @__ly_f64_box_inf(%d) : (f64) -> f64
    %ax = arith.mulf %a, %x : f64
    %re_rec = arith.mulf %zero, %ax : f64
    %nay = arith.mulf %neg_a, %y : f64
    %im_rec = arith.mulf %zero, %nay : f64
    %re_out = arith.select %recover, %re_rec, %r#0 : f64
    %im_out = arith.select %recover, %im_rec, %r#1 : f64
    func.return %re_out, %im_out, %r#2 : f64, f64, i1
  }

  // c_powu: binary powering by repeated _Py_c_prod.
  func.func private @__ly_c_powu(%a: f64, %b: f64, %n: i64) -> (f64, f64) {
    %one = arith.constant 1.0 : f64
    %zero = arith.constant 0.0 : f64
    %mask0 = arith.constant 1 : i64
    %i0 = arith.constant 0 : i64
    %r:6 = scf.while (%mask = %mask0, %rr = %one, %ri = %zero, %pr = %a, %pi = %b, %done = %i0) : (i64, f64, f64, f64, f64, i64) -> (i64, f64, f64, f64, f64, i64) {
      %positive = arith.cmpi sgt, %mask, %i0 : i64
      %reach = arith.cmpi sge, %n, %mask : i64
      %go = arith.andi %positive, %reach : i1
      scf.condition(%go) %mask, %rr, %ri, %pr, %pi, %done : i64, f64, f64, f64, f64, i64
    } do {
    ^bb0(%mask: i64, %rr: f64, %ri: f64, %pr: f64, %pi: f64, %done: i64):
      %bit = arith.andi %n, %mask : i64
      %has = arith.cmpi ne, %bit, %i0 : i64
      %next:2 = scf.if %has -> (f64, f64) {
        %m:2 = func.call @__ly_c_prod(%rr, %ri, %pr, %pi) : (f64, f64, f64, f64) -> (f64, f64)
        scf.yield %m#0, %m#1 : f64, f64
      } else {
        scf.yield %rr, %ri : f64, f64
      }
      %shift = arith.constant 1 : i64
      %mask_next = arith.shli %mask, %shift : i64
      %sq:2 = func.call @__ly_c_prod(%pr, %pi, %pr, %pi) : (f64, f64, f64, f64) -> (f64, f64)
      scf.yield %mask_next, %next#0, %next#1, %sq#0, %sq#1, %done : i64, f64, f64, f64, f64, i64
    }
    func.return %r#1, %r#2 : f64, f64
  }

  // _Py_c_pow. The third result is CPython's errno: 0, 1 = EDOM, 2 = ERANGE
  // (after _Py_ADJUST_ERANGE2: ERANGE exactly when a part is infinite).
  func.func private @__ly_c_pow(%a: f64, %b: f64, %c: f64, %d: f64) -> (f64, f64, i64) {
    %zero = arith.constant 0.0 : f64
    %one = arith.constant 1.0 : f64
    %ok = arith.constant 0 : i64
    %edom = arith.constant 1 : i64
    %erange = arith.constant 2 : i64
    %c_zero = arith.cmpf oeq, %c, %zero : f64
    %d_zero = arith.cmpf oeq, %d, %zero : f64
    %exp_zero = arith.andi %c_zero, %d_zero : i1
    %r:3 = scf.if %exp_zero -> (f64, f64, i64) {
      scf.yield %one, %zero, %ok : f64, f64, i64
    } else {
      %a_zero = arith.cmpf oeq, %a, %zero : f64
      %b_zero = arith.cmpf oeq, %b, %zero : f64
      %base_zero = arith.andi %a_zero, %b_zero : i1
      %q:3 = scf.if %base_zero -> (f64, f64, i64) {
        %d_nonzero = arith.cmpf une, %d, %zero : f64
        %c_negative = arith.cmpf olt, %c, %zero : f64
        %bad = arith.ori %d_nonzero, %c_negative : i1
        %err = arith.select %bad, %edom, %ok : i64
        scf.yield %zero, %zero, %err : f64, f64, i64
      } else {
        %vabs = func.call @hypot(%a, %b) : (f64, f64) -> f64
        %len0 = math.powf %vabs, %c : f64
        %at = math.atan2 %b, %a : f64
        %phase0 = arith.mulf %at, %c : f64
        %d_nonzero = arith.cmpf une, %d, %zero : f64
        %lp:2 = scf.if %d_nonzero -> (f64, f64) {
          %neg_at = arith.negf %at : f64
          %e_arg = arith.mulf %neg_at, %d : f64
          %e = math.exp %e_arg : f64
          %len1 = arith.mulf %len0, %e : f64
          %lg = math.log %vabs : f64
          // phase += b.imag*log(vabs): one expression.
          %phase1 = llvm.intr.fmuladd(%d, %lg, %phase0) : (f64, f64, f64) -> f64
          scf.yield %len1, %phase1 : f64, f64
        } else {
          scf.yield %len0, %phase0 : f64, f64
        }
        %cos = math.cos %lp#1 : f64
        %sin = math.sin %lp#1 : f64
        %re = arith.mulf %lp#0, %cos : f64
        %im = arith.mulf %lp#0, %sin : f64
        %re_inf = func.call @__ly_f64_isinf(%re) : (f64) -> i1
        %im_inf = func.call @__ly_f64_isinf(%im) : (f64) -> i1
        %any_inf = arith.ori %re_inf, %im_inf : i1
        %err = arith.select %any_inf, %erange, %ok : i64
        scf.yield %re, %im, %err : f64, f64, i64
      }
      scf.yield %q#0, %q#1, %q#2 : f64, f64, i64
    }
    func.return %r#0, %r#1, %r#2 : f64, f64, i64
  }

  // complex_pow: c_powi for a small integral exponent, else _Py_c_pow; the
  // third result as in __ly_c_pow.
  func.func private @__ly_complex_pow(%a: f64, %b: f64, %c: f64, %d: f64) -> (f64, f64, i64) {
    %zero = arith.constant 0.0 : f64
    %one = arith.constant 1.0 : f64
    %hundred = arith.constant 100.0 : f64
    %ok = arith.constant 0 : i64
    %edom = arith.constant 1 : i64
    %erange = arith.constant 2 : i64
    %d_zero = arith.cmpf oeq, %d, %zero : f64
    %fl = math.floor %c : f64
    %integral = arith.cmpf oeq, %c, %fl : f64
    %mag = math.absf %c : f64
    %small = arith.cmpf ole, %mag, %hundred : f64
    %int_exp0 = arith.andi %d_zero, %integral : i1
    %int_exp = arith.andi %int_exp0, %small : i1
    %r:3 = scf.if %int_exp -> (f64, f64, i64) {
      %n = arith.fptosi %c : f64 to i64
      %i0 = arith.constant 0 : i64
      %positive = arith.cmpi sgt, %n, %i0 : i64
      %p:3 = scf.if %positive -> (f64, f64, i1) {
        %u:2 = func.call @__ly_c_powu(%a, %b, %n) : (f64, f64, i64) -> (f64, f64)
        %false = arith.constant false
        scf.yield %u#0, %u#1, %false : f64, f64, i1
      } else {
        %neg_n = arith.subi %i0, %n : i64
        %u:2 = func.call @__ly_c_powu(%a, %b, %neg_n) : (f64, f64, i64) -> (f64, f64)
        %q:3 = func.call @__ly_c_quot(%one, %zero, %u#0, %u#1) : (f64, f64, f64, f64) -> (f64, f64, i1)
        scf.yield %q#0, %q#1, %q#2 : f64, f64, i1
      }
      // _Py_ADJUST_ERANGE2 over errno = EDOM-or-0.
      %re_inf = func.call @__ly_f64_isinf(%p#0) : (f64) -> i1
      %im_inf = func.call @__ly_f64_isinf(%p#1) : (f64) -> i1
      %any_inf = arith.ori %re_inf, %im_inf : i1
      %range_err = arith.select %any_inf, %erange, %ok : i64
      %err = arith.select %p#2, %edom, %range_err : i64
      scf.yield %p#0, %p#1, %err : f64, f64, i64
    } else {
      %q:3 = func.call @__ly_c_pow(%a, %b, %c, %d) : (f64, f64, f64, f64) -> (f64, f64, i64)
      scf.yield %q#0, %q#1, %q#2 : f64, f64, i64
    }
    func.return %r#0, %r#1, %r#2 : f64, f64, i64
  }

  // _Py_c_abs; the second result is errno == ERANGE.
  func.func private @__ly_c_abs(%a: f64, %b: f64) -> (f64, i1) {
    %nan = arith.constant 0x7FF8000000000000 : f64
    %false = arith.constant false
    %a_fin = func.call @__ly_f64_isfinite(%a) : (f64) -> i1
    %b_fin = func.call @__ly_f64_isfinite(%b) : (f64) -> i1
    %finite = arith.andi %a_fin, %b_fin : i1
    %r:2 = scf.if %finite -> (f64, i1) {
      %h = func.call @hypot(%a, %b) : (f64, f64) -> f64
      %h_fin = func.call @__ly_f64_isfinite(%h) : (f64) -> i1
      %true = arith.constant true
      %range = arith.xori %h_fin, %true : i1
      scf.yield %h, %range : f64, i1
    } else {
      // C99: an infinite part wins even over a NaN.
      %a_inf = func.call @__ly_f64_isinf(%a) : (f64) -> i1
      %b_inf = func.call @__ly_f64_isinf(%b) : (f64) -> i1
      %abs_a = math.absf %a : f64
      %abs_b = math.absf %b : f64
      %from_b = arith.select %b_inf, %abs_b, %nan : f64
      %v = arith.select %a_inf, %abs_a, %from_b : f64
      scf.yield %v, %false : f64, i1
    }
    func.return %r#0, %r#1 : f64, i1
  }

  func.func @LyComplex_AddReal(%self: memref<4xi64> {ly.ownership.object_header}, %other: memref<3xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.complex"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.complex", ly.runtime.method = "__add__"} {
    %z:2 = func.call @__ly_complex_parts(%self) : (memref<4xi64>) -> (f64, f64)
    %x = func.call @LyFloat_AsF64(%other) : (memref<3xi64>) -> f64
    %re = arith.addf %z#0, %x : f64
    %out = func.call @LyComplex_FromParts(%re, %z#1) : (f64, f64) -> memref<4xi64>
    func.return %out : memref<4xi64>
  }

  func.func @LyComplex_RAddReal(%self: memref<4xi64> {ly.ownership.object_header}, %other: memref<3xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.complex"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.complex", ly.runtime.method = "__radd__"} {
    %z:2 = func.call @__ly_complex_parts(%self) : (memref<4xi64>) -> (f64, f64)
    %x = func.call @LyFloat_AsF64(%other) : (memref<3xi64>) -> f64
    %re = arith.addf %z#0, %x : f64
    %out = func.call @LyComplex_FromParts(%re, %z#1) : (f64, f64) -> memref<4xi64>
    func.return %out : memref<4xi64>
  }

  func.func @LyComplex_SubReal(%self: memref<4xi64> {ly.ownership.object_header}, %other: memref<3xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.complex"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.complex", ly.runtime.method = "__sub__"} {
    %z:2 = func.call @__ly_complex_parts(%self) : (memref<4xi64>) -> (f64, f64)
    %x = func.call @LyFloat_AsF64(%other) : (memref<3xi64>) -> f64
    %re = arith.subf %z#0, %x : f64
    %out = func.call @LyComplex_FromParts(%re, %z#1) : (f64, f64) -> memref<4xi64>
    func.return %out : memref<4xi64>
  }

  func.func @LyComplex_RSubReal(%self: memref<4xi64> {ly.ownership.object_header}, %other: memref<3xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.complex"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.complex", ly.runtime.method = "__rsub__"} {
    %z:2 = func.call @__ly_complex_parts(%self) : (memref<4xi64>) -> (f64, f64)
    %x = func.call @LyFloat_AsF64(%other) : (memref<3xi64>) -> f64
    %re = arith.subf %x, %z#0 : f64
    %im = arith.negf %z#1 : f64
    %out = func.call @LyComplex_FromParts(%re, %im) : (f64, f64) -> memref<4xi64>
    func.return %out : memref<4xi64>
  }

  func.func @LyComplex_MulReal(%self: memref<4xi64> {ly.ownership.object_header}, %other: memref<3xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.complex"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.complex", ly.runtime.method = "__mul__"} {
    %z:2 = func.call @__ly_complex_parts(%self) : (memref<4xi64>) -> (f64, f64)
    %x = func.call @LyFloat_AsF64(%other) : (memref<3xi64>) -> f64
    %re = arith.mulf %z#0, %x : f64
    %im = arith.mulf %z#1, %x : f64
    %out = func.call @LyComplex_FromParts(%re, %im) : (f64, f64) -> memref<4xi64>
    func.return %out : memref<4xi64>
  }

  func.func @LyComplex_RMulReal(%self: memref<4xi64> {ly.ownership.object_header}, %other: memref<3xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.complex"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.complex", ly.runtime.method = "__rmul__"} {
    %z:2 = func.call @__ly_complex_parts(%self) : (memref<4xi64>) -> (f64, f64)
    %x = func.call @LyFloat_AsF64(%other) : (memref<3xi64>) -> f64
    %re = arith.mulf %z#0, %x : f64
    %im = arith.mulf %z#1, %x : f64
    %out = func.call @LyComplex_FromParts(%re, %im) : (f64, f64) -> memref<4xi64>
    func.return %out : memref<4xi64>
  }

  func.func @LyComplex_Mul(%lhs_header: memref<4xi64> {ly.ownership.object_header}, %rhs_header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.complex"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.complex", ly.runtime.method = "__mul__"} {
    %z:2 = func.call @__ly_complex_parts(%lhs_header) : (memref<4xi64>) -> (f64, f64)
    %w:2 = func.call @__ly_complex_parts(%rhs_header) : (memref<4xi64>) -> (f64, f64)
    %r:2 = func.call @__ly_c_prod(%z#0, %z#1, %w#0, %w#1) : (f64, f64, f64, f64) -> (f64, f64)
    %out = func.call @LyComplex_FromParts(%r#0, %r#1) : (f64, f64) -> memref<4xi64>
    func.return %out : memref<4xi64>
  }

  func.func @LyComplex_TrueDiv(%lhs_header: memref<4xi64> {ly.ownership.object_header}, %rhs_header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.complex"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.complex", ly.runtime.method = "__truediv__"} {
    %z:2 = func.call @__ly_complex_parts(%lhs_header) : (memref<4xi64>) -> (f64, f64)
    %w:2 = func.call @__ly_complex_parts(%rhs_header) : (memref<4xi64>) -> (f64, f64)
    %r:3 = func.call @__ly_c_quot(%z#0, %z#1, %w#0, %w#1) : (f64, f64, f64, f64) -> (f64, f64, i1)
    scf.if %r#2 {
      func.call @__ly_complex_raise_zero_division() : () -> ()
    }
    %out = func.call @LyComplex_FromParts(%r#0, %r#1) : (f64, f64) -> memref<4xi64>
    func.return %out : memref<4xi64>
  }

  func.func @LyComplex_TrueDivReal(%self: memref<4xi64> {ly.ownership.object_header}, %other: memref<3xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.complex"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.complex", ly.runtime.method = "__truediv__"} {
    %z:2 = func.call @__ly_complex_parts(%self) : (memref<4xi64>) -> (f64, f64)
    %x = func.call @LyFloat_AsF64(%other) : (memref<3xi64>) -> f64
    %zero = arith.constant 0.0 : f64
    // `if (b)`: NaN is nonzero.
    %is_zero = arith.cmpf oeq, %x, %zero : f64
    scf.if %is_zero {
      func.call @__ly_complex_raise_zero_division() : () -> ()
    }
    %re = arith.divf %z#0, %x : f64
    %im = arith.divf %z#1, %x : f64
    %out = func.call @LyComplex_FromParts(%re, %im) : (f64, f64) -> memref<4xi64>
    func.return %out : memref<4xi64>
  }

  func.func @LyComplex_RTrueDivReal(%self: memref<4xi64> {ly.ownership.object_header}, %other: memref<3xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.complex"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.complex", ly.runtime.method = "__rtruediv__"} {
    %z:2 = func.call @__ly_complex_parts(%self) : (memref<4xi64>) -> (f64, f64)
    %x = func.call @LyFloat_AsF64(%other) : (memref<3xi64>) -> f64
    %r:3 = func.call @__ly_rc_quot(%x, %z#0, %z#1) : (f64, f64, f64) -> (f64, f64, i1)
    scf.if %r#2 {
      func.call @__ly_complex_raise_zero_division() : () -> ()
    }
    %out = func.call @LyComplex_FromParts(%r#0, %r#1) : (f64, f64) -> memref<4xi64>
    func.return %out : memref<4xi64>
  }

  func.func @LyComplex_Pow(%lhs_header: memref<4xi64> {ly.ownership.object_header}, %rhs_header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.complex"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.complex", ly.runtime.method = "__pow__"} {
    %z:2 = func.call @__ly_complex_parts(%lhs_header) : (memref<4xi64>) -> (f64, f64)
    %w:2 = func.call @__ly_complex_parts(%rhs_header) : (memref<4xi64>) -> (f64, f64)
    %r:3 = func.call @__ly_complex_pow(%z#0, %z#1, %w#0, %w#1) : (f64, f64, f64, f64) -> (f64, f64, i64)
    %edom = arith.constant 1 : i64
    %erange = arith.constant 2 : i64
    %is_edom = arith.cmpi eq, %r#2, %edom : i64
    scf.if %is_edom {
      func.call @__ly_complex_raise_pow_zero() : () -> ()
    }
    %is_erange = arith.cmpi eq, %r#2, %erange : i64
    scf.if %is_erange {
      func.call @__ly_complex_raise_pow_overflow() : () -> ()
    }
    %out = func.call @LyComplex_FromParts(%r#0, %r#1) : (f64, f64) -> memref<4xi64>
    func.return %out : memref<4xi64>
  }

  func.func @LyComplex_PowReal(%self: memref<4xi64> {ly.ownership.object_header}, %other: memref<3xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.complex"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.complex", ly.runtime.method = "__pow__"} {
    %z:2 = func.call @__ly_complex_parts(%self) : (memref<4xi64>) -> (f64, f64)
    %x = func.call @LyFloat_AsF64(%other) : (memref<3xi64>) -> f64
    %zero = arith.constant 0.0 : f64
    %r:3 = func.call @__ly_complex_pow(%z#0, %z#1, %x, %zero) : (f64, f64, f64, f64) -> (f64, f64, i64)
    %edom = arith.constant 1 : i64
    %erange = arith.constant 2 : i64
    %is_edom = arith.cmpi eq, %r#2, %edom : i64
    scf.if %is_edom {
      func.call @__ly_complex_raise_pow_zero() : () -> ()
    }
    %is_erange = arith.cmpi eq, %r#2, %erange : i64
    scf.if %is_erange {
      func.call @__ly_complex_raise_pow_overflow() : () -> ()
    }
    %out = func.call @LyComplex_FromParts(%r#0, %r#1) : (f64, f64) -> memref<4xi64>
    func.return %out : memref<4xi64>
  }

  func.func @LyComplex_RPowReal(%self: memref<4xi64> {ly.ownership.object_header}, %other: memref<3xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.complex"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.complex", ly.runtime.method = "__rpow__"} {
    %z:2 = func.call @__ly_complex_parts(%self) : (memref<4xi64>) -> (f64, f64)
    %x = func.call @LyFloat_AsF64(%other) : (memref<3xi64>) -> f64
    %zero = arith.constant 0.0 : f64
    %r:3 = func.call @__ly_complex_pow(%x, %zero, %z#0, %z#1) : (f64, f64, f64, f64) -> (f64, f64, i64)
    %edom = arith.constant 1 : i64
    %erange = arith.constant 2 : i64
    %is_edom = arith.cmpi eq, %r#2, %edom : i64
    scf.if %is_edom {
      func.call @__ly_complex_raise_pow_zero() : () -> ()
    }
    %is_erange = arith.cmpi eq, %r#2, %erange : i64
    scf.if %is_erange {
      func.call @__ly_complex_raise_pow_overflow() : () -> ()
    }
    %out = func.call @LyComplex_FromParts(%r#0, %r#1) : (f64, f64) -> memref<4xi64>
    func.return %out : memref<4xi64>
  }

  func.func @LyComplex_Abs(%header: memref<4xi64> {ly.ownership.object_header}) -> memref<3xi64> attributes {ly.runtime.contract = "builtins.complex", ly.runtime.method = "__abs__", ly.ownership.owned_results = [0], ly.runtime.result_contract = "builtins.float"} {
    %z:2 = func.call @__ly_complex_parts(%header) : (memref<4xi64>) -> (f64, f64)
    %r:2 = func.call @__ly_c_abs(%z#0, %z#1) : (f64, f64) -> (f64, i1)
    scf.if %r#1 {
      func.call @__ly_complex_raise_abs_overflow() : () -> ()
    }
    %h = func.call @LyFloat_FromF64(%r#0) : (f64) -> memref<3xi64>
    func.return %h : memref<3xi64>
  }

  // complex_bool: either part nonzero (a NaN part is nonzero).
  func.func @LyComplex_Bool(%header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.complex", ly.runtime.method = "__bool__"} {
    %z:2 = func.call @__ly_complex_parts(%header) : (memref<4xi64>) -> (f64, f64)
    %zero = arith.constant 0.0 : f64
    %re_nonzero = arith.cmpf une, %z#0, %zero : f64
    %im_nonzero = arith.cmpf une, %z#1, %zero : f64
    %truth = arith.ori %re_nonzero, %im_nonzero : i1
    func.return %truth : i1
  }

  func.func @LyComplex_Conjugate(%header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.complex"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.complex", ly.runtime.method = "conjugate"} {
    %z:2 = func.call @__ly_complex_parts(%header) : (memref<4xi64>) -> (f64, f64)
    %im = arith.negf %z#1 : f64
    %out = func.call @LyComplex_FromParts(%z#0, %im) : (f64, f64) -> memref<4xi64>
    func.return %out : memref<4xi64>
  }

  func.func @LyComplex_EqFloatBool(%self: memref<4xi64> {ly.ownership.object_header}, %other: memref<3xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.complex", ly.runtime.method = "__eq__"} {
    %z:2 = func.call @__ly_complex_parts(%self) : (memref<4xi64>) -> (f64, f64)
    %x = func.call @LyFloat_AsF64(%other) : (memref<3xi64>) -> f64
    %zero = arith.constant 0.0 : f64
    %re_eq = arith.cmpf oeq, %z#0, %x : f64
    %im_zero = arith.cmpf oeq, %z#1, %zero : f64
    %equal = arith.andi %re_eq, %im_zero : i1
    func.return %equal : i1
  }

  func.func @LyComplex_NeFloatBool(%self: memref<4xi64> {ly.ownership.object_header}, %other: memref<3xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.complex", ly.runtime.method = "__ne__"} {
    %z:2 = func.call @__ly_complex_parts(%self) : (memref<4xi64>) -> (f64, f64)
    %x = func.call @LyFloat_AsF64(%other) : (memref<3xi64>) -> f64
    %zero = arith.constant 0.0 : f64
    %re_eq = arith.cmpf oeq, %z#0, %x : f64
    %im_zero = arith.cmpf oeq, %z#1, %zero : f64
    %equal = arith.andi %re_eq, %im_zero : i1
    %true = arith.constant true
    %result = arith.xori %equal, %true : i1
    func.return %result : i1
  }

  func.func @LyComplex_EqLongBool(%self: memref<4xi64> {ly.ownership.object_header}, %other: memref<2xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.complex", ly.runtime.method = "__eq__"} {
    %z:2 = func.call @__ly_complex_parts(%self) : (memref<4xi64>) -> (f64, f64)
    %rhs_meta, %rhs_digits = func.call @__ly_long_parts(%other) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %cmp = func.call @__ly_long_cmp_f64(%rhs_meta, %rhs_digits, %z#0) : (memref<2xi64>, memref<?xi32>, f64) -> i64
    %i0 = arith.constant 0 : i64
    %re_eq = arith.cmpi eq, %cmp, %i0 : i64
    %zero = arith.constant 0.0 : f64
    %im_zero = arith.cmpf oeq, %z#1, %zero : f64
    %equal = arith.andi %re_eq, %im_zero : i1
    func.return %equal : i1
  }

  func.func @LyComplex_NeLongBool(%self: memref<4xi64> {ly.ownership.object_header}, %other: memref<2xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.complex", ly.runtime.method = "__ne__"} {
    %z:2 = func.call @__ly_complex_parts(%self) : (memref<4xi64>) -> (f64, f64)
    %rhs_meta, %rhs_digits = func.call @__ly_long_parts(%other) : (memref<2xi64>) -> (memref<2xi64>, memref<?xi32>)
    %cmp = func.call @__ly_long_cmp_f64(%rhs_meta, %rhs_digits, %z#0) : (memref<2xi64>, memref<?xi32>, f64) -> i64
    %i0 = arith.constant 0 : i64
    %re_eq = arith.cmpi eq, %cmp, %i0 : i64
    %zero = arith.constant 0.0 : f64
    %im_zero = arith.cmpf oeq, %z#1, %zero : f64
    %equal = arith.andi %re_eq, %im_zero : i1
    %true = arith.constant true
    %result = arith.xori %equal, %true : i1
    func.return %result : i1
  }

  func.func @LyComplex_Real(%header: memref<4xi64> {ly.ownership.object_header}) -> memref<3xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.complex", ly.runtime.primitive = "property.real", ly.runtime.result_contract = "builtins.float"} {
    %z:2 = func.call @__ly_complex_parts(%header) : (memref<4xi64>) -> (f64, f64)
    %h = func.call @LyFloat_FromF64(%z#0) : (f64) -> memref<3xi64>
    func.return %h : memref<3xi64>
  }

  func.func @LyComplex_Imag(%header: memref<4xi64> {ly.ownership.object_header}) -> memref<3xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.complex", ly.runtime.primitive = "property.imag", ly.runtime.result_contract = "builtins.float"} {
    %z:2 = func.call @__ly_complex_parts(%header) : (memref<4xi64>) -> (f64, f64)
    %h = func.call @LyFloat_FromF64(%z#1) : (f64) -> memref<3xi64>
    func.return %h : memref<3xi64>
  }

  memref.global "private" constant @__ly_complex_msg_zero_div : memref<16xi8> = dense<[100, 105, 118, 105, 115, 105, 111, 110, 32, 98, 121, 32, 122, 101, 114, 111]>
  func.func private @__ly_complex_raise_zero_division() {
    %class_id = arith.constant {ly.class_id_of = "builtins.ZeroDivisionError"} 61 : i64
    %length = arith.constant 16 : i64
    %message_static = memref.get_global @__ly_complex_msg_zero_div : memref<16xi8>
    %message = memref.cast %message_static : memref<16xi8> to memref<?xi8>
    func.call @__ly_raise_static_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  memref.global "private" constant @__ly_complex_msg_pow_zero : memref<35xi8> = dense<[122, 101, 114, 111, 32, 116, 111, 32, 97, 32, 110, 101, 103, 97, 116, 105, 118, 101, 32, 111, 114, 32, 99, 111, 109, 112, 108, 101, 120, 32, 112, 111, 119, 101, 114]>
  func.func private @__ly_complex_raise_pow_zero() {
    %class_id = arith.constant {ly.class_id_of = "builtins.ZeroDivisionError"} 61 : i64
    %length = arith.constant 35 : i64
    %message_static = memref.get_global @__ly_complex_msg_pow_zero : memref<35xi8>
    %message = memref.cast %message_static : memref<35xi8> to memref<?xi8>
    func.call @__ly_raise_static_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  memref.global "private" constant @__ly_complex_msg_pow_overflow : memref<22xi8> = dense<[99, 111, 109, 112, 108, 101, 120, 32, 101, 120, 112, 111, 110, 101, 110, 116, 105, 97, 116, 105, 111, 110]>
  func.func private @__ly_complex_raise_pow_overflow() {
    %class_id = arith.constant {ly.class_id_of = "builtins.OverflowError"} 104 : i64
    %length = arith.constant 22 : i64
    %message_static = memref.get_global @__ly_complex_msg_pow_overflow : memref<22xi8>
    %message = memref.cast %message_static : memref<22xi8> to memref<?xi8>
    func.call @__ly_raise_static_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  memref.global "private" constant @__ly_complex_msg_abs_overflow : memref<24xi8> = dense<[97, 98, 115, 111, 108, 117, 116, 101, 32, 118, 97, 108, 117, 101, 32, 116, 111, 111, 32, 108, 97, 114, 103, 101]>
  func.func private @__ly_complex_raise_abs_overflow() {
    %class_id = arith.constant {ly.class_id_of = "builtins.OverflowError"} 104 : i64
    %length = arith.constant 24 : i64
    %message_static = memref.get_global @__ly_complex_msg_abs_overflow : memref<24xi8>
    %message = memref.cast %message_static : memref<24xi8> to memref<?xi8>
    func.call @__ly_raise_static_message(%class_id, %message, %length) : (i64, memref<?xi8>, i64) -> ()
    func.return
  }

  // Textbook quotient (not Smith's algorithm): CPython's overflow/underflow
  // edge behavior for extreme components is out of scope for the first cut.

  func.func @LyComplex_Neg(%header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.complex"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.complex", ly.runtime.method = "__neg__"} {
    %re_slot = arith.constant 2 : index
    %im_slot = arith.constant 3 : index
    %a_bits = memref.load %header[%re_slot] : memref<4xi64>
    %a = arith.bitcast %a_bits : i64 to f64
    %b_bits = memref.load %header[%im_slot] : memref<4xi64>
    %b = arith.bitcast %b_bits : i64 to f64
    %re = arith.negf %a : f64
    %im = arith.negf %b : f64
    %out_header = func.call @LyComplex_FromParts(%re, %im) : (f64, f64) -> memref<4xi64>
    func.return %out_header : memref<4xi64>
  }

  func.func @LyComplex_Pos(%header: memref<4xi64> {ly.ownership.object_header}) -> memref<4xi64> attributes {ly.ownership.owned_result_contracts = ["builtins.complex"], ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.complex", ly.runtime.method = "__pos__"} {
    %re_slot = arith.constant 2 : index
    %im_slot = arith.constant 3 : index
    %a_bits = memref.load %header[%re_slot] : memref<4xi64>
    %a = arith.bitcast %a_bits : i64 to f64
    %b_bits = memref.load %header[%im_slot] : memref<4xi64>
    %b = arith.bitcast %b_bits : i64 to f64
    %out_header = func.call @LyComplex_FromParts(%a, %b) : (f64, f64) -> memref<4xi64>
    func.return %out_header : memref<4xi64>
  }

  func.func @LyComplex_EqBool(%lhs_header: memref<4xi64> {ly.ownership.object_header}, %rhs_header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.complex", ly.runtime.method = "__eq__"} {
    %re_slot = arith.constant 2 : index
    %im_slot = arith.constant 3 : index
    %a_bits = memref.load %lhs_header[%re_slot] : memref<4xi64>
    %a = arith.bitcast %a_bits : i64 to f64
    %b_bits = memref.load %lhs_header[%im_slot] : memref<4xi64>
    %b = arith.bitcast %b_bits : i64 to f64
    %c_bits = memref.load %rhs_header[%re_slot] : memref<4xi64>
    %c = arith.bitcast %c_bits : i64 to f64
    %d_bits = memref.load %rhs_header[%im_slot] : memref<4xi64>
    %d = arith.bitcast %d_bits : i64 to f64
    %re_eq = arith.cmpf oeq, %a, %c : f64
    %im_eq = arith.cmpf oeq, %b, %d : f64
    %eq = arith.andi %re_eq, %im_eq : i1
    func.return %eq : i1
  }

  func.func @LyComplex_NeBool(%lhs_header: memref<4xi64> {ly.ownership.object_header}, %rhs_header: memref<4xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.contract = "builtins.complex", ly.runtime.method = "__ne__"} {
    %eq = func.call @LyComplex_EqBool(%lhs_header, %rhs_header) : (memref<4xi64>, memref<4xi64>) -> i1
    %one = arith.constant true
    %ne = arith.xori %eq, %one : i1
    func.return %ne : i1
  }

  // One component as text: float repr with CPython's complex-component rule
  // (format code 'r' without ADD_DOT_0) approximated by stripping a ".0"
  // suffix. Returns an owned str.
  func.func private @__ly_complex_component_repr(%value: f64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_result_contracts = ["builtins.str"], ly.ownership.owned_results = [0]} {
    %float_header = func.call @LyFloat_FromF64(%value) : (f64) -> memref<3xi64>
    %repr_header, %repr_bytes = func.call @LyFloat_Repr(%float_header) : (memref<3xi64>) -> (memref<2xi64>, memref<?xi8>)
    func.call @LyFloat_DecRef(%float_header) : (memref<3xi64>) -> ()
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %n = memref.dim %repr_bytes, %c0 : memref<?xi8>
    %dot = arith.constant 46 : i8
    %zero_char = arith.constant 48 : i8
    %long_enough = arith.cmpi sge, %n, %c2 : index
    %strip = scf.if %long_enough -> (i1) {
      %last = arith.subi %n, %c1 : index
      %second_last = arith.subi %n, %c2 : index
      %tail = memref.load %repr_bytes[%last] : memref<?xi8>
      %before = memref.load %repr_bytes[%second_last] : memref<?xi8>
      %tail_zero = arith.cmpi eq, %tail, %zero_char : i8
      %before_dot = arith.cmpi eq, %before, %dot : i8
      %both = arith.andi %tail_zero, %before_dot : i1
      scf.yield %both : i1
    } else {
      %false_bit = arith.constant false
      scf.yield %false_bit : i1
    }
    // Unconditional copy: releasing the intermediate on only one arm is
    // outside the manifest ownership verifier's model.
    %stripped_index = arith.subi %n, %c2 : index
    %kept_index = arith.select %strip, %stripped_index, %n : index
    %kept = arith.index_cast %kept_index : index to i64
    %start = arith.constant 0 : index
    %h, %bytes = func.call @__ly_unicode_from_valid_utf8(%repr_bytes, %start, %kept) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    func.call @LyUnicode_DecRef(%repr_header) : (memref<2xi64>) -> ()
    func.return %h, %bytes : memref<2xi64>, memref<?xi8>
  }

  memref.global "private" constant @__ly_complex_lparen : memref<1xi8> = dense<40>
  memref.global "private" constant @__ly_complex_rparen_j : memref<2xi8> = dense<[106, 41]>
  memref.global "private" constant @__ly_complex_plus : memref<1xi8> = dense<43>
  memref.global "private" constant @__ly_complex_j : memref<1xi8> = dense<106>

  // CPython complex repr: `Xj` when the real part is exactly +0.0, otherwise
  // `(R+Ij)` / `(R-Ij)` (the sign comes from the imaginary component's own
  // repr when negative or NaN).
  func.func @LyComplex_Repr(%header: memref<4xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.complex", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"} {
    %re_slot = arith.constant 2 : index
    %im_slot = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %real_bits = memref.load %header[%re_slot] : memref<4xi64>
    %real = arith.bitcast %real_bits : i64 to f64
    %imag_bits = memref.load %header[%im_slot] : memref<4xi64>
    %imag = arith.bitcast %imag_bits : i64 to f64
    %zero = arith.constant 0.0 : f64
    %real_is_zero = arith.cmpf oeq, %real, %zero : f64
    // +0.0 vs -0.0: the bit pattern distinguishes them where cmpf cannot. The
    // handle form already loaded the bits, so no round trip back through f64.
    %zero_bits = arith.constant 0 : i64
    %real_is_positive_zero_bits = arith.cmpi eq, %real_bits, %zero_bits : i64
    %bare = arith.andi %real_is_zero, %real_is_positive_zero_bits : i1
    %imag_header, %imag_bytes = func.call @__ly_complex_component_repr(%imag) : (f64) -> (memref<2xi64>, memref<?xi8>)
    %result:2 = scf.if %bare -> (memref<2xi64>, memref<?xi8>) {
      // Xj
      %j_static = memref.get_global @__ly_complex_j : memref<1xi8>
      %j = memref.cast %j_static : memref<1xi8> to memref<?xi8>
      %j_len = arith.constant 1 : i64
      %j_start = arith.constant 0 : index
      %j_header, %j_bytes = func.call @__ly_unicode_from_valid_utf8(%j, %j_start, %j_len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
      %h, %bytes = func.call @LyUnicode_Concat(%imag_header, %imag_bytes, %j_header, %j_bytes) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      func.call @LyUnicode_DecRef(%j_header) : (memref<2xi64>) -> ()
      func.call @LyUnicode_DecRef(%imag_header) : (memref<2xi64>) -> ()
      scf.yield %h, %bytes : memref<2xi64>, memref<?xi8>
    } else {
      // (R[+]Ij) -- the '+' joins only when the imag repr does not begin
      // with a sign of its own.
      %real_header, %real_bytes = func.call @__ly_complex_component_repr(%real) : (f64) -> (memref<2xi64>, memref<?xi8>)
      %lparen_static = memref.get_global @__ly_complex_lparen : memref<1xi8>
      %lparen = memref.cast %lparen_static : memref<1xi8> to memref<?xi8>
      %one_i64 = arith.constant 1 : i64
      %start = arith.constant 0 : index
      %lp_header, %lp_bytes = func.call @__ly_unicode_from_valid_utf8(%lparen, %start, %one_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
      %open_header, %open_bytes = func.call @LyUnicode_Concat(%lp_header, %lp_bytes, %real_header, %real_bytes) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      func.call @LyUnicode_DecRef(%lp_header) : (memref<2xi64>) -> ()
      func.call @LyUnicode_DecRef(%real_header) : (memref<2xi64>) -> ()
      %first = memref.load %imag_bytes[%c0] : memref<?xi8>
      %minus = arith.constant 45 : i8
      %imag_signed = arith.cmpi eq, %first, %minus : i8
      %joined:2 = scf.if %imag_signed -> (memref<2xi64>, memref<?xi8>) {
        %h, %bytes = func.call @LyUnicode_Concat(%open_header, %open_bytes, %imag_header, %imag_bytes) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
        func.call @LyUnicode_DecRef(%open_header) : (memref<2xi64>) -> ()
        scf.yield %h, %bytes : memref<2xi64>, memref<?xi8>
      } else {
        %plus_static = memref.get_global @__ly_complex_plus : memref<1xi8>
        %plus = memref.cast %plus_static : memref<1xi8> to memref<?xi8>
        %plus_header, %plus_bytes = func.call @__ly_unicode_from_valid_utf8(%plus, %start, %one_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
        %with_plus_header, %with_plus_bytes = func.call @LyUnicode_Concat(%open_header, %open_bytes, %plus_header, %plus_bytes) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
        func.call @LyUnicode_DecRef(%plus_header) : (memref<2xi64>) -> ()
        func.call @LyUnicode_DecRef(%open_header) : (memref<2xi64>) -> ()
        %h, %bytes = func.call @LyUnicode_Concat(%with_plus_header, %with_plus_bytes, %imag_header, %imag_bytes) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
        func.call @LyUnicode_DecRef(%with_plus_header) : (memref<2xi64>) -> ()
        scf.yield %h, %bytes : memref<2xi64>, memref<?xi8>
      }
      func.call @LyUnicode_DecRef(%imag_header) : (memref<2xi64>) -> ()
      %rp_static = memref.get_global @__ly_complex_rparen_j : memref<2xi8>
      %rp = memref.cast %rp_static : memref<2xi8> to memref<?xi8>
      %two_i64 = arith.constant 2 : i64
      %rp_header, %rp_bytes = func.call @__ly_unicode_from_valid_utf8(%rp, %start, %two_i64) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
      %h, %bytes = func.call @LyUnicode_Concat(%joined#0, %joined#1, %rp_header, %rp_bytes) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
      func.call @LyUnicode_DecRef(%rp_header) : (memref<2xi64>) -> ()
      func.call @LyUnicode_DecRef(%joined#0) : (memref<2xi64>) -> ()
      scf.yield %h, %bytes : memref<2xi64>, memref<?xi8>
    }
    func.return %result#0, %result#1 : memref<2xi64>, memref<?xi8>
  }

  func.func @LyComplex_Str(%header: memref<4xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.complex", ly.runtime.method = "__str__", ly.runtime.result_contract = "builtins.str"} {
    %h, %bytes = func.call @LyComplex_Repr(%header) : (memref<4xi64>) -> (memref<2xi64>, memref<?xi8>)
    func.return %h, %bytes : memref<2xi64>, memref<?xi8>
  }

  // ⭐ complex.__hash__ IS CPython'S: hash(re) + 1000003 * hash(im), wrapping in
  // 64 bits. With no __hash__ of its own a complex fell back to the boxed
  // default, and `{complex(1, 2): "a"}[complex(1, 2)]` raised KeyError -- two
  // values the class calls EQUAL landing in different buckets, which is the
  // failure the unhashable-class refusal exists to prevent and this type walked
  // straight into. The rule also makes hash(complex(2, 0)) == hash(2), which is
  // CPython's numeric-hash invariant.
  func.func @LyComplex_Hash(%header: memref<4xi64> {ly.ownership.object_header}) -> i64 attributes {ly.runtime.contract = "builtins.complex", ly.runtime.method = "__hash__"} {
    %re_slot = arith.constant 2 : index
    %im_slot = arith.constant 3 : index
    %re_bits = memref.load %header[%re_slot] : memref<4xi64>
    %im_bits = memref.load %header[%im_slot] : memref<4xi64>
    %ptr_index = memref.extract_aligned_pointer_as_index %header : memref<4xi64> -> index
    %p = arith.index_cast %ptr_index : index to i64
    %c4 = arith.constant 4 : i64
    %c60 = arith.constant 60 : i64
    %lo = arith.shrui %p, %c4 : i64
    %hi = arith.shli %p, %c60 : i64
    %ident = arith.ori %lo, %hi : i64
    %re_hash = func.call @__ly_float_hash_bits(%re_bits, %ident) : (i64, i64) -> i64
    %im_hash = func.call @__ly_float_hash_bits(%im_bits, %ident) : (i64, i64) -> i64
    %imag_prime = arith.constant 1000003 : i64
    %scaled = arith.muli %im_hash, %imag_prime : i64
    %combined = arith.addi %re_hash, %scaled : i64
    %fixed = func.call @__ly_hash_fixup(%combined) : (i64) -> i64
    func.return %fixed : i64
  }
}
