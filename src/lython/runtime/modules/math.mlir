// Contract manifest AND runtime implementation for the statically supported
// `math` surface. This file demonstrates the standard-module layout: typing
// contracts (module attributes) and the module's runtime functions live
// together under runtime/modules (CPython's Modules/ counterpart).
//
// Kernels use the MLIR math dialect (lowered via ConvertMathToLLVM).

module attributes {
  ly.typing.module = "math",
  ly.typing.callable_exports = [
    "math.floor",
    "math.ceil",
    "math.sqrt",
    "math.fabs",
    "math.trunc",
    "math.log",
    "math.cos",
    "math.sin",
    "math.exp",
    "math.log2",
    "math.log10",
    "math.exp2",
    "math.atan2",
    "math.fmod",
    "math.copysign",
    "math.degrees",
    "math.radians",
    "math.isclose",
    "math.isnan",
    "math.isinf",
    "math.isfinite",
    "math.pow",
    "math.gcd",
    "math.lcm",
    "math.isqrt",
    "math.factorial",
    "math.comb",
    "math.perm"
  ],
  ly.typing.function_names = [
    "math.floor",
    "math.ceil",
    "math.sqrt",
    "math.fabs",
    "math.trunc",
    "math.log",
    "math.cos",
    "math.sin",
    "math.exp",
    "math.log2",
    "math.log10",
    "math.exp2",
    "math.atan2",
    "math.fmod",
    "math.copysign",
    "math.degrees",
    "math.radians",
    "math.isclose",
    "math.isnan",
    "math.isinf",
    "math.isfinite",
    "math.pow",
    "math.gcd",
    "math.lcm",
    "math.isqrt",
    "math.factorial",
    "math.comb",
    "math.perm"
  ],
  ly.typing.float_constant_names = ["math.pi", "math.e", "math.tau", "math.inf", "math.nan"],
  ly.typing.float_constant_values = [3.141592653589793 : f64, 2.718281828459045 : f64, 6.283185307179586 : f64, 0x7FF0000000000000 : f64, 0x7FF8000000000000 : f64],
  ly.typing.function_contracts = [
    !py.callable<[!py.contract<"builtins.float">], arg_names = ["x"], arg_defaults = [false], returns = [!py.contract<"builtins.int">]>,
    !py.callable<[!py.contract<"builtins.float">], arg_names = ["x"], arg_defaults = [false], returns = [!py.contract<"builtins.int">]>,
    !py.callable<[!py.contract<"builtins.float">], arg_names = ["x"], arg_defaults = [false], returns = [!py.contract<"builtins.float">]>,
    !py.callable<[!py.contract<"builtins.float">], arg_names = ["x"], arg_defaults = [false], returns = [!py.contract<"builtins.float">]>,
    !py.callable<[!py.contract<"builtins.float">], arg_names = ["x"], arg_defaults = [false], returns = [!py.contract<"builtins.int">]>,
    !py.callable<[!py.contract<"builtins.float">], arg_names = ["x"], arg_defaults = [false], returns = [!py.contract<"builtins.float">]>,
    !py.callable<[!py.contract<"builtins.float">], arg_names = ["x"], arg_defaults = [false], returns = [!py.contract<"builtins.float">]>,
    !py.callable<[!py.contract<"builtins.float">], arg_names = ["x"], arg_defaults = [false], returns = [!py.contract<"builtins.float">]>,
    !py.callable<[!py.contract<"builtins.float">], arg_names = ["x"], arg_defaults = [false], returns = [!py.contract<"builtins.float">]>,
    !py.callable<[!py.contract<"builtins.float">], arg_names = ["x"], arg_defaults = [false], returns = [!py.contract<"builtins.float">]>,
    !py.callable<[!py.contract<"builtins.float">], arg_names = ["x"], arg_defaults = [false], returns = [!py.contract<"builtins.float">]>,
    !py.callable<[!py.contract<"builtins.float">], arg_names = ["x"], arg_defaults = [false], returns = [!py.contract<"builtins.float">]>,
    !py.callable<[!py.contract<"builtins.float">, !py.contract<"builtins.float">], arg_names = ["y", "x"], arg_defaults = [false, false], returns = [!py.contract<"builtins.float">]>,
    !py.callable<[!py.contract<"builtins.float">, !py.contract<"builtins.float">], arg_names = ["x", "y"], arg_defaults = [false, false], returns = [!py.contract<"builtins.float">]>,
    !py.callable<[!py.contract<"builtins.float">, !py.contract<"builtins.float">], arg_names = ["x", "y"], arg_defaults = [false, false], returns = [!py.contract<"builtins.float">]>,
    !py.callable<[!py.contract<"builtins.float">], arg_names = ["x"], arg_defaults = [false], returns = [!py.contract<"builtins.float">]>,
    !py.callable<[!py.contract<"builtins.float">], arg_names = ["x"], arg_defaults = [false], returns = [!py.contract<"builtins.float">]>,
    !py.callable<[!py.contract<"builtins.float">, !py.contract<"builtins.float">], arg_names = ["a", "b"], arg_defaults = [false, false], kwonly = [!py.contract<"builtins.float">, !py.contract<"builtins.float">], kw_names = ["rel_tol", "abs_tol"], kw_defaults = [true, true], returns = [!py.contract<"builtins.bool">]>,
    !py.callable<[!py.contract<"builtins.float">], arg_names = ["x"], arg_defaults = [false], returns = [!py.contract<"builtins.bool">]>,
    !py.callable<[!py.contract<"builtins.float">], arg_names = ["x"], arg_defaults = [false], returns = [!py.contract<"builtins.bool">]>,
    !py.callable<[!py.contract<"builtins.float">], arg_names = ["x"], arg_defaults = [false], returns = [!py.contract<"builtins.bool">]>,
    !py.callable<[!py.contract<"builtins.float">, !py.contract<"builtins.float">], arg_names = ["x", "y"], arg_defaults = [false, false], returns = [!py.contract<"builtins.float">]>,
    !py.callable<[!py.contract<"builtins.int">, !py.contract<"builtins.int">], arg_names = ["a", "b"], arg_defaults = [false, false], returns = [!py.contract<"builtins.int">]>,
    !py.callable<[!py.contract<"builtins.int">, !py.contract<"builtins.int">], arg_names = ["a", "b"], arg_defaults = [false, false], returns = [!py.contract<"builtins.int">]>,
    !py.callable<[!py.contract<"builtins.int">], arg_names = ["n"], arg_defaults = [false], returns = [!py.contract<"builtins.int">]>,
    !py.callable<[!py.contract<"builtins.int">], arg_names = ["n"], arg_defaults = [false], returns = [!py.contract<"builtins.int">]>,
    !py.callable<[!py.contract<"builtins.int">, !py.contract<"builtins.int">], arg_names = ["n", "k"], arg_defaults = [false, false], returns = [!py.contract<"builtins.int">]>,
    !py.callable<[!py.contract<"builtins.int">, !py.contract<"builtins.int">], arg_names = ["n", "k"], arg_defaults = [false, false], returns = [!py.contract<"builtins.int">]>
  ]
} {
  func.func private @LyFloat_AsF64(%header: memref<3xi64> {ly.ownership.object_header}) -> f64 attributes {ly.runtime.contract = "builtins.float", ly.runtime.method = "__float__", ly.runtime.primitive = "unbox.f64"}
  func.func private @LyFloat_FromF64(%value: f64 {ly.runtime.default_f64 = 0.0 : f64}) -> memref<3xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 2 : i64, ly.runtime.contract = "builtins.float", ly.runtime.initializer = "__new__"}
  func.func private @LyLong_AsI64(%header: memref<2xi64> {ly.ownership.object_header}) -> i64 attributes {ly.runtime.contract = "builtins.int", ly.runtime.primitive = "unbox.i64"}
  func.func private @LyLong_DecRef(%header: memref<2xi64> {ly.ownership.object_header}) attributes {ly.ownership.release_args = [0], ly.runtime.contract = "builtins.int", ly.runtime.deallocator}
  func.func private @LyLong_FromI64(%value: i64 {ly.runtime.default_i64 = 0 : i64}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 1 : i64, ly.runtime.contract = "builtins.int", ly.runtime.initializer = "__new__"}

  // math domain errors. CPython's math_1 checks the operand (and errno) and
  // raises ValueError("math domain error"); returning the IEEE nan/-inf the
  // hardware produces is the one thing it does not do.
  func.func private @LyFloat_Repr(%header: memref<3xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.float", ly.runtime.method = "__repr__", ly.runtime.result_contract = "builtins.str"}
  func.func private @LyUnicode_Concat(%lhs_header: memref<2xi64> {ly.ownership.object_header}, %lhs_bytes: memref<?xi8>, %rhs_header: memref<2xi64> {ly.ownership.object_header}, %rhs_bytes: memref<?xi8>) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.method = "__add__"}
  func.func private @LyUnicode_FromBytes(%bytes: memref<?xi8>, %offset: index, %length: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.initializer = "from_bytes"}
  func.func private @LyValueError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 53 : i64, ly.runtime.contract = "builtins.ValueError", ly.runtime.initializer = "__new__"}
  func.func private @LyValueError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.ValueError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"}
  func.func private @LyValueError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.ValueError", ly.runtime.primitive = "raise"}

  // CPython 3.14 replaced the one generic "math domain error" with a
  // per-function message that names the constraint and interpolates the
  // operand (gh-101410): sqrt says "expected a nonnegative input, got -1.0"
  // and log says "expected a positive input, got 0.0". The message is a
  // documented behaviour of these functions, so it is built the same way --
  // prefix, then the operand's repr -- rather than approximated.
  memref.global "private" constant @__ly_math_nonnegative_msg : memref<34xi8> = dense<[101, 120, 112, 101, 99, 116, 101, 100, 32, 97, 32, 110, 111, 110, 110, 101, 103, 97, 116, 105, 118, 101, 32, 105, 110, 112, 117, 116, 44, 32, 103, 111, 116, 32]>
  memref.global "private" constant @__ly_math_positive_msg : memref<31xi8> = dense<[101, 120, 112, 101, 99, 116, 101, 100, 32, 97, 32, 112, 111, 115, 105, 116, 105, 118, 101, 32, 105, 110, 112, 117, 116, 44, 32, 103, 111, 116, 32]>

  func.func private @__ly_math_domain_raise(%prefix_bytes: memref<?xi8>, %prefix_len: i64, %value: f64) {
    %c0 = arith.constant 0 : index
    %class_id = arith.constant 53 : i64
    %ph, %pb = func.call @LyUnicode_FromBytes(%prefix_bytes, %c0, %prefix_len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %value_header = func.call @LyFloat_FromF64(%value) : (f64) -> memref<3xi64>
    %vh, %vb = func.call @LyFloat_Repr(%value_header) : (memref<3xi64>) -> (memref<2xi64>, memref<?xi8>)
    %mh, %mb = func.call @LyUnicode_Concat(%ph, %pb, %vh, %vb) : (memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<2xi64>, memref<?xi8>)
    %exc:3 = func.call @LyValueError_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    %init:3 = func.call @LyValueError_Init(%exc#0, %exc#1, %exc#2, %mh, %mb) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.call @LyValueError_Raise(%init#0, %init#1, %init#2) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @__ly_math_nonnegative_error(%value: f64) {
    %c34 = arith.constant 34 : i64
    %msg_ref = memref.get_global @__ly_math_nonnegative_msg : memref<34xi8>
    %msg_dyn = memref.cast %msg_ref : memref<34xi8> to memref<?xi8>
    func.call @__ly_math_domain_raise(%msg_dyn, %c34, %value) : (memref<?xi8>, i64, f64) -> ()
    func.return
  }

  func.func private @__ly_math_positive_error(%value: f64) {
    %c31 = arith.constant 31 : i64
    %msg_ref = memref.get_global @__ly_math_positive_msg : memref<31xi8>
    %msg_dyn = memref.cast %msg_ref : memref<31xi8> to memref<?xi8>
    func.call @__ly_math_domain_raise(%msg_dyn, %c31, %value) : (memref<?xi8>, i64, f64) -> ()
    func.return
  }

  func.func @LyMath_Floor(%header: memref<3xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "math.floor", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.float", ly.runtime.primitive = "math_floor", ly.runtime.result_contract = "builtins.int"} {
    %value = func.call @LyFloat_AsF64(%header) : (memref<3xi64>) -> f64
    %floored = math.floor %value : f64
    %as_int = arith.fptosi %floored : f64 to i64
    %int_header = func.call @LyLong_FromI64(%as_int) : (i64) -> memref<2xi64>
    func.return %int_header : memref<2xi64>
  }

  func.func @LyMath_Ceil(%header: memref<3xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "math.ceil", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.float", ly.runtime.primitive = "math_ceil", ly.runtime.result_contract = "builtins.int"} {
    %value = func.call @LyFloat_AsF64(%header) : (memref<3xi64>) -> f64
    %ceiled = math.ceil %value : f64
    %as_int = arith.fptosi %ceiled : f64 to i64
    %int_header = func.call @LyLong_FromI64(%as_int) : (i64) -> memref<2xi64>
    func.return %int_header : memref<2xi64>
  }

  func.func @LyMath_Trunc(%header: memref<3xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "math.trunc", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.float", ly.runtime.primitive = "math_trunc", ly.runtime.result_contract = "builtins.int"} {
    %value = func.call @LyFloat_AsF64(%header) : (memref<3xi64>) -> f64
    %truncated = math.trunc %value : f64
    %as_int = arith.fptosi %truncated : f64 to i64
    %int_header = func.call @LyLong_FromI64(%as_int) : (i64) -> memref<2xi64>
    func.return %int_header : memref<2xi64>
  }

  // ⭐ THE PREDICATES AND THE INTEGER SURFACE. Every math function here was a
  // float kernel; `isnan`, `gcd`, `factorial` and their neighbours were simply
  // absent -- "module 'math' has no attribute 'gcd' in this runtime" for a
  // spelling every numeric program uses.
  memref.global "private" constant @__ly_math_factorial_msg : memref<43xi8> = dense<[102, 97, 99, 116, 111, 114, 105, 97, 108, 40, 41, 32, 110, 111, 116, 32, 100, 101, 102, 105, 110, 101, 100, 32, 102, 111, 114, 32, 110, 101, 103, 97, 116, 105, 118, 101, 32, 118, 97, 108, 117, 101, 115]>
  func.func private @__ly_math_raise_factorial_domain() {
    %c0 = arith.constant 0 : index
    %len = arith.constant 43 : i64
    %class_id = arith.constant 53 : i64
    %msg_ref = memref.get_global @__ly_math_factorial_msg : memref<43xi8>
    %msg_dyn = memref.cast %msg_ref : memref<43xi8> to memref<?xi8>
    %mh, %mb = func.call @LyUnicode_FromBytes(%msg_dyn, %c0, %len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %exc:3 = func.call @LyValueError_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    %init:3 = func.call @LyValueError_Init(%exc#0, %exc#1, %exc#2, %mh, %mb) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.call @LyValueError_Raise(%init#0, %init#1, %init#2) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  memref.global "private" constant @__ly_math_isqrt_msg : memref<36xi8> = dense<[105, 115, 113, 114, 116, 40, 41, 32, 97, 114, 103, 117, 109, 101, 110, 116, 32, 109, 117, 115, 116, 32, 98, 101, 32, 110, 111, 110, 110, 101, 103, 97, 116, 105, 118, 101]>
  func.func private @__ly_math_raise_isqrt_domain() {
    %c0 = arith.constant 0 : index
    %len = arith.constant 36 : i64
    %class_id = arith.constant 53 : i64
    %msg_ref = memref.get_global @__ly_math_isqrt_msg : memref<36xi8>
    %msg_dyn = memref.cast %msg_ref : memref<36xi8> to memref<?xi8>
    %mh, %mb = func.call @LyUnicode_FromBytes(%msg_dyn, %c0, %len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %exc:3 = func.call @LyValueError_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    %init:3 = func.call @LyValueError_Init(%exc#0, %exc#1, %exc#2, %mh, %mb) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.call @LyValueError_Raise(%init#0, %init#1, %init#2) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  memref.global "private" constant @__ly_math_n_msg : memref<32xi8> = dense<[110, 32, 109, 117, 115, 116, 32, 98, 101, 32, 97, 32, 110, 111, 110, 45, 110, 101, 103, 97, 116, 105, 118, 101, 32, 105, 110, 116, 101, 103, 101, 114]>
  func.func private @__ly_math_raise_n_domain() {
    %c0 = arith.constant 0 : index
    %len = arith.constant 32 : i64
    %class_id = arith.constant 53 : i64
    %msg_ref = memref.get_global @__ly_math_n_msg : memref<32xi8>
    %msg_dyn = memref.cast %msg_ref : memref<32xi8> to memref<?xi8>
    %mh, %mb = func.call @LyUnicode_FromBytes(%msg_dyn, %c0, %len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %exc:3 = func.call @LyValueError_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    %init:3 = func.call @LyValueError_Init(%exc#0, %exc#1, %exc#2, %mh, %mb) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.call @LyValueError_Raise(%init#0, %init#1, %init#2) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  memref.global "private" constant @__ly_math_k_msg : memref<32xi8> = dense<[107, 32, 109, 117, 115, 116, 32, 98, 101, 32, 97, 32, 110, 111, 110, 45, 110, 101, 103, 97, 116, 105, 118, 101, 32, 105, 110, 116, 101, 103, 101, 114]>
  // ⛔ ONE GUARD AND ONE RAISE SITE. Two sequential guarded raises inside one
  // manifest function let the exception escape the caller's handler -- a
  // `try` around `math.comb(-1, 2)` caught it at module scope and did not
  // catch it inside a `for` loop. Every other raising kernel here has exactly
  // one, so the message is chosen inside the helper instead of by the CFG.
  func.func private @__ly_math_raise_nk_domain(%blame_k: i1) {
    %c0 = arith.constant 0 : index
    %len = arith.constant 32 : i64
    %class_id = arith.constant 53 : i64
    %n_ref = memref.get_global @__ly_math_n_msg : memref<32xi8>
    %k_ref = memref.get_global @__ly_math_k_msg : memref<32xi8>
    %n_dyn = memref.cast %n_ref : memref<32xi8> to memref<?xi8>
    %k_dyn = memref.cast %k_ref : memref<32xi8> to memref<?xi8>
    %msg_dyn = arith.select %blame_k, %k_dyn, %n_dyn : memref<?xi8>
    %mh, %mb = func.call @LyUnicode_FromBytes(%msg_dyn, %c0, %len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %exc:3 = func.call @LyValueError_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    %init:3 = func.call @LyValueError_Init(%exc#0, %exc#1, %exc#2, %mh, %mb) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.call @LyValueError_Raise(%init#0, %init#1, %init#2) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func private @__ly_math_raise_k_domain() {
    %c0 = arith.constant 0 : index
    %len = arith.constant 32 : i64
    %class_id = arith.constant 53 : i64
    %msg_ref = memref.get_global @__ly_math_k_msg : memref<32xi8>
    %msg_dyn = memref.cast %msg_ref : memref<32xi8> to memref<?xi8>
    %mh, %mb = func.call @LyUnicode_FromBytes(%msg_dyn, %c0, %len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %exc:3 = func.call @LyValueError_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    %init:3 = func.call @LyValueError_Init(%exc#0, %exc#1, %exc#2, %mh, %mb) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.call @LyValueError_Raise(%init#0, %init#1, %init#2) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyMath_IsNan(%header: memref<3xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.builtin = "math.isnan", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.float", ly.runtime.primitive = "math_isnan", ly.runtime.result_contract = "builtins.bool"} {
    %value = func.call @LyFloat_AsF64(%header) : (memref<3xi64>) -> f64
    // NaN is the only value that is not equal to itself.
    %equal = arith.cmpf oeq, %value, %value : f64
    %true = arith.constant true
    %is_nan = arith.xori %equal, %true : i1
    func.return %is_nan : i1
  }

  func.func @LyMath_IsInf(%header: memref<3xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.builtin = "math.isinf", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.float", ly.runtime.primitive = "math_isinf", ly.runtime.result_contract = "builtins.bool"} {
    %value = func.call @LyFloat_AsF64(%header) : (memref<3xi64>) -> f64
    %magnitude = math.absf %value : f64
    %infinity = arith.constant 0x7FF0000000000000 : f64
    %is_inf = arith.cmpf oeq, %magnitude, %infinity : f64
    func.return %is_inf : i1
  }

  func.func @LyMath_IsFinite(%header: memref<3xi64> {ly.ownership.object_header}) -> i1 attributes {ly.runtime.builtin = "math.isfinite", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.float", ly.runtime.primitive = "math_isfinite", ly.runtime.result_contract = "builtins.bool"} {
    %value = func.call @LyFloat_AsF64(%header) : (memref<3xi64>) -> f64
    %magnitude = math.absf %value : f64
    %infinity = arith.constant 0x7FF0000000000000 : f64
    %below = arith.cmpf olt, %magnitude, %infinity : f64
    func.return %below : i1
  }

  // `math.pow` always answers a float, where the `**` operator keeps int
  // exactness -- that is CPython's split too.
  func.func @LyMath_Pow(%x_header: memref<3xi64> {ly.ownership.object_header}, %y_header: memref<3xi64> {ly.ownership.object_header}) -> memref<3xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "math.pow", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.float", ly.runtime.primitive = "math_pow", ly.runtime.result_contract = "builtins.float"} {
    %x = func.call @LyFloat_AsF64(%x_header) : (memref<3xi64>) -> f64
    %y = func.call @LyFloat_AsF64(%y_header) : (memref<3xi64>) -> f64
    %result = math.powf %x, %y : f64
    %header = func.call @LyFloat_FromF64(%result) : (f64) -> memref<3xi64>
    func.return %header : memref<3xi64>
  }

  // ⛔ THE INTEGER ONES TAKE THE i64 WINDOW. `gcd`, `lcm`, `isqrt`, `factorial`,
  // `comb` and `perm` are declared over `builtins.int`, whose values are
  // arbitrary precision -- these read the machine window and raise
  // OverflowError past it, which is what `unbox.i64` does for every other
  // machine-word kernel in this tree. Matching CPython's unbounded answers
  // needs the LyLong kernels, which is a separate build.
  func.func @LyMath_Gcd(%a_header: memref<2xi64> {ly.ownership.object_header}, %b_header: memref<2xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "math.gcd", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.int", ly.runtime.primitive = "math_gcd", ly.runtime.result_contract = "builtins.int"} {
    %a_raw = func.call @LyLong_AsI64(%a_header) : (memref<2xi64>) -> i64
    %b_raw = func.call @LyLong_AsI64(%b_header) : (memref<2xi64>) -> i64
    %zero = arith.constant 0 : i64
    %a_neg = arith.subi %zero, %a_raw : i64
    %b_neg = arith.subi %zero, %b_raw : i64
    %a_lt = arith.cmpi slt, %a_raw, %zero : i64
    %b_lt = arith.cmpi slt, %b_raw, %zero : i64
    %a0 = arith.select %a_lt, %a_neg, %a_raw : i64
    %b0 = arith.select %b_lt, %b_neg, %b_raw : i64
    %result:2 = scf.while (%x = %a0, %y = %b0) : (i64, i64) -> (i64, i64) {
      %continue = arith.cmpi ne, %y, %zero : i64
      scf.condition(%continue) %x, %y : i64, i64
    } do {
    ^body(%x: i64, %y: i64):
      %remainder = arith.remsi %x, %y : i64
      scf.yield %y, %remainder : i64, i64
    }
    %header = func.call @LyLong_FromI64(%result#0) : (i64) -> memref<2xi64>
    func.return %header : memref<2xi64>
  }

  func.func @LyMath_Lcm(%a_header: memref<2xi64> {ly.ownership.object_header}, %b_header: memref<2xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "math.lcm", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.int", ly.runtime.primitive = "math_lcm", ly.runtime.result_contract = "builtins.int"} {
    %a_raw = func.call @LyLong_AsI64(%a_header) : (memref<2xi64>) -> i64
    %b_raw = func.call @LyLong_AsI64(%b_header) : (memref<2xi64>) -> i64
    %zero = arith.constant 0 : i64
    %a_zero = arith.cmpi eq, %a_raw, %zero : i64
    %b_zero = arith.cmpi eq, %b_raw, %zero : i64
    %either_zero = arith.ori %a_zero, %b_zero : i1
    %divisor = func.call @LyMath_Gcd(%a_header, %b_header) : (memref<2xi64>, memref<2xi64>) -> memref<2xi64>
    %divisor_raw = func.call @LyLong_AsI64(%divisor) : (memref<2xi64>) -> i64
    func.call @LyLong_DecRef(%divisor) : (memref<2xi64>) -> ()
    %one = arith.constant 1 : i64
    %safe_divisor = arith.select %either_zero, %one, %divisor_raw : i64
    %product = arith.muli %a_raw, %b_raw : i64
    %quotient = arith.divsi %product, %safe_divisor : i64
    %magnitude_neg = arith.subi %zero, %quotient : i64
    %negative = arith.cmpi slt, %quotient, %zero : i64
    %magnitude = arith.select %negative, %magnitude_neg, %quotient : i64
    %result = arith.select %either_zero, %zero, %magnitude : i64
    %header = func.call @LyLong_FromI64(%result) : (i64) -> memref<2xi64>
    func.return %header : memref<2xi64>
  }

  func.func @LyMath_Isqrt(%header_in: memref<2xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "math.isqrt", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.int", ly.runtime.primitive = "math_isqrt", ly.runtime.result_contract = "builtins.int"} {
    %value = func.call @LyLong_AsI64(%header_in) : (memref<2xi64>) -> i64
    %zero = arith.constant 0 : i64
    %negative = arith.cmpi slt, %value, %zero : i64
    cf.cond_br %negative, ^domain, ^ok

  ^domain:
    func.call @__ly_math_raise_isqrt_domain() : () -> ()
    cf.br ^ok

  ^ok:
    // CPython's own loop: x = n; y = (x + 1) // 2; while y < x: x = y;
    // y = (x + n // x) // 2. `x` is floor(sqrt(n)) when it stops.
    %one = arith.constant 1 : i64
    %two = arith.constant 2 : i64
    %positive = arith.cmpi sgt, %value, %zero : i64
    %seed_next_sum = arith.addi %value, %one : i64
    %seed_next = arith.divsi %seed_next_sum, %two : i64
    %converged:2 = scf.while (%x = %value, %y = %seed_next) : (i64, i64) -> (i64, i64) {
      %smaller = arith.cmpi slt, %y, %x : i64
      scf.condition(%smaller) %x, %y : i64, i64
    } do {
    ^body(%x: i64, %y: i64):
      %quotient = arith.divsi %value, %y : i64
      %sum = arith.addi %y, %quotient : i64
      %next = arith.divsi %sum, %two : i64
      scf.yield %y, %next : i64, i64
    }
    %answer = arith.select %positive, %converged#0, %zero : i64
    %result_header = func.call @LyLong_FromI64(%answer) : (i64) -> memref<2xi64>
    func.return %result_header : memref<2xi64>
  }

  func.func @LyMath_Factorial(%header_in: memref<2xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "math.factorial", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.int", ly.runtime.primitive = "math_factorial", ly.runtime.result_contract = "builtins.int"} {
    %value = func.call @LyLong_AsI64(%header_in) : (memref<2xi64>) -> i64
    %zero = arith.constant 0 : i64
    %negative = arith.cmpi slt, %value, %zero : i64
    cf.cond_br %negative, ^domain, ^ok

  ^domain:
    func.call @__ly_math_raise_factorial_domain() : () -> ()
    cf.br ^ok

  ^ok:
    %one = arith.constant 1 : i64
    %count = arith.index_cast %value : i64 to index
    %lower = arith.constant 0 : index
    %step = arith.constant 1 : index
    %product = scf.for %index = %lower to %count step %step iter_args(%acc = %one) -> (i64) {
      %term_index = arith.addi %index, %step : index
      %term = arith.index_cast %term_index : index to i64
      %next = arith.muli %acc, %term : i64
      scf.yield %next : i64
    }
    %result_header = func.call @LyLong_FromI64(%product) : (i64) -> memref<2xi64>
    func.return %result_header : memref<2xi64>
  }

  func.func private @__ly_math_perm_product(%n: i64, %k: i64) -> i64 {
    %one = arith.constant 1 : i64
    %count = arith.index_cast %k : i64 to index
    %lower = arith.constant 0 : index
    %step = arith.constant 1 : index
    %product = scf.for %index = %lower to %count step %step iter_args(%acc = %one) -> (i64) {
      %offset = arith.index_cast %index : index to i64
      %term = arith.subi %n, %offset : i64
      %next = arith.muli %acc, %term : i64
      scf.yield %next : i64
    }
    func.return %product : i64
  }

  func.func @LyMath_Perm(%n_header: memref<2xi64> {ly.ownership.object_header}, %k_header: memref<2xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "math.perm", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.int", ly.runtime.primitive = "math_perm", ly.runtime.result_contract = "builtins.int"} {
    %n = func.call @LyLong_AsI64(%n_header) : (memref<2xi64>) -> i64
    %k = func.call @LyLong_AsI64(%k_header) : (memref<2xi64>) -> i64
    %zero = arith.constant 0 : i64
    %n_negative = arith.cmpi slt, %n, %zero : i64
    %k_negative = arith.cmpi slt, %k, %zero : i64
    %true = arith.constant true
    %n_negative_not = arith.xori %n_negative, %true : i1
    %negative = arith.ori %n_negative, %k_negative : i1
    cf.cond_br %negative, ^domain, ^ok

  ^domain:
    func.call @__ly_math_raise_nk_domain(%n_negative_not) : (i1) -> ()
    cf.br ^ok

  ^ok:
    %too_many = arith.cmpi sgt, %k, %n : i64
    %product = func.call @__ly_math_perm_product(%n, %k) : (i64, i64) -> i64
    %answer = arith.select %too_many, %zero, %product : i64
    %result_header = func.call @LyLong_FromI64(%answer) : (i64) -> memref<2xi64>
    func.return %result_header : memref<2xi64>
  }

  func.func @LyMath_Comb(%n_header: memref<2xi64> {ly.ownership.object_header}, %k_header: memref<2xi64> {ly.ownership.object_header}) -> memref<2xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "math.comb", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.int", ly.runtime.primitive = "math_comb", ly.runtime.result_contract = "builtins.int"} {
    %n = func.call @LyLong_AsI64(%n_header) : (memref<2xi64>) -> i64
    %k = func.call @LyLong_AsI64(%k_header) : (memref<2xi64>) -> i64
    %zero = arith.constant 0 : i64
    %n_negative = arith.cmpi slt, %n, %zero : i64
    %k_negative = arith.cmpi slt, %k, %zero : i64
    %true = arith.constant true
    %n_negative_not = arith.xori %n_negative, %true : i1
    %negative = arith.ori %n_negative, %k_negative : i1
    cf.cond_br %negative, ^domain, ^ok

  ^domain:
    func.call @__ly_math_raise_nk_domain(%n_negative_not) : (i1) -> ()
    cf.br ^ok

  ^ok:
    %too_many = arith.cmpi sgt, %k, %n : i64
    // The smaller of k and n-k keeps the running product inside the window for
    // the largest range this can answer at all.
    %complement = arith.subi %n, %k : i64
    %use_complement = arith.cmpi slt, %complement, %k : i64
    %effective = arith.select %use_complement, %complement, %k : i64
    %numerator = func.call @__ly_math_perm_product(%n, %effective) : (i64, i64) -> i64
    %one = arith.constant 1 : i64
    %count = arith.index_cast %effective : i64 to index
    %lower = arith.constant 0 : index
    %step = arith.constant 1 : index
    %denominator = scf.for %index = %lower to %count step %step iter_args(%acc = %one) -> (i64) {
      %offset = arith.index_cast %index : index to i64
      %term = arith.addi %offset, %one : i64
      %next = arith.muli %acc, %term : i64
      scf.yield %next : i64
    }
    %quotient = arith.divsi %numerator, %denominator : i64
    %answer = arith.select %too_many, %zero, %quotient : i64
    %result_header = func.call @LyLong_FromI64(%answer) : (i64) -> memref<2xi64>
    func.return %result_header : memref<2xi64>
  }

  func.func @LyMath_Sqrt(%header: memref<3xi64> {ly.ownership.object_header}) -> memref<3xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "math.sqrt", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.float", ly.runtime.primitive = "math_sqrt", ly.runtime.result_contract = "builtins.float"} {
    %value = func.call @LyFloat_AsF64(%header) : (memref<3xi64>) -> f64
    %zero = arith.constant 0.0 : f64
    %negative = arith.cmpf olt, %value, %zero : f64
    cf.cond_br %negative, ^domain, ^ok

  ^domain:
    func.call @__ly_math_nonnegative_error(%value) : (f64) -> ()
    cf.br ^ok

  ^ok:
    %root = math.sqrt %value : f64
    %out_header = func.call @LyFloat_FromF64(%root) : (f64) -> memref<3xi64>
    func.return %out_header : memref<3xi64>
  }

  func.func @LyMath_Fabs(%header: memref<3xi64> {ly.ownership.object_header}) -> memref<3xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "math.fabs", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.float", ly.runtime.primitive = "math_fabs", ly.runtime.result_contract = "builtins.float"} {
    %value = func.call @LyFloat_AsF64(%header) : (memref<3xi64>) -> f64
    %magnitude = math.absf %value : f64
    %out_header = func.call @LyFloat_FromF64(%magnitude) : (f64) -> memref<3xi64>
    func.return %out_header : memref<3xi64>
  }

  // log / cos / sin / exp: the kernels random.gauss's Box-Muller transform
  // needs. Single-argument only -- CPython's math.log takes an optional base,
  // which would need an overloaded contract; write log(x) / log(b) for that.
  func.func @LyMath_Log(%header: memref<3xi64> {ly.ownership.object_header}) -> memref<3xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "math.log", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.float", ly.runtime.primitive = "math_log", ly.runtime.result_contract = "builtins.float"} {
    %value = func.call @LyFloat_AsF64(%header) : (memref<3xi64>) -> f64
    // CPython's math_1 rejects the whole non-positive domain: log(0.0) is a
    // domain error there, not the -inf the hardware returns.
    %zero = arith.constant 0.0 : f64
    %nonpositive = arith.cmpf ole, %value, %zero : f64
    cf.cond_br %nonpositive, ^domain, ^ok

  ^domain:
    func.call @__ly_math_positive_error(%value) : (f64) -> ()
    cf.br ^ok

  ^ok:
    %result = math.log %value : f64
    %out_header = func.call @LyFloat_FromF64(%result) : (f64) -> memref<3xi64>
    func.return %out_header : memref<3xi64>
  }

  func.func @LyMath_Cos(%header: memref<3xi64> {ly.ownership.object_header}) -> memref<3xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "math.cos", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.float", ly.runtime.primitive = "math_cos", ly.runtime.result_contract = "builtins.float"} {
    %value = func.call @LyFloat_AsF64(%header) : (memref<3xi64>) -> f64
    %result = math.cos %value : f64
    %out_header = func.call @LyFloat_FromF64(%result) : (f64) -> memref<3xi64>
    func.return %out_header : memref<3xi64>
  }

  func.func @LyMath_Sin(%header: memref<3xi64> {ly.ownership.object_header}) -> memref<3xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "math.sin", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.float", ly.runtime.primitive = "math_sin", ly.runtime.result_contract = "builtins.float"} {
    %value = func.call @LyFloat_AsF64(%header) : (memref<3xi64>) -> f64
    %result = math.sin %value : f64
    %out_header = func.call @LyFloat_FromF64(%result) : (f64) -> memref<3xi64>
    func.return %out_header : memref<3xi64>
  }

  // CPython's math_1 wrapper checks errno after libm and raises
  // OverflowError("math range error") when the result overflowed; returning
  // inf silently is the one thing it does not do (Modules/mathmodule.c,
  // is_error). Only exp is repaired here because it is the one the audit
  // reached; the same check belongs on any libm call that can set ERANGE.
  memref.global "private" constant @__ly_math_range_msg : memref<16xi8> = dense<[109, 97, 116, 104, 32, 114, 97, 110, 103, 101, 32, 101, 114, 114, 111, 114]>

  func.func private @LyOverflowError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 104 : i64, ly.runtime.contract = "builtins.OverflowError", ly.runtime.initializer = "__new__"}
  func.func private @LyOverflowError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.OverflowError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"}
  func.func private @LyOverflowError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.OverflowError", ly.runtime.primitive = "raise"}

  func.func private @__ly_math_range_error() {
    %c0 = arith.constant 0 : index
    %len = arith.constant 16 : i64
    %class_id = arith.constant 104 : i64
    %msg_ref = memref.get_global @__ly_math_range_msg : memref<16xi8>
    %msg_dyn = memref.cast %msg_ref : memref<16xi8> to memref<?xi8>
    %mh, %mb = func.call @LyUnicode_FromBytes(%msg_dyn, %c0, %len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %exc:3 = func.call @LyOverflowError_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    %init:3 = func.call @LyOverflowError_Init(%exc#0, %exc#1, %exc#2, %mh, %mb) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.call @LyOverflowError_Raise(%init#0, %init#1, %init#2) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyMath_Exp(%header: memref<3xi64> {ly.ownership.object_header}) -> memref<3xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "math.exp", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.float", ly.runtime.primitive = "math_exp", ly.runtime.result_contract = "builtins.float"} {
    %value = func.call @LyFloat_AsF64(%header) : (memref<3xi64>) -> f64
    %result = math.exp %value : f64
    %inf = arith.constant 0x7FF0000000000000 : f64
    %overflowed = arith.cmpf oeq, %result, %inf : f64
    %finite_input = arith.cmpf one, %value, %inf : f64
    %range_error = arith.andi %overflowed, %finite_input : i1
    cf.cond_br %range_error, ^range, ^ok

  ^range:
    func.call @__ly_math_range_error() : () -> ()
    cf.br ^ok

  ^ok:
    %out_header = func.call @LyFloat_FromF64(%result) : (f64) -> memref<3xi64>
    func.return %out_header : memref<3xi64>
  }

  // The libm-backed rungs CPython also gets from libm, so the bits agree:
  // log2 / log10 / exp2 / atan2 / fmod / copysign, plus the two conversions
  // CPython computes as a single multiply (Modules/mathmodule.c m_degrees,
  // m_radians) rather than as a division.
  //
  // ⛔ Domain and range checks are NOT optional decoration: CPython's math_1
  // wrapper reads errno and raises, so `log2(0.0)` is a ValueError naming the
  // constraint and the operand, `fmod(1.0, 0.0)` is the generic "math domain
  // error", and `exp2(10000.0)` is an OverflowError. Returning -inf / nan / inf
  // silently is the one thing none of them do.
  memref.global "private" constant @__ly_math_domain_msg : memref<17xi8> = dense<[109, 97, 116, 104, 32, 100, 111, 109, 97, 105, 110, 32, 101, 114, 114, 111, 114]>

  func.func private @__ly_math_generic_domain_error() {
    %c0 = arith.constant 0 : index
    %len = arith.constant 17 : i64
    %class_id = arith.constant 53 : i64
    %msg_ref = memref.get_global @__ly_math_domain_msg : memref<17xi8>
    %msg_dyn = memref.cast %msg_ref : memref<17xi8> to memref<?xi8>
    %mh, %mb = func.call @LyUnicode_FromBytes(%msg_dyn, %c0, %len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %exc:3 = func.call @LyValueError_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    %init:3 = func.call @LyValueError_Init(%exc#0, %exc#1, %exc#2, %mh, %mb) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.call @LyValueError_Raise(%init#0, %init#1, %init#2) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  func.func @LyMath_Log2(%header: memref<3xi64> {ly.ownership.object_header}) -> memref<3xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "math.log2", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.float", ly.runtime.primitive = "math_log2", ly.runtime.result_contract = "builtins.float"} {
    %value = func.call @LyFloat_AsF64(%header) : (memref<3xi64>) -> f64
    %zero = arith.constant 0.0 : f64
    %nonpositive = arith.cmpf ole, %value, %zero : f64
    cf.cond_br %nonpositive, ^domain, ^ok

  ^domain:
    func.call @__ly_math_positive_error(%value) : (f64) -> ()
    cf.br ^ok

  ^ok:
    %result = math.log2 %value : f64
    %out_header = func.call @LyFloat_FromF64(%result) : (f64) -> memref<3xi64>
    func.return %out_header : memref<3xi64>
  }

  func.func @LyMath_Log10(%header: memref<3xi64> {ly.ownership.object_header}) -> memref<3xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "math.log10", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.float", ly.runtime.primitive = "math_log10", ly.runtime.result_contract = "builtins.float"} {
    %value = func.call @LyFloat_AsF64(%header) : (memref<3xi64>) -> f64
    %zero = arith.constant 0.0 : f64
    %nonpositive = arith.cmpf ole, %value, %zero : f64
    cf.cond_br %nonpositive, ^domain, ^ok

  ^domain:
    func.call @__ly_math_positive_error(%value) : (f64) -> ()
    cf.br ^ok

  ^ok:
    %result = math.log10 %value : f64
    %out_header = func.call @LyFloat_FromF64(%result) : (f64) -> memref<3xi64>
    func.return %out_header : memref<3xi64>
  }

  func.func @LyMath_Exp2(%header: memref<3xi64> {ly.ownership.object_header}) -> memref<3xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "math.exp2", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.float", ly.runtime.primitive = "math_exp2", ly.runtime.result_contract = "builtins.float"} {
    %value = func.call @LyFloat_AsF64(%header) : (memref<3xi64>) -> f64
    %result = math.exp2 %value : f64
    %inf = arith.constant 0x7FF0000000000000 : f64
    %overflowed = arith.cmpf oeq, %result, %inf : f64
    %finite_input = arith.cmpf one, %value, %inf : f64
    %range_error = arith.andi %overflowed, %finite_input : i1
    cf.cond_br %range_error, ^range, ^ok

  ^range:
    func.call @__ly_math_range_error() : () -> ()
    cf.br ^ok

  ^ok:
    %out_header = func.call @LyFloat_FromF64(%result) : (f64) -> memref<3xi64>
    func.return %out_header : memref<3xi64>
  }

  func.func @LyMath_Atan2(%y_header: memref<3xi64> {ly.ownership.object_header}, %x_header: memref<3xi64> {ly.ownership.object_header}) -> memref<3xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "math.atan2", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.float", ly.runtime.primitive = "math_atan2", ly.runtime.result_contract = "builtins.float"} {
    %y = func.call @LyFloat_AsF64(%y_header) : (memref<3xi64>) -> f64
    %x = func.call @LyFloat_AsF64(%x_header) : (memref<3xi64>) -> f64
    // Defined on the whole plane, signed zeros included, which is why there is
    // no check here and why the operand order is (y, x) like CPython's.
    %result = math.atan2 %y, %x : f64
    %out_header = func.call @LyFloat_FromF64(%result) : (f64) -> memref<3xi64>
    func.return %out_header : memref<3xi64>
  }

  func.func @LyMath_Fmod(%x_header: memref<3xi64> {ly.ownership.object_header}, %y_header: memref<3xi64> {ly.ownership.object_header}) -> memref<3xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "math.fmod", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.float", ly.runtime.primitive = "math_fmod", ly.runtime.result_contract = "builtins.float"} {
    %x = func.call @LyFloat_AsF64(%x_header) : (memref<3xi64>) -> f64
    %y = func.call @LyFloat_AsF64(%y_header) : (memref<3xi64>) -> f64
    %result = arith.remf %x, %y : f64
    // ⭐ CPython's own test, not a list of the cases that reach it: a NaN
    // result from operands that were not NaN is EDOM (Modules/mathmodule.c
    // math_fmod), which covers fmod(inf, 2.0) and fmod(1.0, 0.0) together and
    // leaves fmod(nan, 1.0) returning nan the way it does there.
    %true = arith.constant true
    %result_is_nan = arith.cmpf uno, %result, %result : f64
    %x_is_nan = arith.cmpf uno, %x, %x : f64
    %y_is_nan = arith.cmpf uno, %y, %y : f64
    %operand_is_nan = arith.ori %x_is_nan, %y_is_nan : i1
    %operands_are_numbers = arith.xori %operand_is_nan, %true : i1
    %domain = arith.andi %result_is_nan, %operands_are_numbers : i1
    cf.cond_br %domain, ^domain_error, ^ok

  ^domain_error:
    func.call @__ly_math_generic_domain_error() : () -> ()
    cf.br ^ok

  ^ok:
    %out_header = func.call @LyFloat_FromF64(%result) : (f64) -> memref<3xi64>
    func.return %out_header : memref<3xi64>
  }

  func.func @LyMath_Copysign(%x_header: memref<3xi64> {ly.ownership.object_header}, %y_header: memref<3xi64> {ly.ownership.object_header}) -> memref<3xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "math.copysign", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.float", ly.runtime.primitive = "math_copysign", ly.runtime.result_contract = "builtins.float"} {
    %x = func.call @LyFloat_AsF64(%x_header) : (memref<3xi64>) -> f64
    %y = func.call @LyFloat_AsF64(%y_header) : (memref<3xi64>) -> f64
    %result = math.copysign %x, %y : f64
    %out_header = func.call @LyFloat_FromF64(%result) : (f64) -> memref<3xi64>
    func.return %out_header : memref<3xi64>
  }

  func.func @LyMath_Degrees(%header: memref<3xi64> {ly.ownership.object_header}) -> memref<3xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "math.degrees", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.float", ly.runtime.primitive = "math_degrees", ly.runtime.result_contract = "builtins.float"} {
    %value = func.call @LyFloat_AsF64(%header) : (memref<3xi64>) -> f64
    // ⛔ The CONSTANT is 180/pi rounded once, not a division at run time: that
    // is what CPython multiplies by, and dividing instead lands an ulp away on
    // inputs whose product is near a tie.
    %factor = arith.constant 57.29577951308232 : f64
    %result = arith.mulf %value, %factor : f64
    %out_header = func.call @LyFloat_FromF64(%result) : (f64) -> memref<3xi64>
    func.return %out_header : memref<3xi64>
  }


  // "tolerances must be non-negative"
  memref.global "private" constant @__ly_math_msg_negative_tolerance : memref<31xi8> = dense<[116, 111, 108, 101, 114, 97, 110, 99, 101, 115, 32, 109, 117, 115, 116, 32, 98, 101, 32, 110, 111, 110, 45, 110, 101, 103, 97, 116, 105, 118, 101]>

  func.func private @__ly_math_negative_tolerance_error() {
    %c0 = arith.constant 0 : index
    %len = arith.constant 31 : i64
    %class_id = arith.constant 53 : i64
    %msg_ref = memref.get_global @__ly_math_msg_negative_tolerance : memref<31xi8>
    %msg_dyn = memref.cast %msg_ref : memref<31xi8> to memref<?xi8>
    %mh, %mb = func.call @LyUnicode_FromBytes(%msg_dyn, %c0, %len) : (memref<?xi8>, index, i64) -> (memref<2xi64>, memref<?xi8>)
    %exc:3 = func.call @LyValueError_New(%class_id) : (i64) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    %init:3 = func.call @LyValueError_Init(%exc#0, %exc#1, %exc#2, %mh, %mb) : (memref<3xi64>, memref<2xi64>, memref<?xi8>, memref<2xi64>, memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>)
    func.call @LyValueError_Raise(%init#0, %init#1, %init#2) : (memref<3xi64>, memref<2xi64>, memref<?xi8>) -> ()
    func.return
  }

  // ⭐ The RAW f64 parameters are what let the two tolerances have defaults:
  // ly.runtime.default_f64 fills a missing scalar, and a float OBJECT parameter
  // has no such spelling. A float argument reaches an f64 input through float's
  // unbox.f64 in the ABI adapter, which is the same road complex(...) takes.
  //
  // The body is CPython's math_isclose verbatim (Modules/mathmodule.c), and
  // each line of it is load-bearing: `a == b` first so infinities compare equal
  // to themselves, the isinf check so inf vs finite is False rather than
  // inf <= inf, both relative tests because the tolerance is relative to
  // EITHER operand, and no NaN case at all -- every comparison is false for a
  // NaN, which is the answer.
  func.func @LyMath_IsClose(%a: f64, %b: f64, %rel_tol: f64 {ly.runtime.default_f64 = 1.000000e-09 : f64}, %abs_tol: f64 {ly.runtime.default_f64 = 0.0 : f64}) -> i1 attributes {ly.runtime.builtin = "math.isclose", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.float", ly.runtime.primitive = "math_isclose", ly.runtime.result_contract = "builtins.bool"} {
    %zero = arith.constant 0.0 : f64
    %rel_negative = arith.cmpf olt, %rel_tol, %zero : f64
    %abs_negative = arith.cmpf olt, %abs_tol, %zero : f64
    %negative = arith.ori %rel_negative, %abs_negative : i1
    cf.cond_br %negative, ^tolerance_error, ^compare

  ^tolerance_error:
    func.call @__ly_math_negative_tolerance_error() : () -> ()
    cf.br ^compare

  ^compare:
    %equal = arith.cmpf oeq, %a, %b : f64
    cf.cond_br %equal, ^exact, ^finite_check

  ^exact:
    %true_bit = arith.constant true
    func.return %true_bit : i1

  ^finite_check:
    %inf = arith.constant 0x7FF0000000000000 : f64
    %abs_a = math.absf %a : f64
    %abs_b = math.absf %b : f64
    %a_infinite = arith.cmpf oeq, %abs_a, %inf : f64
    %b_infinite = arith.cmpf oeq, %abs_b, %inf : f64
    %either_infinite = arith.ori %a_infinite, %b_infinite : i1
    cf.cond_br %either_infinite, ^apart, ^tolerances

  ^apart:
    %false_bit = arith.constant false
    func.return %false_bit : i1

  ^tolerances:
    %delta = arith.subf %b, %a : f64
    %diff = math.absf %delta : f64
    %scaled_b = arith.mulf %rel_tol, %b : f64
    %scaled_a = arith.mulf %rel_tol, %a : f64
    %bound_b = math.absf %scaled_b : f64
    %bound_a = math.absf %scaled_a : f64
    %within_b = arith.cmpf ole, %diff, %bound_b : f64
    %within_a = arith.cmpf ole, %diff, %bound_a : f64
    %within_abs = arith.cmpf ole, %diff, %abs_tol : f64
    %relative = arith.ori %within_b, %within_a : i1
    %result = arith.ori %relative, %within_abs : i1
    func.return %result : i1
  }

  func.func @LyMath_Radians(%header: memref<3xi64> {ly.ownership.object_header}) -> memref<3xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "math.radians", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.float", ly.runtime.primitive = "math_radians", ly.runtime.result_contract = "builtins.float"} {
    %value = func.call @LyFloat_AsF64(%header) : (memref<3xi64>) -> f64
    %factor = arith.constant 0.017453292519943295 : f64
    %result = arith.mulf %value, %factor : f64
    %out_header = func.call @LyFloat_FromF64(%result) : (f64) -> memref<3xi64>
    func.return %out_header : memref<3xi64>
  }
}
