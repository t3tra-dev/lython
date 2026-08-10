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
    "math.exp"
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
    "math.exp"
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
    !py.callable<[!py.contract<"builtins.float">], arg_names = ["x"], arg_defaults = [false], returns = [!py.contract<"builtins.float">]>
  ]
} {
  func.func private @LyFloat_AsF64(%header: memref<3xi64> {ly.ownership.object_header}) -> f64 attributes {ly.runtime.contract = "builtins.float", ly.runtime.method = "__float__", ly.runtime.primitive = "unbox.f64"}
  func.func private @LyFloat_FromF64(%value: f64 {ly.runtime.default_f64 = 0.0 : f64}) -> memref<3xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 2 : i64, ly.runtime.contract = "builtins.float", ly.runtime.initializer = "__new__"}
  func.func private @LyLong_FromI64(%value: i64 {ly.runtime.default_i64 = 0 : i64}) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 1 : i64, ly.runtime.contract = "builtins.int", ly.runtime.initializer = "__new__"}

  // math domain errors. CPython's math_1 checks the operand (and errno) and
  // raises ValueError("math domain error"); returning the IEEE nan/-inf the
  // hardware produces is the one thing it does not do.
  func.func private @LyUnicode_FromBytes(%bytes: memref<?xi8>, %offset: index, %length: i64) -> (memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.contract = "builtins.str", ly.runtime.initializer = "from_bytes"}
  func.func private @LyValueError_New(%class_id: i64 {ly.runtime.class_id_argument}) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 53 : i64, ly.runtime.contract = "builtins.ValueError", ly.runtime.initializer = "__new__"}
  func.func private @LyValueError_Init(%header: memref<3xi64> {ly.ownership.object_header}, %old_message_header: memref<2xi64> {ly.ownership.object_header}, %old_message_bytes: memref<?xi8>, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) -> (memref<3xi64>, memref<2xi64>, memref<?xi8>) attributes {ly.ownership.owned_results = [0], ly.ownership.release_args = [1], ly.ownership.transfer_args = [0, 3], ly.runtime.contract = "builtins.ValueError", ly.runtime.method = "__init__", ly.runtime.result_evidence = "receiver"}
  func.func private @LyValueError_Raise(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0, 1], ly.runtime.contract = "builtins.ValueError", ly.runtime.primitive = "raise"}

  memref.global "private" constant @__ly_math_domain_msg : memref<17xi8> = dense<[109, 97, 116, 104, 32, 100, 111, 109, 97, 105, 110, 32, 101, 114, 114, 111, 114]>

  func.func private @__ly_math_domain_error() {
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

  func.func @LyMath_Floor(%header: memref<3xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "math.floor", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.float", ly.runtime.primitive = "math_floor", ly.runtime.result_contract = "builtins.int"} {
    %value = func.call @LyFloat_AsF64(%header) : (memref<3xi64>) -> f64
    %floored = math.floor %value : f64
    %as_int = arith.fptosi %floored : f64 to i64
    %int_header, %int_meta, %int_digits = func.call @LyLong_FromI64(%as_int) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %int_header, %int_meta, %int_digits : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyMath_Ceil(%header: memref<3xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "math.ceil", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.float", ly.runtime.primitive = "math_ceil", ly.runtime.result_contract = "builtins.int"} {
    %value = func.call @LyFloat_AsF64(%header) : (memref<3xi64>) -> f64
    %ceiled = math.ceil %value : f64
    %as_int = arith.fptosi %ceiled : f64 to i64
    %int_header, %int_meta, %int_digits = func.call @LyLong_FromI64(%as_int) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %int_header, %int_meta, %int_digits : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyMath_Trunc(%header: memref<3xi64> {ly.ownership.object_header}) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "math.trunc", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.float", ly.runtime.primitive = "math_trunc", ly.runtime.result_contract = "builtins.int"} {
    %value = func.call @LyFloat_AsF64(%header) : (memref<3xi64>) -> f64
    %truncated = math.trunc %value : f64
    %as_int = arith.fptosi %truncated : f64 to i64
    %int_header, %int_meta, %int_digits = func.call @LyLong_FromI64(%as_int) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %int_header, %int_meta, %int_digits : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyMath_Sqrt(%header: memref<3xi64> {ly.ownership.object_header}) -> memref<3xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "math.sqrt", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.float", ly.runtime.primitive = "math_sqrt", ly.runtime.result_contract = "builtins.float"} {
    %value = func.call @LyFloat_AsF64(%header) : (memref<3xi64>) -> f64
    %zero = arith.constant 0.0 : f64
    %negative = arith.cmpf olt, %value, %zero : f64
    cf.cond_br %negative, ^domain, ^ok

  ^domain:
    func.call @__ly_math_domain_error() : () -> ()
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
    func.call @__ly_math_domain_error() : () -> ()
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

  func.func @LyMath_Exp(%header: memref<3xi64> {ly.ownership.object_header}) -> memref<3xi64> attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "math.exp", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.float", ly.runtime.primitive = "math_exp", ly.runtime.result_contract = "builtins.float"} {
    %value = func.call @LyFloat_AsF64(%header) : (memref<3xi64>) -> f64
    %result = math.exp %value : f64
    %out_header = func.call @LyFloat_FromF64(%result) : (f64) -> memref<3xi64>
    func.return %out_header : memref<3xi64>
  }
}
