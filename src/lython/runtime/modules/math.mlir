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
    "math.trunc"
  ],
  ly.typing.function_names = [
    "math.floor",
    "math.ceil",
    "math.sqrt",
    "math.fabs",
    "math.trunc"
  ],
  ly.typing.float_constant_names = ["math.pi", "math.e", "math.tau", "math.inf", "math.nan"],
  ly.typing.float_constant_values = [3.141592653589793 : f64, 2.718281828459045 : f64, 6.283185307179586 : f64, 0x7FF0000000000000 : f64, 0x7FF8000000000000 : f64],
  ly.typing.function_contracts = [
    !py.callable<[!py.contract<"builtins.float">], arg_names = ["x"], arg_defaults = [false], returns = [!py.contract<"builtins.int">]>,
    !py.callable<[!py.contract<"builtins.float">], arg_names = ["x"], arg_defaults = [false], returns = [!py.contract<"builtins.int">]>,
    !py.callable<[!py.contract<"builtins.float">], arg_names = ["x"], arg_defaults = [false], returns = [!py.contract<"builtins.float">]>,
    !py.callable<[!py.contract<"builtins.float">], arg_names = ["x"], arg_defaults = [false], returns = [!py.contract<"builtins.float">]>,
    !py.callable<[!py.contract<"builtins.float">], arg_names = ["x"], arg_defaults = [false], returns = [!py.contract<"builtins.int">]>
  ]
} {
  func.func private @LyFloat_AsF64(%header: memref<2xi64> {ly.ownership.object_header}, %payload: memref<1xf64>) -> f64 attributes {ly.runtime.contract = "builtins.float", ly.runtime.method = "__float__", ly.runtime.primitive = "unbox.f64"}
  func.func private @LyFloat_FromF64(%value: f64 {ly.runtime.default_f64 = 0.0 : f64}) -> (memref<2xi64>, memref<1xf64>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 2 : i64, ly.runtime.contract = "builtins.float", ly.runtime.initializer = "__new__"}
  func.func private @LyLong_FromI64(%value: i64 {ly.runtime.default_i64 = 0 : i64}) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.class_id = 1 : i64, ly.runtime.contract = "builtins.int", ly.runtime.initializer = "__new__"}

  func.func @LyMath_Floor(%header: memref<2xi64> {ly.ownership.object_header}, %payload: memref<1xf64>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "math.floor", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.float", ly.runtime.primitive = "math_floor", ly.runtime.result_contract = "builtins.int"} {
    %value = func.call @LyFloat_AsF64(%header, %payload) : (memref<2xi64>, memref<1xf64>) -> f64
    %floored = math.floor %value : f64
    %as_int = arith.fptosi %floored : f64 to i64
    %int_header, %int_meta, %int_digits = func.call @LyLong_FromI64(%as_int) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %int_header, %int_meta, %int_digits : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyMath_Ceil(%header: memref<2xi64> {ly.ownership.object_header}, %payload: memref<1xf64>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "math.ceil", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.float", ly.runtime.primitive = "math_ceil", ly.runtime.result_contract = "builtins.int"} {
    %value = func.call @LyFloat_AsF64(%header, %payload) : (memref<2xi64>, memref<1xf64>) -> f64
    %ceiled = math.ceil %value : f64
    %as_int = arith.fptosi %ceiled : f64 to i64
    %int_header, %int_meta, %int_digits = func.call @LyLong_FromI64(%as_int) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %int_header, %int_meta, %int_digits : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyMath_Trunc(%header: memref<2xi64> {ly.ownership.object_header}, %payload: memref<1xf64>) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "math.trunc", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.float", ly.runtime.primitive = "math_trunc", ly.runtime.result_contract = "builtins.int"} {
    %value = func.call @LyFloat_AsF64(%header, %payload) : (memref<2xi64>, memref<1xf64>) -> f64
    %truncated = math.trunc %value : f64
    %as_int = arith.fptosi %truncated : f64 to i64
    %int_header, %int_meta, %int_digits = func.call @LyLong_FromI64(%as_int) : (i64) -> (memref<2xi64>, memref<2xi64>, memref<?xi32>)
    func.return %int_header, %int_meta, %int_digits : memref<2xi64>, memref<2xi64>, memref<?xi32>
  }

  func.func @LyMath_Sqrt(%header: memref<2xi64> {ly.ownership.object_header}, %payload: memref<1xf64>) -> (memref<2xi64>, memref<1xf64>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "math.sqrt", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.float", ly.runtime.primitive = "math_sqrt", ly.runtime.result_contract = "builtins.float"} {
    %value = func.call @LyFloat_AsF64(%header, %payload) : (memref<2xi64>, memref<1xf64>) -> f64
    %root = math.sqrt %value : f64
    %out_header, %out_payload = func.call @LyFloat_FromF64(%root) : (f64) -> (memref<2xi64>, memref<1xf64>)
    func.return %out_header, %out_payload : memref<2xi64>, memref<1xf64>
  }

  func.func @LyMath_Fabs(%header: memref<2xi64> {ly.ownership.object_header}, %payload: memref<1xf64>) -> (memref<2xi64>, memref<1xf64>) attributes {ly.ownership.owned_results = [0], ly.runtime.builtin = "math.fabs", ly.runtime.builtin_lowering = "direct", ly.runtime.contract = "builtins.float", ly.runtime.primitive = "math_fabs", ly.runtime.result_contract = "builtins.float"} {
    %value = func.call @LyFloat_AsF64(%header, %payload) : (memref<2xi64>, memref<1xf64>) -> f64
    %magnitude = math.absf %value : f64
    %out_header, %out_payload = func.call @LyFloat_FromF64(%magnitude) : (f64) -> (memref<2xi64>, memref<1xf64>)
    func.return %out_header, %out_payload : memref<2xi64>, memref<1xf64>
  }
}
