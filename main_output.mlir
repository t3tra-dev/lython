module {
  "py.func"() ({
  ^bb0(%arg0: !py.int):
    "py.return"(%arg0) : (!py.int) -> ()
  }) {arg_names = ["x"], function_type = !py.funcsig<[!py.int] -> [!py.int]>, sym_name = "py_id"} : () -> ()
  "py.func"() ({
  ^bb0(%arg0: !py.int):
    %0 = "py.func.object"() {target = @py_id} : () -> !py.func<!py.funcsig<[!py.int] -> [!py.int]>>
    %1 = "py.tuple.create"(%arg0) : (!py.int) -> !py.tuple<!py.int>
    %2 = "py.tuple.empty"() : () -> !py.tuple<>
    %3 = "py.tuple.empty"() : () -> !py.tuple<>
    %4 = "py.call.vector"(%0, %1, %2, %3) : (!py.func<!py.funcsig<[!py.int] -> [!py.int]>>, !py.tuple<!py.int>, !py.tuple<>, !py.tuple<>) -> !py.int
    "py.return"(%4) : (!py.int) -> ()
  }) {arg_names = ["value"], function_type = !py.funcsig<[!py.int] -> [!py.int]>, sym_name = "py_use_id"} : () -> ()
  "py.func"() ({
  ^bb0(%arg0: !py.int):
    %0 = "py.func.object"() {target = @py_id} : () -> !py.func<!py.funcsig<[!py.int] -> [!py.int]>>
    %1 = "py.tuple.create"(%arg0) : (!py.int) -> !py.tuple<!py.int>
    %2 = "py.none"() : () -> !py.none
    %3 = "py.call"(%0, %1, %2) : (!py.func<!py.funcsig<[!py.int] -> [!py.int]>>, !py.tuple<!py.int>, !py.none) -> !py.int
    "py.return"(%3) : (!py.int) -> ()
  }) {arg_names = ["value"], function_type = !py.funcsig<[!py.int] -> [!py.int]>, sym_name = "py_use_call_none"} : () -> ()
  "py.func"() ({
  ^bb0(%arg0: !py.int):
    %0 = "py.func.object"() {target = @py_id} : () -> !py.func<!py.funcsig<[!py.int] -> [!py.int]>>
    %1 = "py.tuple.create"(%arg0) : (!py.int) -> !py.tuple<!py.int>
    %2 = "py.dict.empty"() : () -> !py.dict<!py.str, !py.object>
    %3 = "py.call"(%0, %1, %2) : (!py.func<!py.funcsig<[!py.int] -> [!py.int]>>, !py.tuple<!py.int>, !py.dict<!py.str, !py.object>) -> !py.int
    "py.return"(%3) : (!py.int) -> ()
  }) {arg_names = ["value"], function_type = !py.funcsig<[!py.int] -> [!py.int]>, sym_name = "py_use_call_dict"} : () -> ()
  "py.func"() ({
  ^bb0(%arg0: !py.int):
    %0 = "py.tuple.empty"() : () -> !py.tuple<>
    %1 = "py.tuple.empty"() : () -> !py.tuple<>
    %2 = "py.none"() : () -> !py.none
    %3 = "py.upcast"(%2) : (!py.none) -> !py.object
    %4 = "py.dict.empty"() : () -> !py.dict<!py.str, !py.object>
    %5 = "py.str.constant"() {value = "default"} : () -> !py.str
    %6 = "py.dict.insert"(%4, %5, %3) : (!py.dict<!py.str, !py.object>, !py.str, !py.object) -> !py.dict<!py.str, !py.object>
    %7 = "py.dict.empty"() : () -> !py.dict<!py.str, !py.object>
    %8 = "py.str.constant"() {value = "returns"} : () -> !py.str
    %9 = "py.dict.insert"(%7, %8, %3) : (!py.dict<!py.str, !py.object>, !py.str, !py.object) -> !py.dict<!py.str, !py.object>
    %10 = "py.str.constant"() {value = "demo_module"} : () -> !py.str
    %11 = "py.make_function"(%0, %6, %1, %9, %10) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>, target = @py_id} : (!py.tuple<>, !py.dict<!py.str, !py.object>, !py.tuple<>, !py.dict<!py.str, !py.object>, !py.str) -> !py.func<!py.funcsig<[!py.int] -> [!py.int]>>
    %12 = "py.tuple.create"(%arg0) : (!py.int) -> !py.tuple<!py.int>
    %13 = "py.dict.empty"() : () -> !py.dict<!py.str, !py.object>
    %14 = "py.call"(%11, %12, %13) : (!py.func<!py.funcsig<[!py.int] -> [!py.int]>>, !py.tuple<!py.int>, !py.dict<!py.str, !py.object>) -> !py.int
    "py.return"(%14) : (!py.int) -> ()
  }) {arg_names = ["value"], function_type = !py.funcsig<[!py.int] -> [!py.int]>, sym_name = "py_use_make_function"} : () -> ()
  "py.func"() ({
  ^bb0(%arg0: !py.int, %arg1: !py.int):
    %0 = "py.num.add"(%arg0, %arg1) : (!py.int, !py.int) -> !py.int
    "py.return"(%0) : (!py.int) -> ()
  }) {arg_names = ["lhs", "rhs"], function_type = !py.funcsig<[!py.int, !py.int] -> [!py.int]>, sym_name = "py_add"} : () -> ()
  "py.class"() ({
    "py.func"() ({
    ^bb0(%arg0: !py.class<"DemoCallable">, %arg1: !py.int):
      "py.return"(%arg1) : (!py.int) -> ()
    }) {arg_names = ["self", "value"], function_type = !py.funcsig<[!py.class<"DemoCallable">, !py.int] -> [!py.int]>, sym_name = "__call__"} : () -> ()
  }) {sym_name = "DemoCallable"} : () -> ()
  func.func @native_add(%arg0: i32, %arg1: i32) -> i32 {
    %0 = arith.addi %arg0, %arg1 : i32
    return %0 : i32
  }
  func.func @invoke_native(%arg0: i32, %arg1: i32) -> i32 {
    %0 = "py.make_native"() {target = @native_add} : () -> !py.prim.func<(i32, i32) -> (i32)>
    %1 = "py.native_call"(%0, %arg0, %arg1) : (!py.prim.func<(i32, i32) -> (i32)>, i32, i32) -> i32
    return %1 : i32
  }
  func.func @cast_bridge(%arg0: i32) -> i32 {
    %0 = "py.cast.from_prim"(%arg0) : (i32) -> !py.int
    %1 = "py.upcast"(%0) : (!py.int) -> !py.object
    %2 = "py.dict.empty"() : () -> !py.dict<!py.str, !py.object>
    %3 = "py.str.constant"() {value = "boxed"} : () -> !py.str
    %4 = "py.dict.insert"(%2, %3, %1) : (!py.dict<!py.str, !py.object>, !py.str, !py.object) -> !py.dict<!py.str, !py.object>
    %5 = "py.cast.to_prim"(%0) {mode = "exact"} : (!py.int) -> i32
    return %5 : i32
  }
}
