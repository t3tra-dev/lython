py.func @add(%a: !py.i32, %b: !py.i32) -> !py.i32 attributes {native = true} {
  %0 = py.add %a, %b : !py.i32
  py.return %0 : !py.i32
}
