func.func @add(%a: i32, %b: i32) -> i32 {           // ← py.func → func.func
  %0 = arith.addi %a, %b : i32                      // ← py.add → arith.addi
  return %0 : i32
}
