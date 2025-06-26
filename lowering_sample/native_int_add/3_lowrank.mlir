llvm.func @add(i32 %arg0, i32 %arg1) -> i32 {
  %0 = llvm.add %arg0, %arg1 : i32
  llvm.return %0 : i32
}
