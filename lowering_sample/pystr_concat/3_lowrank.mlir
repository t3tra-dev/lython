llvm.func @PyStr_Concat(llvm.ptr %lhs, llvm.ptr %rhs) -> llvm.ptr

func.func @greet(%name: llvm.ptr) -> llvm.ptr {
  %cst = llvm.mlir.global @"str.Hello"() : (i64) -> llvm.ptr
  %0   = llvm.call @PyStr_Concat(%cst, %name) : (llvm.ptr, llvm.ptr) -> llvm.ptr
  return %0 : llvm.ptr
}
