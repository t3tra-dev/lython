llvm.func @sum_list(ptr %xs, i64 %n) -> i32 {
entry:
  %sum = llvm.alloca i32
  llvm.store i32 0, ptr %sum
  %i0  = llvm.constant i64 0
  llvm.br ^loop(%i0)

^loop(%i: i64):
  %cond = llvm.icmp "slt" i64 %i, %n
  llvm.cond_br %cond, ^body, ^exit

^body:
  %ptr   = llvm.getelementptr %xs[%i] : (ptr, i64) -> ptr
  %x     = llvm.load %ptr : ptr -> i32
  %sold  = llvm.load %sum : ptr -> i32
  %snew  = llvm.add %sold, %x : i32
  llvm.store %snew, %sum
  %i1    = llvm.add %i, 1 : i64
  llvm.br ^loop(%i1)

^exit:
  %ret = llvm.load %sum : ptr -> i32
  llvm.return %ret : i32
}
