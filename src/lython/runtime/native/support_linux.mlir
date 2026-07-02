// Linux native runtime support.
//
// Layouts here are the glibc user-space ABI for stack_t and struct sigaction
// on 64-bit Linux.

!LinuxStackT = !llvm.struct<"LinuxStackT", (ptr, i32, i32, i64)>
!LinuxSigMask = !llvm.array<16 x i64>
!LinuxSigSet = !llvm.struct<"LinuxSigSet", (!LinuxSigMask)>
!LinuxSigAction = !llvm.struct<"LinuxSigAction", (ptr, !LinuxSigSet, i32, ptr)>
module {
  llvm.mlir.global internal @g_stack_guard_installed(false) {addr_space = 0 : i32, alignment = 4 : i64, dso_local} : i1
  llvm.mlir.global internal @g_main_stack_limit(0 : i64) {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : i64
  llvm.mlir.global internal @g_alternate_stack() {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !LinuxStackT {
    %zero = llvm.mlir.zero : !LinuxStackT
    llvm.return %zero : !LinuxStackT
  }
  llvm.mlir.global private unnamed_addr constant @".stack_guard_message"("RecursionError: maximum recursion depth exceeded (native stack overflow)\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.func @malloc(i64) -> (!llvm.ptr {llvm.noalias})
  llvm.func @write(i32, !llvm.ptr, i64) -> i64
  llvm.func @signal(i32, !llvm.ptr) -> !llvm.ptr
  llvm.func @pthread_self() -> !llvm.ptr
  llvm.func @pthread_getattr_np(!llvm.ptr, !llvm.ptr) -> i32
  llvm.func @pthread_attr_getstack(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
  llvm.func @pthread_attr_destroy(!llvm.ptr) -> i32
  llvm.func @sigaltstack(!llvm.ptr, !llvm.ptr) -> i32
  llvm.func @sigaction(i32, !llvm.ptr, !llvm.ptr) -> i32
  llvm.func @raise(i32) -> i32
  llvm.func @_exit(i32)
  llvm.func @LyRt_InstallStackGuard() {
    %0 = llvm.mlir.addressof @g_stack_guard_installed : !llvm.ptr
    %1 = llvm.mlir.constant(true) : i1
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.addressof @g_main_stack_limit : !llvm.ptr
    %5 = llvm.mlir.constant(0 : i64) : i64
    %6 = llvm.mlir.constant(524288 : i64) : i64
    %7 = llvm.mlir.zero : !llvm.ptr
    %8 = llvm.mlir.addressof @g_alternate_stack : !llvm.ptr
    %9 = llvm.mlir.constant(2 : i32) : i32
    %10 = llvm.mlir.constant(3 : i32) : i32
    %11 = llvm.mlir.addressof @stack_guard_handler : !llvm.ptr
    %12 = llvm.mlir.constant(dense<0> : tensor<16xi64>) : !LinuxSigMask
    %13 = llvm.mlir.undef : !LinuxSigSet
    %14 = llvm.insertvalue %12, %13[0] : !LinuxSigSet 
    %15 = llvm.mlir.constant(134217732 : i32) : i32
    %16 = llvm.mlir.constant(11 : i32) : i32
    %17 = llvm.mlir.constant(7 : i32) : i32
    %18 = llvm.load %0 {alignment = 4 : i64} : !llvm.ptr -> i1
    llvm.cond_br %18, ^bb7, ^bb1
  ^bb1:  // pred: ^bb0
    llvm.store %1, %0 {alignment = 4 : i64} : i1, !llvm.ptr
    %19 = llvm.alloca %2 x !llvm.array<128 x i8> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %20 = llvm.alloca %2 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %21 = llvm.alloca %2 x i64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %22 = llvm.call @pthread_self() : () -> !llvm.ptr
    %23 = llvm.call @pthread_getattr_np(%22, %19) : (!llvm.ptr, !llvm.ptr) -> i32
    %24 = llvm.icmp "eq" %23, %3 : i32
    llvm.cond_br %24, ^bb2, ^bb7
  ^bb2:  // pred: ^bb1
    %25 = llvm.call @pthread_attr_getstack(%19, %20, %21) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    %26 = llvm.call @pthread_attr_destroy(%19) : (!llvm.ptr) -> i32
    %27 = llvm.icmp "eq" %25, %3 : i32
    llvm.cond_br %27, ^bb3, ^bb7
  ^bb3:  // pred: ^bb2
    %28 = llvm.load %20 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %29 = llvm.ptrtoint %28 : !llvm.ptr to i64
    llvm.store %29, %4 {alignment = 8 : i64} : i64, !llvm.ptr
    %30 = llvm.icmp "eq" %29, %5 : i64
    llvm.cond_br %30, ^bb7, ^bb4
  ^bb4:  // pred: ^bb3
    %31 = llvm.call @malloc(%6) : (i64) -> !llvm.ptr
    %32 = llvm.icmp "eq" %31, %7 : !llvm.ptr
    llvm.cond_br %32, ^bb7, ^bb5
  ^bb5:  // pred: ^bb4
    %33 = llvm.getelementptr inbounds %8[%3, 0] : (!llvm.ptr, i32) -> !llvm.ptr, !LinuxStackT
    %34 = llvm.getelementptr inbounds %8[%3, 1] : (!llvm.ptr, i32) -> !llvm.ptr, !LinuxStackT
    %35 = llvm.getelementptr inbounds %8[%3, 2] : (!llvm.ptr, i32) -> !llvm.ptr, !LinuxStackT
    %36 = llvm.getelementptr inbounds %8[%3, 3] : (!llvm.ptr, i32) -> !llvm.ptr, !LinuxStackT
    llvm.store %31, %33 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %3, %34 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.store %3, %35 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.store %6, %36 {alignment = 8 : i64} : i64, !llvm.ptr
    %37 = llvm.call @sigaltstack(%8, %7) : (!llvm.ptr, !llvm.ptr) -> i32
    %38 = llvm.icmp "eq" %37, %3 : i32
    llvm.cond_br %38, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %39 = llvm.alloca %2 x !LinuxSigAction {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %40 = llvm.getelementptr inbounds %39[%3, 0] : (!llvm.ptr, i32) -> !llvm.ptr, !LinuxSigAction
    %41 = llvm.getelementptr inbounds %39[%3, 1] : (!llvm.ptr, i32) -> !llvm.ptr, !LinuxSigAction
    %42 = llvm.getelementptr inbounds %39[%3, 2] : (!llvm.ptr, i32) -> !llvm.ptr, !LinuxSigAction
    %43 = llvm.getelementptr inbounds %39[%3, 3] : (!llvm.ptr, i32) -> !llvm.ptr, !LinuxSigAction
    llvm.store %11, %40 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %14, %41 {alignment = 8 : i64} : !LinuxSigSet, !llvm.ptr
    llvm.store %15, %42 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.store %7, %43 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    %44 = llvm.call @sigaction(%16, %39, %7) : (i32, !llvm.ptr, !llvm.ptr) -> i32
    %45 = llvm.call @sigaction(%17, %39, %7) : (i32, !llvm.ptr, !llvm.ptr) -> i32
    llvm.br ^bb7
  ^bb7:  // 7 preds: ^bb0, ^bb1, ^bb2, ^bb3, ^bb4, ^bb5, ^bb6
    llvm.return
  }
  llvm.func internal @stack_guard_handler(%arg0: i32, %arg1: !llvm.ptr, %arg2: !llvm.ptr) attributes {dso_local} {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.addressof @g_main_stack_limit : !llvm.ptr
    %2 = llvm.mlir.constant(0 : i64) : i64
    %3 = llvm.mlir.constant(16 : i64) : i64
    %4 = llvm.mlir.constant(262144 : i64) : i64
    %5 = llvm.mlir.constant(4096 : i64) : i64
    %6 = llvm.mlir.constant(2 : i32) : i32
    %7 = llvm.mlir.addressof @".stack_guard_message" : !llvm.ptr
    %8 = llvm.mlir.constant(73 : i64) : i64
    %9 = llvm.mlir.constant(1 : i32) : i32
    %10 = llvm.icmp "eq" %arg1, %0 : !llvm.ptr
    llvm.cond_br %10, ^bb4, ^bb1
  ^bb1:  // pred: ^bb0
    %11 = llvm.load %1 {alignment = 8 : i64} : !llvm.ptr -> i64
    %12 = llvm.icmp "eq" %11, %2 : i64
    llvm.cond_br %12, ^bb4, ^bb2
  ^bb2:  // pred: ^bb1
    %13 = llvm.getelementptr %arg1[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %14 = llvm.load %13 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %15 = llvm.ptrtoint %14 : !llvm.ptr to i64
    %16 = llvm.sub %11, %4 : i64
    %17 = llvm.add %11, %5 : i64
    %18 = llvm.icmp "ult" %15, %16 : i64
    %19 = llvm.icmp "uge" %15, %17 : i64
    %20 = llvm.or %18, %19 : i1
    llvm.cond_br %20, ^bb4, ^bb3
  ^bb3:  // pred: ^bb2
    %21 = llvm.call @write(%6, %7, %8) : (i32, !llvm.ptr, i64) -> i64
    llvm.call @_exit(%9) : (i32) -> ()
    llvm.unreachable
  ^bb4:  // 3 preds: ^bb0, ^bb1, ^bb2
    %22 = llvm.call @signal(%arg0, %0) : (i32, !llvm.ptr) -> !llvm.ptr
    %23 = llvm.call @raise(%arg0) : (i32) -> i32
    llvm.return
  }
}
