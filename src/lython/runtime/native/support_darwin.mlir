// Darwin native runtime support.
//
// Keep ABI-sensitive stack and signal layouts out of the common runtime.

!DarwinSigAltStack = !llvm.struct<"DarwinSigAltStack", (ptr, i64, i32)>
!DarwinSigAction = !llvm.struct<"DarwinSigAction", (ptr, i32, i32)>
module {
  llvm.mlir.global internal @g_stack_guard_installed(false) {addr_space = 0 : i32, alignment = 4 : i64, dso_local} : i1
  llvm.mlir.global internal @g_main_stack_limit(0 : i64) {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : i64
  llvm.mlir.global internal @g_alternate_stack() {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !DarwinSigAltStack {
    %zero = llvm.mlir.zero : !DarwinSigAltStack
    llvm.return %zero : !DarwinSigAltStack
  }
  llvm.mlir.global private unnamed_addr constant @".stack_guard_message"("RecursionError: maximum recursion depth exceeded (native stack overflow)\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.func @malloc(i64) -> (!llvm.ptr {llvm.noalias})
  llvm.func @write(i32, !llvm.ptr, i64) -> i64
  llvm.func @signal(i32, !llvm.ptr) -> !llvm.ptr
  llvm.func @pthread_self() -> !llvm.ptr
  llvm.func @pthread_get_stackaddr_np(!llvm.ptr) -> !llvm.ptr
  llvm.func @pthread_get_stacksize_np(!llvm.ptr) -> i64
  llvm.func @sigaltstack(!llvm.ptr, !llvm.ptr) -> i32
  llvm.func @sigaction(i32, !llvm.ptr, !llvm.ptr) -> i32
  llvm.func @raise(i32) -> i32
  llvm.func @_exit(i32)
  llvm.func @LyRt_InstallStackGuard() {
    %0 = llvm.mlir.addressof @g_stack_guard_installed : !llvm.ptr
    %1 = llvm.mlir.constant(true) : i1
    %2 = llvm.mlir.addressof @g_main_stack_limit : !llvm.ptr
    %3 = llvm.mlir.constant(0 : i64) : i64
    %4 = llvm.mlir.constant(524288 : i64) : i64
    %5 = llvm.mlir.zero : !llvm.ptr
    %6 = llvm.mlir.addressof @g_alternate_stack : !llvm.ptr
    %7 = llvm.mlir.constant(0 : i32) : i32
    %8 = llvm.mlir.constant(1 : i32) : i32
    %9 = llvm.mlir.constant(2 : i32) : i32
    %10 = llvm.mlir.addressof @stack_guard_handler : !llvm.ptr
    %11 = llvm.mlir.constant(65 : i32) : i32
    %12 = llvm.mlir.constant(11 : i32) : i32
    %13 = llvm.mlir.constant(10 : i32) : i32
    %14 = llvm.load %0 {alignment = 4 : i64} : !llvm.ptr -> i1
    llvm.cond_br %14, ^bb5, ^bb1
  ^bb1:  // pred: ^bb0
    llvm.store %1, %0 {alignment = 4 : i64} : i1, !llvm.ptr
    %15 = llvm.call @pthread_self() : () -> !llvm.ptr
    %16 = llvm.call @pthread_get_stackaddr_np(%15) : (!llvm.ptr) -> !llvm.ptr
    %17 = llvm.ptrtoint %16 : !llvm.ptr to i64
    %18 = llvm.call @pthread_get_stacksize_np(%15) : (!llvm.ptr) -> i64
    %19 = llvm.sub %17, %18 : i64
    llvm.store %19, %2 {alignment = 8 : i64} : i64, !llvm.ptr
    %20 = llvm.icmp "eq" %19, %3 : i64
    llvm.cond_br %20, ^bb5, ^bb2
  ^bb2:  // pred: ^bb1
    %21 = llvm.call @malloc(%4) : (i64) -> !llvm.ptr
    %22 = llvm.icmp "eq" %21, %5 : !llvm.ptr
    llvm.cond_br %22, ^bb5, ^bb3
  ^bb3:  // pred: ^bb2
    %23 = llvm.getelementptr inbounds %6[%7, 0] : (!llvm.ptr, i32) -> !llvm.ptr, !DarwinSigAltStack
    %24 = llvm.getelementptr inbounds %6[%7, 1] : (!llvm.ptr, i32) -> !llvm.ptr, !DarwinSigAltStack
    %25 = llvm.getelementptr inbounds %6[%7, 2] : (!llvm.ptr, i32) -> !llvm.ptr, !DarwinSigAltStack
    llvm.store %21, %23 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %4, %24 {alignment = 8 : i64} : i64, !llvm.ptr
    llvm.store %7, %25 {alignment = 4 : i64} : i32, !llvm.ptr
    %26 = llvm.call @sigaltstack(%6, %5) : (!llvm.ptr, !llvm.ptr) -> i32
    %27 = llvm.icmp "eq" %26, %7 : i32
    llvm.cond_br %27, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %28 = llvm.alloca %8 x !DarwinSigAction {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %29 = llvm.getelementptr inbounds %28[%7, 0] : (!llvm.ptr, i32) -> !llvm.ptr, !DarwinSigAction
    %30 = llvm.getelementptr inbounds %28[%7, 1] : (!llvm.ptr, i32) -> !llvm.ptr, !DarwinSigAction
    %31 = llvm.getelementptr inbounds %28[%7, 2] : (!llvm.ptr, i32) -> !llvm.ptr, !DarwinSigAction
    llvm.store %10, %29 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %7, %30 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.store %11, %31 {alignment = 4 : i64} : i32, !llvm.ptr
    %32 = llvm.call @sigaction(%12, %28, %5) : (i32, !llvm.ptr, !llvm.ptr) -> i32
    %33 = llvm.call @sigaction(%13, %28, %5) : (i32, !llvm.ptr, !llvm.ptr) -> i32
    llvm.br ^bb5
  ^bb5:  // 5 preds: ^bb0, ^bb1, ^bb2, ^bb3, ^bb4
    llvm.return
  }
  llvm.func internal @stack_guard_handler(%arg0: i32, %arg1: !llvm.ptr, %arg2: !llvm.ptr) attributes {dso_local} {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.addressof @g_main_stack_limit : !llvm.ptr
    %2 = llvm.mlir.constant(0 : i64) : i64
    %3 = llvm.mlir.constant(24 : i64) : i64
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
