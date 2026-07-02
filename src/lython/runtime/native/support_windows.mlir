module {
  llvm.mlir.global internal @g_stack_guard_installed(false) {addr_space = 0 : i32, alignment = 4 : i64, dso_local} : i1
  llvm.mlir.global private unnamed_addr constant @".stack_guard_message"("RecursionError: maximum recursion depth exceeded (native stack overflow)\0D\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.func @SetUnhandledExceptionFilter(!llvm.ptr) -> !llvm.ptr
  llvm.func @GetStdHandle(i32) -> !llvm.ptr
  llvm.func @WriteFile(!llvm.ptr, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr) -> i32
  llvm.func @ExitProcess(i32)
  llvm.func @LyRt_InstallStackGuard() {
    %installed_ptr = llvm.mlir.addressof @g_stack_guard_installed : !llvm.ptr
    %installed = llvm.load %installed_ptr {alignment = 4 : i64} : !llvm.ptr -> i1
    llvm.cond_br %installed, ^done, ^install
  ^install:
    %true = llvm.mlir.constant(true) : i1
    %handler = llvm.mlir.addressof @stack_guard_unhandled_filter : !llvm.ptr
    llvm.store %true, %installed_ptr {alignment = 4 : i64} : i1, !llvm.ptr
    %previous = llvm.call @SetUnhandledExceptionFilter(%handler) : (!llvm.ptr) -> !llvm.ptr
    llvm.br ^done
  ^done:
    llvm.return
  }
  llvm.func internal @stack_guard_unhandled_filter(%arg0: !llvm.ptr) -> i32 attributes {dso_local} {
    %null = llvm.mlir.zero : !llvm.ptr
    %zero = llvm.mlir.constant(0 : i32) : i32
    %is_null = llvm.icmp "eq" %arg0, %null : !llvm.ptr
    llvm.cond_br %is_null, ^continue_search, ^load_record
  ^load_record:
    %record = llvm.load %arg0 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %record_null = llvm.icmp "eq" %record, %null : !llvm.ptr
    llvm.cond_br %record_null, ^continue_search, ^check_code
  ^check_code:
    %code = llvm.load %record {alignment = 4 : i64} : !llvm.ptr -> i32
    %stack_overflow = llvm.mlir.constant(-1073741571 : i32) : i32
    %matches = llvm.icmp "eq" %code, %stack_overflow : i32
    llvm.cond_br %matches, ^report_stack_overflow, ^continue_search
  ^report_stack_overflow:
    %stderr_id = llvm.mlir.constant(-12 : i32) : i32
    %stderr = llvm.call @GetStdHandle(%stderr_id) : (i32) -> !llvm.ptr
    %msg = llvm.mlir.addressof @".stack_guard_message" : !llvm.ptr
    %len = llvm.mlir.constant(74 : i32) : i32
    %one = llvm.mlir.constant(1 : i32) : i32
    %written = llvm.alloca %one x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %ok = llvm.call @WriteFile(%stderr, %msg, %len, %written, %null) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr) -> i32
    llvm.call @ExitProcess(%one) : (i32) -> ()
    llvm.unreachable
  ^continue_search:
    llvm.return %zero : i32
  }
}
