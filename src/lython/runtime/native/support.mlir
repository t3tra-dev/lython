// Native runtime support shared by every target OS.
//
// This file is the source of truth. CMake translates it to LLVM IR text and
// strips export-only metadata before embedding it in the compiler.

!I64MemRef = !llvm.struct<"I64MemRef", (ptr, ptr, i64, i64, i64)>
!I8MemRef = !llvm.struct<"I8MemRef", (ptr, ptr, i64, i64, i64)>
!ExceptionParts = !llvm.struct<"ExceptionParts", (!I64MemRef, !I64MemRef, !I8MemRef)>
!TracebackFrame = !llvm.struct<"TracebackFrame", (ptr, ptr, i32, i32, i32, i32, i32, i32)>
!TracebackStack = !llvm.array<1024 x !TracebackFrame>
!ExceptionTypeInfo = !llvm.struct<(ptr, ptr)>
module {
  llvm.mlir.global internal @g_traceback_size(0 : i64) {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : i64
  llvm.mlir.global internal @g_traceback_stack() {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !TracebackStack {
    %zero = llvm.mlir.zero : !TracebackStack
    llvm.return %zero : !TracebackStack
  }
  llvm.mlir.global internal @g_current_exception(false) {addr_space = 0 : i32, alignment = 4 : i64, dso_local} : i1
  llvm.mlir.global internal @g_native_catch_active(false) {addr_space = 0 : i32, alignment = 4 : i64, dso_local} : i1
  llvm.mlir.global internal @g_current_parts() {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !ExceptionParts {
    %zero = llvm.mlir.zero : !ExceptionParts
    llvm.return %zero : !ExceptionParts
  }
  llvm.mlir.global private unnamed_addr constant @".empty"(dense<0> : tensor<1xi8>) {addr_space = 0 : i32, alignment = 1 : i64, dso_local} : !llvm.array<1 x i8>
  llvm.mlir.global private unnamed_addr constant @".newline"("\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".indent"("    \00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".read_mode"("r\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".traceback_header"("Traceback (most recent call last):\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".fmt_frame"("  File \22%s\22, line %d, in %s\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".fmt_class"("%s\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".fmt_invalid"("%s: <invalid>\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".fmt_unknown"("%s: <unknown>\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".fmt_message"("%s: %s\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".fmt_i64"("%lld\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".fmt_f64"("%.12g\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".repr_none"("None\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".repr_ellipsis"("...\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".repr_object"("<object object>\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".native_exception"("error: uncaught native exception during Python execution\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".class.BaseException"("BaseException\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".class.Exception"("Exception\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".class.RuntimeError"("RuntimeError\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".class.TypeError"("TypeError\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".class.ValueError"("ValueError\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".class.KeyError"("KeyError\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".class.IndexError"("IndexError\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".class.AssertionError"("AssertionError\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".class.StopIteration"("StopIteration\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".class.StopAsyncIteration"("StopAsyncIteration\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".class.ArithmeticError"("ArithmeticError\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".class.LookupError"("LookupError\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".class.ZeroDivisionError"("ZeroDivisionError\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".class.CancelledError"("CancelledError\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global linkonce_odr hidden constant @_ZTI17LyPythonException() {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !ExceptionTypeInfo {
    %0 = llvm.mlir.addressof @_ZTS17LyPythonException : !llvm.ptr
    %1 = llvm.mlir.constant(2 : i64) : i64
    %2 = llvm.mlir.addressof @_ZTVN10__cxxabiv117__class_type_infoE : !llvm.ptr
    %3 = llvm.getelementptr inbounds %2[%1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %4 = llvm.mlir.undef : !ExceptionTypeInfo
    %5 = llvm.insertvalue %3, %4[0] : !ExceptionTypeInfo 
    %6 = llvm.insertvalue %0, %5[1] : !ExceptionTypeInfo 
    llvm.return %6 : !ExceptionTypeInfo
  }
  llvm.mlir.global external @_ZTVN10__cxxabiv117__class_type_infoE() {addr_space = 0 : i32} : !llvm.array<0 x ptr>
  llvm.mlir.global linkonce_odr hidden constant @_ZTS17LyPythonException("17LyPythonException\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.func @abort()
  llvm.func @malloc(i64) -> (!llvm.ptr {llvm.noalias})
  llvm.func @free(!llvm.ptr)
  llvm.func @write(i32, !llvm.ptr, i64) -> i64
  llvm.func @fopen(!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  llvm.func @fgets(!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
  llvm.func @fclose(!llvm.ptr) -> i32
  llvm.func @strlen(!llvm.ptr) -> i64
  llvm.func @snprintf(!llvm.ptr, i64, !llvm.ptr, ...) -> i32
  llvm.func @pow(f64, f64) -> f64
  llvm.func @nearbyint(f64) -> f64
  llvm.func @_exit(i32)
  llvm.func @LyRt_InstallStackGuard()
  llvm.func @__cxa_allocate_exception(i64) -> !llvm.ptr
  llvm.func @__cxa_throw(!llvm.ptr, !llvm.ptr, !llvm.ptr)
  llvm.func @__cxa_begin_catch(!llvm.ptr) -> !llvm.ptr
  llvm.func @__cxa_end_catch()
  llvm.func @__gxx_personality_v0(...) -> i32
  llvm.func internal @rt_abort() attributes {dso_local} {
    llvm.call @abort() : () -> ()
    llvm.unreachable
  }

  func.func private @boxed_slot_ptr(%arg0: !llvm.ptr, %arg1: i64) -> !llvm.ptr {
    %0 = llvm.getelementptr %arg0[%arg1] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    func.return %0 : !llvm.ptr
  }

  func.func private @boxed_load_i64(%arg0: !llvm.ptr, %arg1: i64) -> i64 {
    %0 = func.call @boxed_slot_ptr(%arg0, %arg1) : (!llvm.ptr, i64) -> !llvm.ptr
    %1 = llvm.load %0 {alignment = 8 : i64} : !llvm.ptr -> i64
    func.return %1 : i64
  }

  func.func private @free_raw_i64_ptr(%arg0: i64) {
    %0 = arith.constant 0 : i64
    %1 = arith.cmpi eq, %arg0, %0 : i64
    cf.cond_br %1, ^bb2, ^bb1
  ^bb1:
    %2 = llvm.inttoptr %arg0 : i64 to !llvm.ptr
    llvm.call @free(%2) : (!llvm.ptr) -> ()
    cf.br ^bb2
  ^bb2:
    func.return
  }

  func.func private @release_storage_raw_to_zero(%arg0: i64) -> i1 {
    %0 = arith.constant 0 : i64
    %1 = arith.constant 1 : i64
    %immortal = arith.constant 9223372036854775807 : i64
    %false = arith.constant false
    %is_null = arith.cmpi eq, %arg0, %0 : i64
    cf.cond_br %is_null, ^bb_done, ^bb_tag
  ^bb_tag:
    %tag = arith.andi %arg0, %1 : i64
    %is_tagged = arith.cmpi eq, %tag, %1 : i64
    cf.cond_br %is_tagged, ^bb_done, ^bb_probe
  ^bb_probe:
    %ptr = llvm.inttoptr %arg0 : i64 to !llvm.ptr
    %observed = llvm.load %ptr {alignment = 8 : i64} : !llvm.ptr -> i64
    %pre_immortal = arith.cmpi eq, %observed, %immortal : i64
    cf.cond_br %pre_immortal, ^bb_done, ^bb_positive
  ^bb_positive:
    %observed_positive = arith.cmpi sgt, %observed, %0 : i64
    cf.assert %observed_positive, "release_storage_raw_to_zero observed non-positive refcount"
    %previous = llvm.atomicrmw sub %ptr, %1 acq_rel : !llvm.ptr, i64
    %previous_positive = arith.cmpi sgt, %previous, %0 : i64
    cf.assert %previous_positive, "release_storage_raw_to_zero raced with non-positive refcount"
    %became_zero = arith.cmpi eq, %previous, %1 : i64
    func.return %became_zero : i1
  ^bb_done:
    func.return %false : i1
  }

  func.func private @release_unicode_raw(%arg0: i64, %arg1: i64) {
    %0 = func.call @release_storage_raw_to_zero(%arg0) : (i64) -> i1
    cf.cond_br %0, ^bb1, ^bb2
  ^bb1:
    func.call @free_raw_i64_ptr(%arg1) : (i64) -> ()
    func.call @free_raw_i64_ptr(%arg0) : (i64) -> ()
    cf.br ^bb2
  ^bb2:
    func.return
  }

  func.func private @release_single_storage_raw(%arg0: i64) {
    %0 = func.call @release_storage_raw_to_zero(%arg0) : (i64) -> i1
    cf.cond_br %0, ^bb1, ^bb2
  ^bb1:
    func.call @free_raw_i64_ptr(%arg0) : (i64) -> ()
    cf.br ^bb2
  ^bb2:
    func.return
  }

  func.func private @release_task_raw(%arg0: i64, %arg1: i64) {
    %0 = func.call @release_storage_raw_to_zero(%arg0) : (i64) -> i1
    cf.cond_br %0, ^bb1, ^bb2
  ^bb1:
    func.call @release_single_storage_raw(%arg1) : (i64) -> ()
    func.call @free_raw_i64_ptr(%arg0) : (i64) -> ()
    cf.br ^bb2
  ^bb2:
    func.return
  }

  func.func private @release_sequence_payload(%arg0: i64, %arg1: i64) {
    %0 = arith.constant 0 : i64
    %1 = arith.constant 1 : i64
    %2 = arith.constant 16 : i64
    %3 = arith.cmpi eq, %arg0, %0 : i64
    %4 = arith.cmpi eq, %arg1, %0 : i64
    %5 = arith.ori %3, %4 : i1
    cf.cond_br %5, ^bb3, ^bb1
  ^bb1:
    %6 = llvm.inttoptr %arg0 : i64 to !llvm.ptr
    %7 = llvm.load %6 {alignment = 8 : i64} : !llvm.ptr -> i64
    %8 = arith.index_cast %7 : i64 to index
    %9 = llvm.inttoptr %arg1 : i64 to !llvm.ptr
    %10 = arith.constant 0 : index
    %11 = arith.constant 1 : index
    scf.for %arg2 = %10 to %8 step %11 {
      %12 = arith.index_cast %arg2 : index to i64
      %13 = arith.muli %12, %2 : i64
      %14 = llvm.getelementptr %9[%13] : (!llvm.ptr, i64) -> !llvm.ptr, i64
      func.call @release_payload_slot_ptr(%14) : (!llvm.ptr) -> ()
    }
    cf.br ^bb3
  ^bb3:
    func.return
  }

  func.func private @release_dict_payload(%arg0: i64, %arg1: i64, %arg2: i64, %arg3: i64) {
    %0 = arith.constant 0 : i64
    %1 = arith.constant 1 : i64
    %2 = arith.constant 16 : i64
    %3 = arith.cmpi eq, %arg0, %0 : i64
    %4 = arith.cmpi eq, %arg1, %0 : i64
    %5 = arith.cmpi eq, %arg2, %0 : i64
    %6 = arith.cmpi eq, %arg3, %0 : i64
    %7 = arith.ori %3, %4 : i1
    %8 = arith.ori %5, %6 : i1
    %9 = arith.ori %7, %8 : i1
    cf.cond_br %9, ^bb3, ^bb1
  ^bb1:
    %10 = llvm.inttoptr %arg0 : i64 to !llvm.ptr
    %11 = llvm.getelementptr %10[%1] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %12 = llvm.load %11 {alignment = 8 : i64} : !llvm.ptr -> i64
    %13 = arith.index_cast %12 : i64 to index
    %14 = llvm.inttoptr %arg1 : i64 to !llvm.ptr
    %15 = llvm.inttoptr %arg2 : i64 to !llvm.ptr
    %16 = llvm.inttoptr %arg3 : i64 to !llvm.ptr
    %17 = arith.constant 0 : index
    %18 = arith.constant 1 : index
    scf.for %arg4 = %17 to %13 step %18 {
      %19 = arith.index_cast %arg4 : index to i64
      %20 = llvm.getelementptr %16[%19] : (!llvm.ptr, i64) -> !llvm.ptr, i64
      %21 = llvm.load %20 {alignment = 8 : i64} : !llvm.ptr -> i64
      %22 = arith.cmpi ne, %21, %0 : i64
      scf.if %22 {
        %23 = arith.muli %19, %2 : i64
        %24 = llvm.getelementptr %14[%23] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        %25 = llvm.getelementptr %15[%23] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        func.call @release_payload_slot_ptr(%24) : (!llvm.ptr) -> ()
        func.call @release_payload_slot_ptr(%25) : (!llvm.ptr) -> ()
      }
    }
    cf.br ^bb3
  ^bb3:
    func.return
  }

  func.func private @release_payload_slot_ptr(%arg0: !llvm.ptr) {
    %0 = arith.constant 0 : i64
    %1 = arith.constant 1 : i64
    %2 = arith.constant 2 : i64
    %3 = arith.constant 3 : i64
    %4 = arith.constant 4 : i64
    %5 = arith.constant 5 : i64
    %6 = arith.constant 6 : i64
    %7 = arith.constant 7 : i64
    %8 = arith.constant 8 : i64
    %9 = arith.constant 9 : i64
    %10 = arith.constant 10 : i64
    %11 = arith.constant 11 : i64
    %12 = arith.constant 12 : i64
    %13 = arith.constant 13 : i64
    %14 = arith.constant 14 : i64
    %15 = arith.constant 15 : i64
    %16 = arith.constant 16 : i64
    %17 = arith.constant 17 : i64
    %c18 = arith.constant 18 : i64
    %owned_slot = arith.constant 14 : i64
    %18 = func.call @boxed_load_i64(%arg0, %owned_slot) : (!llvm.ptr, i64) -> i64
    %19 = arith.cmpi eq, %18, %0 : i64
    cf.cond_br %19, ^bb_done, ^bb_owned
  ^bb_owned:
    %20 = func.call @boxed_load_i64(%arg0, %2) : (!llvm.ptr, i64) -> i64
    %21 = func.call @release_storage_raw_to_zero(%20) : (i64) -> i1
    cf.cond_br %21, ^bb_zero, ^bb_done
  ^bb_zero:
    %22 = func.call @boxed_load_i64(%arg0, %1) : (!llvm.ptr, i64) -> i64
    %23 = func.call @boxed_load_i64(%arg0, %4) : (!llvm.ptr, i64) -> i64
    %24 = func.call @boxed_load_i64(%arg0, %5) : (!llvm.ptr, i64) -> i64
    %25 = func.call @boxed_load_i64(%arg0, %6) : (!llvm.ptr, i64) -> i64
    %26 = func.call @boxed_load_i64(%arg0, %7) : (!llvm.ptr, i64) -> i64
    %27 = func.call @boxed_load_i64(%arg0, %8) : (!llvm.ptr, i64) -> i64
    %28 = arith.cmpi eq, %22, %1 : i64
    cf.cond_br %28, ^bb_int, ^bb_check_float
  ^bb_check_float:
    %29 = arith.cmpi eq, %22, %2 : i64
    cf.cond_br %29, ^bb_float, ^bb_check_str
  ^bb_check_str:
    %30 = arith.cmpi eq, %22, %4 : i64
    cf.cond_br %30, ^bb_str, ^bb_check_exc
  ^bb_check_exc:
    %31 = arith.cmpi eq, %22, %5 : i64
    cf.cond_br %31, ^bb_exc, ^bb_check_list
  ^bb_check_list:
    %32 = arith.cmpi eq, %22, %10 : i64
    cf.cond_br %32, ^bb_seq, ^bb_check_tuple
  ^bb_check_tuple:
    %33 = arith.cmpi eq, %22, %11 : i64
    cf.cond_br %33, ^bb_seq, ^bb_check_dict
  ^bb_check_dict:
    %34 = arith.cmpi eq, %22, %12 : i64
    cf.cond_br %34, ^bb_dict, ^bb_check_single6
  ^bb_check_single6:
    %35 = arith.cmpi eq, %22, %6 : i64
    cf.cond_br %35, ^bb_single, ^bb_check_single8
  ^bb_check_single8:
    %36 = arith.cmpi eq, %22, %8 : i64
    cf.cond_br %36, ^bb_single, ^bb_check_single9
  ^bb_check_single9:
    %37 = arith.cmpi eq, %22, %9 : i64
    cf.cond_br %37, ^bb_single, ^bb_check_single13
  ^bb_check_single13:
    %38 = arith.cmpi eq, %22, %13 : i64
    cf.cond_br %38, ^bb_single, ^bb_check_single14
  ^bb_check_single14:
    %39 = arith.cmpi eq, %22, %14 : i64
    cf.cond_br %39, ^bb_single, ^bb_check_task
  ^bb_check_task:
    %task = arith.cmpi eq, %22, %15 : i64
    cf.cond_br %task, ^bb_task, ^bb_check_future_iter
  ^bb_check_future_iter:
    %40 = arith.cmpi eq, %22, %16 : i64
    cf.cond_br %40, ^bb_child_single, ^bb_check_task_iter
  ^bb_check_task_iter:
    %41 = arith.cmpi eq, %22, %17 : i64
    cf.cond_br %41, ^bb_task_iter, ^bb_check_await_iter
  ^bb_check_await_iter:
    %await_iter = arith.cmpi eq, %22, %c18 : i64
    cf.cond_br %await_iter, ^bb_child_single, ^bb_header_only
  ^bb_int:
    func.call @free_raw_i64_ptr(%25) : (i64) -> ()
    func.call @free_raw_i64_ptr(%24) : (i64) -> ()
    func.call @free_raw_i64_ptr(%23) : (i64) -> ()
    cf.br ^bb_done
  ^bb_float:
    func.call @free_raw_i64_ptr(%24) : (i64) -> ()
    func.call @free_raw_i64_ptr(%23) : (i64) -> ()
    cf.br ^bb_done
  ^bb_str:
    func.call @free_raw_i64_ptr(%24) : (i64) -> ()
    func.call @free_raw_i64_ptr(%23) : (i64) -> ()
    cf.br ^bb_done
  ^bb_exc:
    func.call @release_unicode_raw(%24, %25) : (i64, i64) -> ()
    func.call @free_raw_i64_ptr(%23) : (i64) -> ()
    cf.br ^bb_done
  ^bb_seq:
    func.call @release_sequence_payload(%24, %25) : (i64, i64) -> ()
    func.call @free_raw_i64_ptr(%25) : (i64) -> ()
    func.call @free_raw_i64_ptr(%24) : (i64) -> ()
    func.call @free_raw_i64_ptr(%23) : (i64) -> ()
    cf.br ^bb_done
  ^bb_dict:
    func.call @release_dict_payload(%24, %25, %26, %27) : (i64, i64, i64, i64) -> ()
    func.call @free_raw_i64_ptr(%27) : (i64) -> ()
    func.call @free_raw_i64_ptr(%26) : (i64) -> ()
    func.call @free_raw_i64_ptr(%25) : (i64) -> ()
    func.call @free_raw_i64_ptr(%24) : (i64) -> ()
    func.call @free_raw_i64_ptr(%23) : (i64) -> ()
    cf.br ^bb_done
  ^bb_single:
    func.call @free_raw_i64_ptr(%23) : (i64) -> ()
    cf.br ^bb_done
  ^bb_child_single:
    func.call @release_single_storage_raw(%24) : (i64) -> ()
    func.call @free_raw_i64_ptr(%23) : (i64) -> ()
    cf.br ^bb_done
  ^bb_task:
    func.call @release_single_storage_raw(%24) : (i64) -> ()
    func.call @free_raw_i64_ptr(%23) : (i64) -> ()
    cf.br ^bb_done
  ^bb_task_iter:
    func.call @release_task_raw(%24, %25) : (i64, i64) -> ()
    func.call @free_raw_i64_ptr(%23) : (i64) -> ()
    cf.br ^bb_done
  ^bb_header_only:
    func.call @free_raw_i64_ptr(%23) : (i64) -> ()
    cf.br ^bb_done
  ^bb_done:
    func.return
  }

  func.func @LyObject_ReleaseBoxedPayloadRaw(%arg0: memref<16xi64>) {
    %0 = memref.extract_aligned_pointer_as_index %arg0 : memref<16xi64> -> index
    %1 = arith.index_cast %0 : index to i64
    %2 = llvm.inttoptr %1 : i64 to !llvm.ptr
    func.call @release_payload_slot_ptr(%2) : (!llvm.ptr) -> ()
    func.return
  }

  func.func @LyObject_ReleaseBoxedPayloadArraySlotRaw(%arg0: memref<?xi64>, %arg1: i64) {
    %0 = arith.constant 16 : i64
    %1 = memref.extract_aligned_pointer_as_index %arg0 : memref<?xi64> -> index
    %2 = arith.index_cast %1 : index to i64
    %3 = llvm.inttoptr %2 : i64 to !llvm.ptr
    %4 = arith.muli %arg1, %0 : i64
    %5 = llvm.getelementptr %3[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    func.call @release_payload_slot_ptr(%5) : (!llvm.ptr) -> ()
    func.return
  }

  func.func private @write_len(%arg0: i32, %arg1: !llvm.ptr, %arg2: i64) {
    %0 = arith.constant 0 : i64
    %1 = llvm.mlir.zero : !llvm.ptr
    %2 = arith.cmpi sgt, %arg2, %0 : i64
    %3 = llvm.icmp "ne" %arg1, %1 : !llvm.ptr
    %4 = arith.andi %2, %3 : i1
    cf.cond_br %4, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %5 = llvm.call @write(%arg0, %arg1, %arg2) : (i32, !llvm.ptr, i64) -> i64
    cf.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    func.return
  }
  func.func private @write_cstr(%arg0: i32, %arg1: !llvm.ptr) {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.icmp "eq" %arg1, %0 : !llvm.ptr
    cf.cond_br %1, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    %2 = llvm.call @strlen(%arg1) : (!llvm.ptr) -> i64
    func.call @write_len(%arg0, %arg1, %2) : (i32, !llvm.ptr, i64) -> ()
    cf.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    func.return
  }
  func.func private @write_char(%arg0: i32, %arg1: i8) {
    %0 = arith.constant 0 : index
    %1 = arith.constant 1 : i64
    %2 = memref.alloca() : memref<1xi8>
    memref.store %arg1, %2[%0] : memref<1xi8>
    %3 = memref.extract_aligned_pointer_as_index %2 : memref<1xi8> -> index
    %4 = arith.index_cast %3 : index to i64
    %5 = llvm.inttoptr %4 : i64 to !llvm.ptr
    func.call @write_len(%arg0, %5, %1) : (i32, !llvm.ptr, i64) -> ()
    func.return
  }
  func.func private @write_buffered(%arg0: i32, %arg1: !llvm.ptr, %arg2: i32) {
    %0 = arith.constant 0 : i32
    %1 = arith.constant 1023 : i64
    %2 = arith.cmpi slt, %arg2, %0 : i32
    cf.cond_br %2, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    %3 = arith.extsi %arg2 : i32 to i64
    %4 = arith.cmpi sgt, %3, %1 : i64
    %5 = arith.select %4, %1, %3 : i64
    func.call @write_len(%arg0, %arg1, %5) : (i32, !llvm.ptr, i64) -> ()
    cf.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    func.return
  }

  func.func private @write_i64(%arg0: i32, %arg1: i64) {
    %0 = arith.constant 1 : i32
    %1 = arith.constant 0 : i64
    %2 = arith.constant 64 : i64
    %3 = llvm.mlir.addressof @".fmt_i64" : !llvm.ptr
    %4 = llvm.alloca %0 x !llvm.array<64 x i8> {alignment = 1 : i64} : (i32) -> !llvm.ptr
    %5 = llvm.getelementptr inbounds %4[%1, %1] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<64 x i8>
    %6 = llvm.call @snprintf(%5, %2, %3, %arg1) vararg(!llvm.func<i32 (ptr, i64, ptr, ...)>) : (!llvm.ptr, i64, !llvm.ptr, i64) -> i32
    func.call @write_buffered(%arg0, %5, %6) : (i32, !llvm.ptr, i32) -> ()
    func.return
  }

  func.func private @write_f64(%arg0: i32, %arg1: f64) {
    %0 = arith.constant 1 : i32
    %1 = arith.constant 0 : i64
    %2 = arith.constant 64 : i64
    %3 = llvm.mlir.addressof @".fmt_f64" : !llvm.ptr
    %4 = llvm.alloca %0 x !llvm.array<64 x i8> {alignment = 1 : i64} : (i32) -> !llvm.ptr
    %5 = llvm.getelementptr inbounds %4[%1, %1] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<64 x i8>
    %6 = llvm.call @snprintf(%5, %2, %3, %arg1) vararg(!llvm.func<i32 (ptr, i64, ptr, ...)>) : (!llvm.ptr, i64, !llvm.ptr, f64) -> i32
    func.call @write_buffered(%arg0, %5, %6) : (i32, !llvm.ptr, i32) -> ()
    func.return
  }

  func.func private @write_raw_bytes(%arg0: i32, %arg1: i64, %arg2: i64) {
    %0 = arith.constant 0 : i64
    %1 = arith.cmpi ne, %arg1, %0 : i64
    %2 = arith.cmpi sgt, %arg2, %0 : i64
    %3 = arith.andi %1, %2 : i1
    scf.if %3 {
      %4 = llvm.inttoptr %arg1 : i64 to !llvm.ptr
      func.call @write_len(%arg0, %4, %arg2) : (i32, !llvm.ptr, i64) -> ()
    }
    func.return
  }

  func.func private @print_boxed_int(%arg0: i32, %arg1: i64, %arg2: i64) {
    %0 = arith.constant 0 : i64
    %1 = arith.constant 1 : i64
    %2 = arith.constant 1073741824 : i64
    %3 = arith.cmpi eq, %arg1, %0 : i64
    %4 = arith.cmpi eq, %arg2, %0 : i64
    %5 = arith.ori %3, %4 : i1
    cf.cond_br %5, ^bb_fallback, ^bb_decode
  ^bb_decode:
    %6 = llvm.inttoptr %arg1 : i64 to !llvm.ptr
    %7 = llvm.inttoptr %arg2 : i64 to !llvm.ptr
    %8 = llvm.getelementptr %6[%0] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %9 = llvm.getelementptr %6[%1] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %10 = llvm.load %8 {alignment = 8 : i64} : !llvm.ptr -> i64
    %11 = llvm.load %9 {alignment = 8 : i64} : !llvm.ptr -> i64
    %12 = arith.cmpi sle, %11, %0 : i64
    cf.cond_br %12, ^bb_zero, ^bb_digits
  ^bb_zero:
    func.call @write_i64(%arg0, %0) : (i32, i64) -> ()
    func.return
  ^bb_digits:
    %13 = arith.index_cast %11 : i64 to index
    %14 = arith.constant 0 : index
    %15 = arith.constant 1 : index
    %16 = scf.for %arg3 = %14 to %13 step %15 iter_args(%arg4 = %0) -> (i64) {
      %21 = arith.index_cast %arg3 : index to i64
      %22 = arith.addi %21, %1 : i64
      %23 = arith.subi %11, %22 : i64
      %24 = llvm.getelementptr %7[%23] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %25 = llvm.load %24 {alignment = 4 : i64} : !llvm.ptr -> i32
      %26 = arith.extui %25 : i32 to i64
      %27 = arith.muli %arg4, %2 : i64
      %28 = arith.addi %27, %26 : i64
      scf.yield %28 : i64
    }
    %17 = arith.cmpi slt, %10, %0 : i64
    %18 = arith.subi %0, %16 : i64
    %19 = arith.select %17, %18, %16 : i1, i64
    func.call @write_i64(%arg0, %19) : (i32, i64) -> ()
    func.return
  ^bb_fallback:
    %20 = llvm.mlir.addressof @".repr_object" : !llvm.ptr
    func.call @write_cstr(%arg0, %20) : (i32, !llvm.ptr) -> ()
    func.return
  }

  func.func private @print_boxed_float(%arg0: i32, %arg1: i64) {
    %0 = arith.constant 0 : i64
    %1 = arith.cmpi eq, %arg1, %0 : i64
    cf.cond_br %1, ^bb_fallback, ^bb_value
  ^bb_value:
    %2 = llvm.inttoptr %arg1 : i64 to !llvm.ptr
    %3 = llvm.getelementptr %2[%0] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %4 = llvm.load %3 {alignment = 8 : i64} : !llvm.ptr -> f64
    func.call @write_f64(%arg0, %4) : (i32, f64) -> ()
    func.return
  ^bb_fallback:
    %5 = llvm.mlir.addressof @".repr_object" : !llvm.ptr
    func.call @write_cstr(%arg0, %5) : (i32, !llvm.ptr) -> ()
    func.return
  }

  func.func private @print_boxed_sequence(%arg0: i32, %arg1: i64, %arg2: i64, %arg3: i8, %arg4: i8, %arg5: i1, %arg6: i64) {
    %0 = arith.constant 0 : i64
    %1 = arith.constant 1 : i64
    %2 = arith.constant 16 : i64
    %3 = arith.constant 44 : i8
    %4 = arith.constant 32 : i8
    %5 = arith.cmpi eq, %arg1, %0 : i64
    %6 = arith.cmpi eq, %arg2, %0 : i64
    %7 = arith.ori %5, %6 : i1
    cf.cond_br %7, ^bb_empty, ^bb_body
  ^bb_empty:
    func.call @write_char(%arg0, %arg3) : (i32, i8) -> ()
    func.call @write_char(%arg0, %arg4) : (i32, i8) -> ()
    func.return
  ^bb_body:
    %8 = llvm.inttoptr %arg1 : i64 to !llvm.ptr
    %9 = llvm.inttoptr %arg2 : i64 to !llvm.ptr
    %10 = llvm.getelementptr %8[%0] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %11 = llvm.load %10 {alignment = 8 : i64} : !llvm.ptr -> i64
    %12 = arith.subi %arg6, %1 : i64
    func.call @write_char(%arg0, %arg3) : (i32, i8) -> ()
    %13 = arith.cmpi sgt, %11, %0 : i64
    scf.if %13 {
      %14 = arith.index_cast %11 : i64 to index
      %15 = arith.constant 0 : index
      %16 = arith.constant 1 : index
      scf.for %arg7 = %15 to %14 step %16 {
        %17 = arith.index_cast %arg7 : index to i64
        %18 = arith.cmpi ne, %17, %0 : i64
        scf.if %18 {
          func.call @write_char(%arg0, %3) : (i32, i8) -> ()
          func.call @write_char(%arg0, %4) : (i32, i8) -> ()
        }
        %19 = arith.muli %17, %2 : i64
        %20 = llvm.getelementptr %9[%19] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        func.call @print_boxed_slot_ptr(%arg0, %20, %12) : (i32, !llvm.ptr, i64) -> ()
      }
    }
    %21 = arith.cmpi eq, %11, %1 : i64
    %22 = arith.andi %arg5, %21 : i1
    scf.if %22 {
      func.call @write_char(%arg0, %3) : (i32, i8) -> ()
    }
    func.call @write_char(%arg0, %arg4) : (i32, i8) -> ()
    func.return
  }

  func.func private @print_boxed_dict(%arg0: i32, %arg1: i64, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64) {
    %0 = arith.constant 0 : i64
    %1 = arith.constant 1 : i64
    %2 = arith.constant 16 : i64
    %3 = arith.constant 123 : i8
    %4 = arith.constant 125 : i8
    %5 = arith.constant 44 : i8
    %6 = arith.constant 32 : i8
    %7 = arith.constant 58 : i8
    %8 = arith.cmpi eq, %arg1, %0 : i64
    %9 = arith.cmpi eq, %arg2, %0 : i64
    %10 = arith.cmpi eq, %arg3, %0 : i64
    %11 = arith.cmpi eq, %arg4, %0 : i64
    %12 = arith.ori %8, %9 : i1
    %13 = arith.ori %10, %11 : i1
    %14 = arith.ori %12, %13 : i1
    cf.cond_br %14, ^bb_empty, ^bb_body
  ^bb_empty:
    func.call @write_char(%arg0, %3) : (i32, i8) -> ()
    func.call @write_char(%arg0, %4) : (i32, i8) -> ()
    func.return
  ^bb_body:
    %15 = llvm.inttoptr %arg1 : i64 to !llvm.ptr
    %16 = llvm.inttoptr %arg2 : i64 to !llvm.ptr
    %17 = llvm.inttoptr %arg3 : i64 to !llvm.ptr
    %18 = llvm.inttoptr %arg4 : i64 to !llvm.ptr
    %19 = llvm.getelementptr %15[%1] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %20 = llvm.load %19 {alignment = 8 : i64} : !llvm.ptr -> i64
    %21 = arith.subi %arg5, %1 : i64
    func.call @write_char(%arg0, %3) : (i32, i8) -> ()
    %22 = arith.cmpi sgt, %20, %0 : i64
    scf.if %22 {
      %23 = arith.index_cast %20 : i64 to index
      %24 = arith.constant 0 : index
      %25 = arith.constant 1 : index
      %26 = scf.for %arg6 = %24 to %23 step %25 iter_args(%arg7 = %0) -> (i64) {
        %27 = arith.index_cast %arg6 : index to i64
        %28 = llvm.getelementptr %18[%27] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        %29 = llvm.load %28 {alignment = 8 : i64} : !llvm.ptr -> i64
        %30 = arith.cmpi ne, %29, %0 : i64
        %31 = scf.if %30 -> (i64) {
          %32 = arith.cmpi ne, %arg7, %0 : i64
          scf.if %32 {
            func.call @write_char(%arg0, %5) : (i32, i8) -> ()
            func.call @write_char(%arg0, %6) : (i32, i8) -> ()
          }
          %33 = arith.muli %27, %2 : i64
          %34 = llvm.getelementptr %16[%33] : (!llvm.ptr, i64) -> !llvm.ptr, i64
          %35 = llvm.getelementptr %17[%33] : (!llvm.ptr, i64) -> !llvm.ptr, i64
          func.call @print_boxed_slot_ptr(%arg0, %34, %21) : (i32, !llvm.ptr, i64) -> ()
          func.call @write_char(%arg0, %7) : (i32, i8) -> ()
          func.call @write_char(%arg0, %6) : (i32, i8) -> ()
          func.call @print_boxed_slot_ptr(%arg0, %35, %21) : (i32, !llvm.ptr, i64) -> ()
          %36 = arith.addi %arg7, %1 : i64
          scf.yield %36 : i64
        } else {
          scf.yield %arg7 : i64
        }
        scf.yield %31 : i64
      }
    }
    func.call @write_char(%arg0, %4) : (i32, i8) -> ()
    func.return
  }

  func.func private @print_boxed_slot_ptr(%arg0: i32, %arg1: !llvm.ptr, %arg2: i64) {
    %0 = arith.constant 0 : i64
    %1 = arith.constant 1 : i64
    %2 = arith.constant 2 : i64
    %4 = arith.constant 4 : i64
    %5 = arith.constant 5 : i64
    %6 = arith.constant 6 : i64
    %7 = arith.constant 7 : i64
    %8 = arith.constant 8 : i64
    %10 = arith.constant 10 : i64
    %11 = arith.constant 11 : i64
    %12 = arith.constant 12 : i64
    %open_list = arith.constant 91 : i8
    %close_list = arith.constant 93 : i8
    %open_tuple = arith.constant 40 : i8
    %close_tuple = arith.constant 41 : i8
    %false = arith.constant false
    %true = arith.constant true
    %13 = arith.cmpi sle, %arg2, %0 : i64
    cf.cond_br %13, ^bb_ellipsis, ^bb_probe
  ^bb_ellipsis:
    %14 = llvm.mlir.addressof @".repr_ellipsis" : !llvm.ptr
    func.call @write_cstr(%arg0, %14) : (i32, !llvm.ptr) -> ()
    func.return
  ^bb_probe:
    %15 = func.call @boxed_load_i64(%arg1, %0) : (!llvm.ptr, i64) -> i64
    %16 = arith.cmpi eq, %15, %0 : i64
    cf.cond_br %16, ^bb_none, ^bb_dispatch
  ^bb_none:
    %17 = llvm.mlir.addressof @".repr_none" : !llvm.ptr
    func.call @write_cstr(%arg0, %17) : (i32, !llvm.ptr) -> ()
    func.return
  ^bb_dispatch:
    %18 = func.call @boxed_load_i64(%arg1, %1) : (!llvm.ptr, i64) -> i64
    %19 = arith.cmpi eq, %18, %1 : i64
    cf.cond_br %19, ^bb_int, ^bb_check_float
  ^bb_check_float:
    %20 = arith.cmpi eq, %18, %2 : i64
    cf.cond_br %20, ^bb_float, ^bb_check_str
  ^bb_check_str:
    %21 = arith.cmpi eq, %18, %4 : i64
    cf.cond_br %21, ^bb_str, ^bb_check_list
  ^bb_check_list:
    %22 = arith.cmpi eq, %18, %10 : i64
    cf.cond_br %22, ^bb_list, ^bb_check_tuple
  ^bb_check_tuple:
    %23 = arith.cmpi eq, %18, %11 : i64
    cf.cond_br %23, ^bb_tuple, ^bb_check_dict
  ^bb_check_dict:
    %24 = arith.cmpi eq, %18, %12 : i64
    cf.cond_br %24, ^bb_dict, ^bb_fallback
  ^bb_int:
    %25 = func.call @boxed_load_i64(%arg1, %5) : (!llvm.ptr, i64) -> i64
    %26 = func.call @boxed_load_i64(%arg1, %6) : (!llvm.ptr, i64) -> i64
    func.call @print_boxed_int(%arg0, %25, %26) : (i32, i64, i64) -> ()
    func.return
  ^bb_float:
    %27 = func.call @boxed_load_i64(%arg1, %5) : (!llvm.ptr, i64) -> i64
    func.call @print_boxed_float(%arg0, %27) : (i32, i64) -> ()
    func.return
  ^bb_str:
    %28 = func.call @boxed_load_i64(%arg1, %5) : (!llvm.ptr, i64) -> i64
    %29 = func.call @boxed_load_i64(%arg1, %10) : (!llvm.ptr, i64) -> i64
    func.call @write_raw_bytes(%arg0, %28, %29) : (i32, i64, i64) -> ()
    func.return
  ^bb_list:
    %30 = func.call @boxed_load_i64(%arg1, %5) : (!llvm.ptr, i64) -> i64
    %31 = func.call @boxed_load_i64(%arg1, %6) : (!llvm.ptr, i64) -> i64
    func.call @print_boxed_sequence(%arg0, %30, %31, %open_list, %close_list, %false, %arg2) : (i32, i64, i64, i8, i8, i1, i64) -> ()
    func.return
  ^bb_tuple:
    %32 = func.call @boxed_load_i64(%arg1, %5) : (!llvm.ptr, i64) -> i64
    %33 = func.call @boxed_load_i64(%arg1, %6) : (!llvm.ptr, i64) -> i64
    func.call @print_boxed_sequence(%arg0, %32, %33, %open_tuple, %close_tuple, %true, %arg2) : (i32, i64, i64, i8, i8, i1, i64) -> ()
    func.return
  ^bb_dict:
    %34 = func.call @boxed_load_i64(%arg1, %5) : (!llvm.ptr, i64) -> i64
    %35 = func.call @boxed_load_i64(%arg1, %6) : (!llvm.ptr, i64) -> i64
    %36 = func.call @boxed_load_i64(%arg1, %7) : (!llvm.ptr, i64) -> i64
    %37 = func.call @boxed_load_i64(%arg1, %8) : (!llvm.ptr, i64) -> i64
    func.call @print_boxed_dict(%arg0, %34, %35, %36, %37, %arg2) : (i32, i64, i64, i64, i64, i64) -> ()
    func.return
  ^bb_fallback:
    %38 = llvm.mlir.addressof @".repr_object" : !llvm.ptr
    func.call @write_cstr(%arg0, %38) : (i32, !llvm.ptr) -> ()
    func.return
  }

  func.func @LyObject_PrintLine(%arg0: memref<16xi64>) {
    %0 = memref.extract_aligned_pointer_as_index %arg0 : memref<16xi64> -> index
    %1 = arith.index_cast %0 : index to i64
    %2 = llvm.inttoptr %1 : i64 to !llvm.ptr
    %3 = arith.constant 1 : i32
    %4 = arith.constant 8 : i64
    %5 = llvm.mlir.addressof @".newline" : !llvm.ptr
    %6 = arith.constant 1 : i64
    func.call @print_boxed_slot_ptr(%3, %2, %4) : (i32, !llvm.ptr, i64) -> ()
    func.call @write_len(%3, %5, %6) : (i32, !llvm.ptr, i64) -> ()
    func.return
  }

  func.func private @copy_cstr(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = arith.constant 1 : i64
    %2 = arith.constant 0 : i64
    %3 = arith.constant 0 : i8
    %4 = llvm.icmp "eq" %arg0, %0 : !llvm.ptr
    cf.cond_br %4, ^bb1, ^bb3
  ^bb1:  // pred: ^bb0
    %5 = llvm.call @malloc(%1) : (i64) -> !llvm.ptr
    %6 = llvm.icmp "eq" %5, %0 : !llvm.ptr
    cf.cond_br %6, ^bb7, ^bb2
  ^bb2:  // pred: ^bb1
    llvm.store %3, %5 {alignment = 1 : i64} : i8, !llvm.ptr
    func.return %5 : !llvm.ptr
  ^bb3:  // pred: ^bb0
    %7 = llvm.call @strlen(%arg0) : (!llvm.ptr) -> i64
    %8 = arith.addi %7, %1 : i64
    %9 = llvm.call @malloc(%8) : (i64) -> !llvm.ptr
    %10 = llvm.icmp "eq" %9, %0 : !llvm.ptr
    cf.cond_br %10, ^bb7, ^bb4
  ^bb4:  // pred: ^bb3
    %11 = arith.cmpi ne, %7, %2 : i64
    cf.cond_br %11, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    "llvm.intr.memcpy"(%9, %arg0, %7) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    cf.br ^bb6
  ^bb6:  // 2 preds: ^bb4, ^bb5
    %12 = llvm.getelementptr %9[%7] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %3, %12 {alignment = 1 : i64} : i8, !llvm.ptr
    func.return %9 : !llvm.ptr
  ^bb7:  // 2 preds: ^bb1, ^bb3
    llvm.call @rt_abort() : () -> ()
    llvm.unreachable
  }
  func.func private @copy_i8_memref(%arg0: !llvm.ptr, %arg1: i64, %arg2: i64, %arg3: i64) -> !llvm.ptr {
    %0 = arith.constant 0 : i64
    %1 = arith.constant 1 : i64
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = arith.constant true
    %4 = arith.constant 0 : i8
    %5 = arith.cmpi slt, %arg1, %0 : i64
    %6 = arith.cmpi slt, %arg2, %0 : i64
    %7 = arith.cmpi slt, %arg3, %1 : i64
    %8 = arith.ori %5, %6 : i1
    %9 = arith.ori %8, %7 : i1
    cf.cond_br %9, ^bb6, ^bb1
  ^bb1:  // pred: ^bb0
    %10 = arith.cmpi eq, %arg2, %0 : i64
    %11 = llvm.icmp "eq" %arg0, %2 : !llvm.ptr
    %12 = arith.xori %10, %3 : i1
    %13 = arith.andi %11, %12 : i1
    cf.cond_br %13, ^bb6, ^bb2
  ^bb2:  // pred: ^bb1
    %14 = arith.addi %arg2, %1 : i64
    %15 = llvm.call @malloc(%14) : (i64) -> !llvm.ptr
    %16 = llvm.icmp "eq" %15, %2 : !llvm.ptr
    cf.cond_br %16, ^bb6, ^bb3(%0 : i64)
  ^bb3(%17: i64):  // 2 preds: ^bb2, ^bb4
    %18 = arith.cmpi eq, %17, %arg2 : i64
    cf.cond_br %18, ^bb5, ^bb4
  ^bb4:  // pred: ^bb3
    %19 = arith.muli %17, %arg3 : i64
    %20 = arith.addi %arg1, %19 : i64
    %21 = llvm.getelementptr %arg0[%20] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %22 = llvm.load %21 {alignment = 1 : i64} : !llvm.ptr -> i8
    %23 = llvm.getelementptr %15[%17] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %22, %23 {alignment = 1 : i64} : i8, !llvm.ptr
    %24 = arith.addi %17, %1 : i64
    cf.br ^bb3(%24 : i64)
  ^bb5:  // pred: ^bb3
    %25 = llvm.getelementptr %15[%arg2] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %4, %25 {alignment = 1 : i64} : i8, !llvm.ptr
    func.return %15 : !llvm.ptr
  ^bb6:  // 3 preds: ^bb0, ^bb1, ^bb2
    llvm.call @rt_abort() : () -> ()
    llvm.unreachable
  }
  func.func private @print_bytes(%arg0: i32, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64) {
    %0 = arith.constant 0 : i64
    %1 = arith.constant 1 : i64
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = arith.cmpi slt, %arg5, %0 : i64
    cf.cond_br %3, ^bb8, ^bb1
  ^bb1:  // pred: ^bb0
    %4 = arith.cmpi eq, %arg5, %0 : i64
    cf.cond_br %4, ^bb7, ^bb2
  ^bb2:  // pred: ^bb1
    %5 = arith.cmpi slt, %arg2, %0 : i64
    %6 = arith.cmpi slt, %arg3, %0 : i64
    %7 = arith.cmpi slt, %arg4, %1 : i64
    %8 = llvm.icmp "eq" %arg1, %2 : !llvm.ptr
    %9 = arith.cmpi sgt, %arg5, %arg3 : i64
    %10 = arith.ori %5, %6 : i1
    %11 = arith.ori %7, %8 : i1
    %12 = arith.ori %10, %11 : i1
    %13 = arith.ori %12, %9 : i1
    cf.cond_br %13, ^bb8, ^bb3
  ^bb3:  // pred: ^bb2
    %14 = arith.cmpi eq, %arg4, %1 : i64
    cf.cond_br %14, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %15 = llvm.getelementptr %arg1[%arg2] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    func.call @write_len(%arg0, %15, %arg5) : (i32, !llvm.ptr, i64) -> ()
    cf.br ^bb7
  ^bb5:  // pred: ^bb3
    %16 = arith.constant 0 : index
    %17 = arith.constant 1 : index
    %18 = arith.index_cast %arg5 : i64 to index
    scf.for %19 = %16 to %18 step %17 {
      %20 = arith.index_cast %19 : index to i64
      %21 = arith.muli %20, %arg4 : i64
      %22 = arith.addi %arg2, %21 : i64
      %23 = llvm.getelementptr %arg1[%22] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %24 = llvm.load %23 {alignment = 1 : i64} : !llvm.ptr -> i8
      func.call @write_char(%arg0, %24) : (i32, i8) -> ()
    }
    cf.br ^bb7
  ^bb7:  // 3 preds: ^bb1, ^bb4, ^bb5
    func.return
  ^bb8:  // 2 preds: ^bb0, ^bb2
    llvm.call @rt_abort() : () -> ()
    llvm.unreachable
  }
  func.func @LyHost_Print(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64) {
    %0 = arith.constant 1 : i32
    func.call @print_bytes(%0, %arg1, %arg2, %arg3, %arg4, %arg5) : (i32, !llvm.ptr, i64, i64, i64, i64) -> ()
    func.return
  }
  func.func @LyHost_PrintLine(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64) {
    %0 = arith.constant 1 : i32
    %1 = llvm.mlir.addressof @".newline" : !llvm.ptr
    %2 = arith.constant 1 : i64
    func.call @print_bytes(%0, %arg1, %arg2, %arg3, %arg4, %arg5) : (i32, !llvm.ptr, i64, i64, i64, i64) -> ()
    func.call @write_len(%0, %1, %2) : (i32, !llvm.ptr, i64) -> ()
    func.return
  }
  func.func @LyFloat_Round(%arg0: f64, %arg1: i64) -> f64 {
    %0 = arith.constant 0.000000e+00 : f64
    %1 = arith.constant 0x7FF0000000000000 : f64
    %2 = arith.constant 308 : i64
    %3 = arith.constant -308 : i64
    %4 = arith.constant 0 : i64
    %5 = arith.constant 1.000000e+01 : f64
    %6 = arith.constant -1 : i64
    %7 = arith.cmpf uno, %arg0, %0 : f64
    %8 = llvm.intr.fabs(%arg0) : (f64) -> f64
    %9 = arith.cmpf oeq, %8, %1 : f64
    %10 = arith.ori %7, %9 : i1
    cf.cond_br %10, ^bb7, ^bb1
  ^bb1:  // pred: ^bb0
    %11 = arith.cmpi sgt, %arg1, %2 : i64
    %12 = arith.cmpi slt, %arg1, %3 : i64
    %13 = arith.ori %11, %12 : i1
    cf.cond_br %13, ^bb6, ^bb2
  ^bb2:  // pred: ^bb1
    %14 = arith.cmpi slt, %arg1, %4 : i64
    %15 = arith.subi %4, %arg1 : i64
    %16 = arith.select %14, %15, %arg1 : i64
    %17 = arith.uitofp %16 : i64 to f64
    %18 = llvm.call @pow(%5, %17) : (f64, f64) -> f64
    %19 = arith.cmpf uno, %18, %0 : f64
    %20 = llvm.intr.fabs(%18) : (f64) -> f64
    %21 = arith.cmpf oeq, %20, %1 : f64
    %22 = arith.cmpf oeq, %18, %0 : f64
    %23 = arith.ori %19, %21 : i1
    %24 = arith.ori %23, %22 : i1
    cf.cond_br %24, ^bb7, ^bb3
  ^bb3:  // pred: ^bb2
    %25 = arith.cmpi sgt, %arg1, %6 : i64
    cf.cond_br %25, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %26 = arith.mulf %arg0, %18 : f64
    %27 = llvm.call @nearbyint(%26) : (f64) -> f64
    %28 = arith.divf %27, %18 : f64
    func.return %28 : f64
  ^bb5:  // pred: ^bb3
    %29 = arith.divf %arg0, %18 : f64
    %30 = llvm.call @nearbyint(%29) : (f64) -> f64
    %31 = arith.mulf %30, %18 : f64
    func.return %31 : f64
  ^bb6:  // pred: ^bb1
    func.return %arg0 : f64
  ^bb7:  // 2 preds: ^bb0, ^bb2
    llvm.call @rt_abort() : () -> ()
    llvm.unreachable
  }
  func.func @LyFloat_RoundToI64(%arg0: f64) -> i64 {
    %0 = arith.constant 0.000000e+00 : f64
    %1 = arith.constant 0x7FF0000000000000 : f64
    %2 = arith.constant -9.2233720368547758E+18 : f64
    %3 = arith.constant 9.2233720368547758E+18 : f64
    %4 = arith.cmpf uno, %arg0, %0 : f64
    %5 = llvm.intr.fabs(%arg0) : (f64) -> f64
    %6 = arith.cmpf oeq, %5, %1 : f64
    %7 = arith.ori %4, %6 : i1
    cf.cond_br %7, ^bb3, ^bb1
  ^bb1:  // pred: ^bb0
    %8 = llvm.call @nearbyint(%arg0) : (f64) -> f64
    %9 = arith.cmpf olt, %8, %2 : f64
    %10 = arith.cmpf oge, %8, %3 : f64
    %11 = arith.ori %9, %10 : i1
    cf.cond_br %11, ^bb3, ^bb2
  ^bb2:  // pred: ^bb1
    %12 = arith.fptosi %8 : f64 to i64
    func.return %12 : i64
  ^bb3:  // 2 preds: ^bb0, ^bb1
    llvm.call @rt_abort() : () -> ()
    llvm.unreachable
  }
  func.func @LyInt_Round(%arg0: i64, %arg1: i64) -> i64 {
    %0 = arith.constant -1 : i64
    %1 = arith.constant 0 : i64
    %2 = arith.constant 19 : i64
    %3 = arith.constant 1 : i64
    %4 = arith.constant 1844674407370955161 : i64
    %5 = arith.constant 10 : i64
    %6 = arith.constant -9223372036854775808 : i64
    %7 = arith.constant 9223372036854775807 : i64
    %8 = arith.cmpi sgt, %arg1, %0 : i64
    cf.cond_br %8, ^bb13, ^bb1
  ^bb1:  // pred: ^bb0
    %9 = arith.subi %1, %arg1 : i64
    %10 = arith.cmpi ugt, %9, %2 : i64
    cf.cond_br %10, ^bb12, ^bb2(%1, %3 : i64, i64)
  ^bb2(%11: i64, %12: i64):  // 2 preds: ^bb1, ^bb4
    %13 = arith.cmpi eq, %11, %9 : i64
    cf.cond_br %13, ^bb5, ^bb3
  ^bb3:  // pred: ^bb2
    %14 = arith.cmpi ugt, %12, %4 : i64
    cf.cond_br %14, ^bb12, ^bb4
  ^bb4:  // pred: ^bb3
    %15 = arith.muli %12, %5 : i64
    %16 = arith.addi %11, %3 : i64
    cf.br ^bb2(%16, %15 : i64, i64)
  ^bb5:  // pred: ^bb2
    %17 = arith.cmpi sgt, %arg0, %0 : i64
    %18 = arith.addi %arg0, %3 : i64
    %19 = arith.subi %1, %18 : i64
    %20 = arith.addi %19, %3 : i64
    %21 = arith.select %17, %arg0, %20 : i64
    %22 = arith.divui %21, %12 : i64
    %23 = arith.remui %21, %12 : i64
    %24 = arith.subi %12, %23 : i64
    %25 = arith.cmpi ugt, %23, %24 : i64
    %26 = arith.cmpi eq, %23, %24 : i64
    %27 = arith.andi %22, %3 : i64
    %28 = arith.cmpi ne, %27, %1 : i64
    %29 = arith.andi %26, %28 : i1
    %30 = arith.ori %25, %29 : i1
    %31 = arith.extui %30 : i1 to i64
    %32 = arith.addi %22, %31 : i64
    %33 = arith.muli %32, %12 : i64
    cf.cond_br %17, ^bb6, ^bb8
  ^bb6:  // pred: ^bb5
    %34 = arith.cmpi ugt, %33, %7 : i64
    cf.cond_br %34, ^bb14, ^bb7
  ^bb7:  // pred: ^bb6
    func.return %33 : i64
  ^bb8:  // pred: ^bb5
    %35 = arith.cmpi eq, %33, %6 : i64
    cf.cond_br %35, ^bb11, ^bb9
  ^bb9:  // pred: ^bb8
    %36 = arith.cmpi ugt, %33, %7 : i64
    cf.cond_br %36, ^bb14, ^bb10
  ^bb10:  // pred: ^bb9
    %37 = arith.subi %1, %33 : i64
    func.return %37 : i64
  ^bb11:  // pred: ^bb8
    func.return %6 : i64
  ^bb12:  // 2 preds: ^bb1, ^bb3
    func.return %1 : i64
  ^bb13:  // pred: ^bb0
    func.return %arg0 : i64
  ^bb14:  // 2 preds: ^bb6, ^bb9
    llvm.call @rt_abort() : () -> ()
    llvm.unreachable
  }
  func.func private @frame_at(%arg0: i64) -> !llvm.ptr {
    %0 = llvm.mlir.addressof @g_traceback_stack : !llvm.ptr
    %1 = arith.constant 0 : i64
    %2 = llvm.getelementptr inbounds %0[%1, %arg0] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !TracebackStack
    func.return %2 : !llvm.ptr
  }
  func.func private @free_frame(%arg0: !llvm.ptr) {
    %0 = arith.constant 0 : i32
    %1 = arith.constant 1 : i32
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = llvm.getelementptr inbounds %arg0[%0, 0] : (!llvm.ptr, i32) -> !llvm.ptr, !TracebackFrame
    %4 = llvm.getelementptr inbounds %arg0[%0, 1] : (!llvm.ptr, i32) -> !llvm.ptr, !TracebackFrame
    %5 = llvm.load %3 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %6 = llvm.load %4 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %7 = llvm.icmp "ne" %5, %2 : !llvm.ptr
    cf.cond_br %7, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    llvm.call @free(%5) : (!llvm.ptr) -> ()
    cf.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    %8 = llvm.icmp "ne" %6, %2 : !llvm.ptr
    cf.cond_br %8, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    llvm.call @free(%6) : (!llvm.ptr) -> ()
    cf.br ^bb4
  ^bb4:  // 2 preds: ^bb2, ^bb3
    func.return
  }
  func.func @LyTraceback_Push(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i32, %arg11: i32) {
    %0 = llvm.mlir.addressof @g_traceback_size : !llvm.ptr
    %1 = arith.constant 1024 : i64
    %2 = arith.constant 0 : i32
    %3 = arith.constant 1 : i32
    %4 = arith.constant 2 : i32
    %5 = arith.constant 3 : i32
    %6 = arith.constant 4 : i32
    %7 = arith.constant 5 : i32
    %8 = arith.constant 6 : i32
    %9 = arith.constant 1 : i64
    %10 = llvm.load %0 {alignment = 8 : i64} : !llvm.ptr -> i64
    %11 = arith.cmpi uge, %10, %1 : i64
    cf.cond_br %11, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    %12 = func.call @copy_i8_memref(%arg1, %arg2, %arg3, %arg4) : (!llvm.ptr, i64, i64, i64) -> !llvm.ptr
    %13 = func.call @copy_i8_memref(%arg6, %arg7, %arg8, %arg9) : (!llvm.ptr, i64, i64, i64) -> !llvm.ptr
    %14 = func.call @frame_at(%10) : (i64) -> !llvm.ptr
    %15 = llvm.getelementptr inbounds %14[%2, 0] : (!llvm.ptr, i32) -> !llvm.ptr, !TracebackFrame
    %16 = llvm.getelementptr inbounds %14[%2, 1] : (!llvm.ptr, i32) -> !llvm.ptr, !TracebackFrame
    %17 = llvm.getelementptr inbounds %14[%2, 2] : (!llvm.ptr, i32) -> !llvm.ptr, !TracebackFrame
    %18 = llvm.getelementptr inbounds %14[%2, 3] : (!llvm.ptr, i32) -> !llvm.ptr, !TracebackFrame
    %19 = llvm.getelementptr inbounds %14[%2, 4] : (!llvm.ptr, i32) -> !llvm.ptr, !TracebackFrame
    %20 = llvm.getelementptr inbounds %14[%2, 5] : (!llvm.ptr, i32) -> !llvm.ptr, !TracebackFrame
    %21 = llvm.getelementptr inbounds %14[%2, 6] : (!llvm.ptr, i32) -> !llvm.ptr, !TracebackFrame
    llvm.store %12, %15 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %13, %16 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %arg10, %17 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.store %arg11, %18 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.store %2, %19 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.store %2, %20 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.store %2, %21 {alignment = 4 : i64} : i32, !llvm.ptr
    %22 = arith.addi %10, %9 : i64
    llvm.store %22, %0 {alignment = 8 : i64} : i64, !llvm.ptr
    func.return
  ^bb2:  // pred: ^bb0
    llvm.call @rt_abort() : () -> ()
    llvm.unreachable
  }
  func.func @LyTraceback_PushCString(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i32, %arg3: i32) {
    %0 = arith.constant 0 : i32
    func.call @LyTraceback_PushCStringRange(%arg0, %arg1, %arg2, %arg3, %arg2, %0) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    func.return
  }
  func.func @LyTraceback_PushCStringRange(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
    %0 = llvm.mlir.addressof @g_traceback_size : !llvm.ptr
    %1 = arith.constant 1024 : i64
    %2 = arith.constant 0 : i32
    %3 = arith.constant 1 : i32
    %4 = arith.constant 2 : i32
    %5 = arith.constant 3 : i32
    %6 = arith.constant 4 : i32
    %7 = arith.constant 5 : i32
    %8 = arith.constant 6 : i32
    %9 = arith.constant 1 : i64
    %10 = llvm.load %0 {alignment = 8 : i64} : !llvm.ptr -> i64
    %11 = arith.cmpi uge, %10, %1 : i64
    cf.cond_br %11, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    %12 = func.call @copy_cstr(%arg0) : (!llvm.ptr) -> !llvm.ptr
    %13 = func.call @copy_cstr(%arg1) : (!llvm.ptr) -> !llvm.ptr
    %14 = func.call @frame_at(%10) : (i64) -> !llvm.ptr
    %15 = llvm.getelementptr inbounds %14[%2, 0] : (!llvm.ptr, i32) -> !llvm.ptr, !TracebackFrame
    %16 = llvm.getelementptr inbounds %14[%2, 1] : (!llvm.ptr, i32) -> !llvm.ptr, !TracebackFrame
    %17 = llvm.getelementptr inbounds %14[%2, 2] : (!llvm.ptr, i32) -> !llvm.ptr, !TracebackFrame
    %18 = llvm.getelementptr inbounds %14[%2, 3] : (!llvm.ptr, i32) -> !llvm.ptr, !TracebackFrame
    %19 = llvm.getelementptr inbounds %14[%2, 4] : (!llvm.ptr, i32) -> !llvm.ptr, !TracebackFrame
    %20 = llvm.getelementptr inbounds %14[%2, 5] : (!llvm.ptr, i32) -> !llvm.ptr, !TracebackFrame
    %21 = llvm.getelementptr inbounds %14[%2, 6] : (!llvm.ptr, i32) -> !llvm.ptr, !TracebackFrame
    llvm.store %12, %15 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %13, %16 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %arg2, %17 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.store %arg3, %18 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.store %arg4, %19 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.store %arg5, %20 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.store %3, %21 {alignment = 4 : i64} : i32, !llvm.ptr
    %22 = arith.addi %10, %9 : i64
    llvm.store %22, %0 {alignment = 8 : i64} : i64, !llvm.ptr
    func.return
  ^bb2:  // pred: ^bb0
    llvm.call @rt_abort() : () -> ()
    llvm.unreachable
  }
  func.func @LyTraceback_Pop() {
    %0 = llvm.mlir.addressof @g_traceback_size : !llvm.ptr
    %1 = arith.constant 0 : i64
    %2 = arith.constant 1 : i64
    %3 = llvm.load %0 {alignment = 8 : i64} : !llvm.ptr -> i64
    %4 = arith.cmpi eq, %3, %1 : i64
    cf.cond_br %4, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    %5 = arith.subi %3, %2 : i64
    %6 = func.call @frame_at(%5) : (i64) -> !llvm.ptr
    func.call @free_frame(%6) : (!llvm.ptr) -> ()
    llvm.store %5, %0 {alignment = 8 : i64} : i64, !llvm.ptr
    cf.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    func.return
  }
  func.func @LyTraceback_Clear() {
    %0 = llvm.mlir.addressof @g_traceback_size : !llvm.ptr
    %1 = arith.constant 0 : i64
    cf.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb2
    %2 = llvm.load %0 {alignment = 8 : i64} : !llvm.ptr -> i64
    %3 = arith.cmpi eq, %2, %1 : i64
    cf.cond_br %3, ^bb3, ^bb2
  ^bb2:  // pred: ^bb1
    func.call @LyTraceback_Pop() : () -> ()
    cf.br ^bb1
  ^bb3:  // pred: ^bb1
    func.return
  }
  func.func private @read_source_line(%arg0: !llvm.ptr, %arg1: i32) -> !llvm.ptr {
    %0 = arith.constant 512 : i64
    %1 = llvm.mlir.zero : !llvm.ptr
    %2 = arith.constant 0 : i8
    %3 = arith.constant 1 : i32
    %4 = llvm.mlir.addressof @".read_mode" : !llvm.ptr
    %5 = arith.constant 512 : i32
    %6 = arith.constant 0 : i64
    %7 = arith.constant 1 : i64
    %8 = arith.constant 10 : i8
    %9 = arith.constant 13 : i8
    %10 = llvm.call @malloc(%0) : (i64) -> !llvm.ptr
    %11 = llvm.icmp "eq" %10, %1 : !llvm.ptr
    cf.cond_br %11, ^bb13, ^bb1
  ^bb1:  // pred: ^bb0
    llvm.store %2, %10 {alignment = 1 : i64} : i8, !llvm.ptr
    %12 = llvm.icmp "eq" %arg0, %1 : !llvm.ptr
    %13 = arith.cmpi slt, %arg1, %3 : i32
    %14 = arith.ori %12, %13 : i1
    cf.cond_br %14, ^bb12, ^bb2
  ^bb2:  // pred: ^bb1
    %15 = llvm.call @fopen(%arg0, %4) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %16 = llvm.icmp "eq" %15, %1 : !llvm.ptr
    cf.cond_br %16, ^bb12, ^bb3(%3 : i32)
  ^bb3(%17: i32):  // 2 preds: ^bb2, ^bb5
    %18 = llvm.call @fgets(%10, %5, %15) : (!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
    %19 = llvm.icmp "eq" %18, %1 : !llvm.ptr
    cf.cond_br %19, ^bb6, ^bb4
  ^bb4:  // pred: ^bb3
    %20 = arith.cmpi eq, %17, %arg1 : i32
    cf.cond_br %20, ^bb7, ^bb5
  ^bb5:  // pred: ^bb4
    %21 = arith.addi %17, %3 : i32
    cf.br ^bb3(%21 : i32)
  ^bb6:  // pred: ^bb3
    llvm.store %2, %10 {alignment = 1 : i64} : i8, !llvm.ptr
    %22 = llvm.call @fclose(%15) : (!llvm.ptr) -> i32
    cf.br ^bb12
  ^bb7:  // pred: ^bb4
    %23 = llvm.call @fclose(%15) : (!llvm.ptr) -> i32
    cf.br ^bb8
  ^bb8:  // pred: ^bb7
    %24 = llvm.call @strlen(%10) : (!llvm.ptr) -> i64
    cf.br ^bb9(%24 : i64)
  ^bb9(%25: i64):  // 2 preds: ^bb8, ^bb11
    %26 = arith.cmpi ne, %25, %6 : i64
    cf.cond_br %26, ^bb10, ^bb12
  ^bb10:  // pred: ^bb9
    %27 = arith.subi %25, %7 : i64
    %28 = llvm.getelementptr %10[%27] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %29 = llvm.load %28 {alignment = 1 : i64} : !llvm.ptr -> i8
    %30 = arith.cmpi eq, %29, %8 : i8
    %31 = arith.cmpi eq, %29, %9 : i8
    %32 = arith.ori %30, %31 : i1
    cf.cond_br %32, ^bb11, ^bb12
  ^bb11:  // pred: ^bb10
    llvm.store %2, %28 {alignment = 1 : i64} : i8, !llvm.ptr
    %33 = arith.subi %25, %7 : i64
    cf.br ^bb9(%33 : i64)
  ^bb12:  // 5 preds: ^bb1, ^bb2, ^bb6, ^bb9, ^bb10
    func.return %10 : !llvm.ptr
  ^bb13:  // pred: ^bb0
    llvm.call @rt_abort() : () -> ()
    llvm.unreachable
  }
  func.func private @exception_class_name(%arg0: i64) -> !llvm.ptr {
    %0 = llvm.mlir.addressof @".class.CancelledError" : !llvm.ptr
    %1 = llvm.mlir.addressof @".class.ZeroDivisionError" : !llvm.ptr
    %2 = llvm.mlir.addressof @".class.LookupError" : !llvm.ptr
    %3 = llvm.mlir.addressof @".class.ArithmeticError" : !llvm.ptr
    %4 = llvm.mlir.addressof @".class.StopAsyncIteration" : !llvm.ptr
    %5 = llvm.mlir.addressof @".class.StopIteration" : !llvm.ptr
    %6 = llvm.mlir.addressof @".class.AssertionError" : !llvm.ptr
    %7 = llvm.mlir.addressof @".class.IndexError" : !llvm.ptr
    %8 = llvm.mlir.addressof @".class.KeyError" : !llvm.ptr
    %9 = llvm.mlir.addressof @".class.ValueError" : !llvm.ptr
    %10 = llvm.mlir.addressof @".class.TypeError" : !llvm.ptr
    %11 = llvm.mlir.addressof @".class.RuntimeError" : !llvm.ptr
    %12 = llvm.mlir.addressof @".class.Exception" : !llvm.ptr
    %13 = llvm.mlir.addressof @".class.BaseException" : !llvm.ptr
    cf.switch %arg0 : i64, [
      default: ^bb15,
      5: ^bb1,
      50: ^bb2,
      51: ^bb3,
      52: ^bb4,
      53: ^bb5,
      54: ^bb6,
      55: ^bb7,
      56: ^bb8,
      57: ^bb9,
      58: ^bb10,
      59: ^bb11,
      60: ^bb12,
      61: ^bb13,
      62: ^bb14
    ]
  ^bb1:  // pred: ^bb0
    func.return %13 : !llvm.ptr
  ^bb2:  // pred: ^bb0
    func.return %12 : !llvm.ptr
  ^bb3:  // pred: ^bb0
    func.return %11 : !llvm.ptr
  ^bb4:  // pred: ^bb0
    func.return %10 : !llvm.ptr
  ^bb5:  // pred: ^bb0
    func.return %9 : !llvm.ptr
  ^bb6:  // pred: ^bb0
    func.return %8 : !llvm.ptr
  ^bb7:  // pred: ^bb0
    func.return %7 : !llvm.ptr
  ^bb8:  // pred: ^bb0
    func.return %6 : !llvm.ptr
  ^bb9:  // pred: ^bb0
    func.return %5 : !llvm.ptr
  ^bb10:  // pred: ^bb0
    func.return %4 : !llvm.ptr
  ^bb11:  // pred: ^bb0
    func.return %3 : !llvm.ptr
  ^bb12:  // pred: ^bb0
    func.return %2 : !llvm.ptr
  ^bb13:  // pred: ^bb0
    func.return %1 : !llvm.ptr
  ^bb14:  // pred: ^bb0
    func.return %0 : !llvm.ptr
  ^bb15:  // pred: ^bb0
    func.return %12 : !llvm.ptr
  }
  func.func private @leading_whitespace(%arg0: !llvm.ptr) -> i64 {
    %0 = arith.constant 0 : i64
    %1 = arith.constant 32 : i8
    %2 = arith.constant 9 : i8
    %3 = arith.constant 1 : i64
    %4 = llvm.call @strlen(%arg0) : (!llvm.ptr) -> i64
    cf.br ^bb1(%0 : i64)
  ^bb1(%5: i64):  // 2 preds: ^bb0, ^bb3
    %6 = arith.cmpi eq, %5, %4 : i64
    cf.cond_br %6, ^bb4, ^bb2
  ^bb2:  // pred: ^bb1
    %7 = llvm.getelementptr %arg0[%5] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %8 = llvm.load %7 {alignment = 1 : i64} : !llvm.ptr -> i8
    %9 = arith.cmpi eq, %8, %1 : i8
    %10 = arith.cmpi eq, %8, %2 : i8
    %11 = arith.ori %9, %10 : i1
    cf.cond_br %11, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %12 = arith.addi %5, %3 : i64
    cf.br ^bb1(%12 : i64)
  ^bb4:  // 2 preds: ^bb1, ^bb2
    func.return %5 : i64
  }
  func.func private @print_marker(%arg0: !llvm.ptr, %arg1: i32, %arg2: i32) {
    %0 = arith.constant 0 : i64
    %1 = arith.constant 0 : i32
    %2 = arith.constant 32 : i8
    %3 = arith.constant 9 : i8
    %4 = arith.constant 1 : i64
    %5 = arith.constant 2 : i32
    %6 = llvm.mlir.addressof @".indent" : !llvm.ptr
    %7 = arith.constant 2 : i64
    %8 = arith.constant 126 : i8
    %9 = arith.constant 94 : i8
    %10 = llvm.mlir.addressof @".newline" : !llvm.ptr
    %11 = llvm.call @strlen(%arg0) : (!llvm.ptr) -> i64
    %12 = arith.cmpi eq, %11, %0 : i64
    cf.cond_br %12, ^bb19, ^bb1
  ^bb1:  // pred: ^bb0
    %13 = arith.cmpi sgt, %arg1, %1 : i32
    %14 = arith.extsi %arg1 : i32 to i64
    %15 = arith.cmpi slt, %14, %11 : i64
    %16 = arith.andi %13, %15 : i1
    cf.cond_br %16, ^bb2, ^bb3(%0 : i64)
  ^bb2:  // pred: ^bb1
    cf.br ^bb6(%14 : i64)
  ^bb3(%17: i64):  // 2 preds: ^bb1, ^bb5
    %18 = arith.cmpi eq, %17, %11 : i64
    cf.cond_br %18, ^bb6(%17 : i64), ^bb4
  ^bb4:  // pred: ^bb3
    %19 = llvm.getelementptr %arg0[%17] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %20 = llvm.load %19 {alignment = 1 : i64} : !llvm.ptr -> i8
    %21 = arith.cmpi eq, %20, %2 : i8
    %22 = arith.cmpi eq, %20, %3 : i8
    %23 = arith.ori %21, %22 : i1
    cf.cond_br %23, ^bb5, ^bb6(%17 : i64)
  ^bb5:  // pred: ^bb4
    %24 = arith.addi %17, %4 : i64
    cf.br ^bb3(%24 : i64)
  ^bb6(%25: i64):  // 3 preds: ^bb2, ^bb3, ^bb4
    %26 = arith.cmpi uge, %25, %11 : i64
    cf.cond_br %26, ^bb19, ^bb7
  ^bb7:  // pred: ^bb6
    %27 = arith.cmpi sgt, %arg2, %arg1 : i32
    %28 = arith.cmpi sgt, %arg2, %1 : i32
    %29 = arith.andi %27, %28 : i1
    %30 = arith.extsi %arg2 : i32 to i64
    %31 = arith.cmpi sgt, %30, %11 : i64
    %32 = arith.select %31, %11, %30 : i64
    %33 = arith.select %29, %32, %11 : i64
    %34 = arith.cmpi ule, %33, %25 : i64
    %35 = arith.addi %25, %4 : i64
    %36 = arith.cmpi sgt, %35, %11 : i64
    %37 = arith.select %36, %11, %35 : i64
    %38 = arith.select %34, %37, %33 : i64
    %39 = arith.subi %38, %25 : i64
    func.call @write_cstr(%5, %6) : (i32, !llvm.ptr) -> ()
    cf.br ^bb8(%0 : i64)
  ^bb8(%40: i64):  // 2 preds: ^bb7, ^bb10
    %41 = arith.cmpi eq, %40, %25 : i64
    cf.cond_br %41, ^bb11, ^bb9
  ^bb9:  // pred: ^bb8
    %42 = llvm.getelementptr %arg0[%40] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %43 = llvm.load %42 {alignment = 1 : i64} : !llvm.ptr -> i8
    %44 = arith.cmpi eq, %43, %3 : i8
    %45 = arith.select %44, %3, %2 : i8
    func.call @write_char(%5, %45) : (i32, i8) -> ()
    cf.br ^bb10
  ^bb10:  // pred: ^bb9
    %46 = arith.addi %40, %4 : i64
    cf.br ^bb8(%46 : i64)
  ^bb11:  // pred: ^bb8
    %47 = arith.cmpi ule, %39, %7 : i64
    cf.cond_br %47, ^bb12(%0 : i64), ^bb14
  ^bb12(%48: i64):  // 2 preds: ^bb11, ^bb13
    %49 = arith.cmpi eq, %48, %39 : i64
    cf.cond_br %49, ^bb18, ^bb13
  ^bb13:  // pred: ^bb12
    func.call @write_char(%5, %9) : (i32, i8) -> ()
    %50 = arith.addi %48, %4 : i64
    cf.br ^bb12(%50 : i64)
  ^bb14:  // pred: ^bb11
    %51 = arith.subi %39, %7 : i64
    cf.br ^bb15(%0 : i64)
  ^bb15(%52: i64):  // 2 preds: ^bb14, ^bb16
    %53 = arith.cmpi eq, %52, %51 : i64
    cf.cond_br %53, ^bb17, ^bb16
  ^bb16:  // pred: ^bb15
    func.call @write_char(%5, %8) : (i32, i8) -> ()
    %54 = arith.addi %52, %4 : i64
    cf.br ^bb15(%54 : i64)
  ^bb17:  // pred: ^bb15
    func.call @write_char(%5, %9) : (i32, i8) -> ()
    func.call @write_char(%5, %9) : (i32, i8) -> ()
    cf.br ^bb18
  ^bb18:  // 2 preds: ^bb12, ^bb17
    func.call @write_len(%5, %10, %4) : (i32, !llvm.ptr, i64) -> ()
    cf.br ^bb19
  ^bb19:  // 3 preds: ^bb0, ^bb6, ^bb18
    func.return
  }
  func.func private @print_trace_frame(%arg0: !llvm.ptr) {
    %0 = arith.constant 1 : i32
    %1 = arith.constant 0 : i64
    %2 = arith.constant 0 : i32
    %3 = arith.constant 2 : i32
    %4 = arith.constant 3 : i32
    %5 = arith.constant 5 : i32
    %6 = arith.constant 6 : i32
    %7 = arith.constant 1024 : i64
    %8 = llvm.mlir.addressof @".fmt_frame" : !llvm.ptr
    %9 = llvm.mlir.addressof @".indent" : !llvm.ptr
    %10 = llvm.mlir.addressof @".newline" : !llvm.ptr
    %11 = arith.constant 1 : i64
    %12 = llvm.alloca %0 x !llvm.array<1024 x i8> {alignment = 1 : i64} : (i32) -> !llvm.ptr
    %13 = llvm.getelementptr inbounds %12[%1, %1] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<1024 x i8>
    %14 = llvm.getelementptr inbounds %arg0[%2, 0] : (!llvm.ptr, i32) -> !llvm.ptr, !TracebackFrame
    %15 = llvm.getelementptr inbounds %arg0[%2, 1] : (!llvm.ptr, i32) -> !llvm.ptr, !TracebackFrame
    %16 = llvm.getelementptr inbounds %arg0[%2, 2] : (!llvm.ptr, i32) -> !llvm.ptr, !TracebackFrame
    %17 = llvm.getelementptr inbounds %arg0[%2, 3] : (!llvm.ptr, i32) -> !llvm.ptr, !TracebackFrame
    %18 = llvm.getelementptr inbounds %arg0[%2, 5] : (!llvm.ptr, i32) -> !llvm.ptr, !TracebackFrame
    %19 = llvm.getelementptr inbounds %arg0[%2, 6] : (!llvm.ptr, i32) -> !llvm.ptr, !TracebackFrame
    %20 = llvm.load %14 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %21 = llvm.load %15 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %22 = llvm.load %16 {alignment = 4 : i64} : !llvm.ptr -> i32
    %23 = llvm.load %17 {alignment = 4 : i64} : !llvm.ptr -> i32
    %24 = llvm.load %18 {alignment = 4 : i64} : !llvm.ptr -> i32
    %25 = llvm.load %19 {alignment = 4 : i64} : !llvm.ptr -> i32
    %26 = llvm.call @snprintf(%13, %7, %8, %20, %22, %21) vararg(!llvm.func<i32 (ptr, i64, ptr, ...)>) : (!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
    func.call @write_buffered(%3, %13, %26) : (i32, !llvm.ptr, i32) -> ()
    %27 = func.call @read_source_line(%20, %22) : (!llvm.ptr, i32) -> !llvm.ptr
    %28 = llvm.call @strlen(%27) : (!llvm.ptr) -> i64
    %29 = arith.cmpi ne, %28, %1 : i64
    cf.cond_br %29, ^bb1, ^bb3
  ^bb1:  // pred: ^bb0
    %30 = func.call @leading_whitespace(%27) : (!llvm.ptr) -> i64
    %31 = llvm.getelementptr %27[%30] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %32 = llvm.call @strlen(%31) : (!llvm.ptr) -> i64
    func.call @write_cstr(%3, %9) : (i32, !llvm.ptr) -> ()
    func.call @write_len(%3, %31, %32) : (i32, !llvm.ptr, i64) -> ()
    func.call @write_len(%3, %10, %11) : (i32, !llvm.ptr, i64) -> ()
    %33 = arith.cmpi ne, %25, %2 : i32
    cf.cond_br %33, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %34 = arith.trunci %30 : i64 to i32
    %35 = arith.cmpi sgt, %23, %34 : i32
    %36 = arith.subi %23, %34 : i32
    %37 = arith.select %35, %36, %2 : i32
    %38 = arith.cmpi sgt, %24, %34 : i32
    %39 = arith.subi %24, %34 : i32
    %40 = arith.select %38, %39, %2 : i32
    func.call @print_marker(%31, %37, %40) : (!llvm.ptr, i32, i32) -> ()
    cf.br ^bb3
  ^bb3:  // 3 preds: ^bb0, ^bb1, ^bb2
    llvm.call @free(%27) : (!llvm.ptr) -> ()
    func.return
  }
  func.func private @print_exception_summary(%arg0: i64, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64) {
    %0 = arith.constant 1 : i32
    %1 = arith.constant 0 : i64
    %2 = arith.constant 1 : i64
    %3 = llvm.mlir.zero : !llvm.ptr
    %4 = arith.constant 1024 : i64
    %5 = llvm.mlir.addressof @".fmt_message" : !llvm.ptr
    %6 = arith.constant 2 : i32
    %7 = llvm.mlir.addressof @".fmt_unknown" : !llvm.ptr
    %8 = llvm.mlir.addressof @".fmt_class" : !llvm.ptr
    %9 = llvm.mlir.addressof @".fmt_invalid" : !llvm.ptr
    %10 = llvm.alloca %0 x !llvm.array<1024 x i8> {alignment = 1 : i64} : (i32) -> !llvm.ptr
    %11 = llvm.getelementptr inbounds %10[%1, %1] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<1024 x i8>
    %12 = func.call @exception_class_name(%arg0) : (i64) -> !llvm.ptr
    %13 = arith.cmpi slt, %arg2, %1 : i64
    %14 = arith.cmpi slt, %arg3, %1 : i64
    %15 = arith.cmpi slt, %arg4, %2 : i64
    %16 = arith.ori %13, %14 : i1
    %17 = arith.ori %16, %15 : i1
    cf.cond_br %17, ^bb5, ^bb1
  ^bb1:  // pred: ^bb0
    %18 = arith.cmpi eq, %arg3, %1 : i64
    cf.cond_br %18, ^bb4, ^bb2
  ^bb2:  // pred: ^bb1
    %19 = llvm.icmp "eq" %arg1, %3 : !llvm.ptr
    cf.cond_br %19, ^bb6, ^bb3
  ^bb3:  // pred: ^bb2
    %20 = func.call @copy_i8_memref(%arg1, %arg2, %arg3, %arg4) : (!llvm.ptr, i64, i64, i64) -> !llvm.ptr
    %21 = llvm.call @snprintf(%11, %4, %5, %12, %20) vararg(!llvm.func<i32 (ptr, i64, ptr, ...)>) : (!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    func.call @write_buffered(%6, %11, %21) : (i32, !llvm.ptr, i32) -> ()
    llvm.call @free(%20) : (!llvm.ptr) -> ()
    func.return
  ^bb4:  // pred: ^bb1
    %22 = llvm.call @snprintf(%11, %4, %8, %12) vararg(!llvm.func<i32 (ptr, i64, ptr, ...)>) : (!llvm.ptr, i64, !llvm.ptr, !llvm.ptr) -> i32
    func.call @write_buffered(%6, %11, %22) : (i32, !llvm.ptr, i32) -> ()
    func.return
  ^bb5:  // pred: ^bb0
    %23 = llvm.call @snprintf(%11, %4, %9, %12) vararg(!llvm.func<i32 (ptr, i64, ptr, ...)>) : (!llvm.ptr, i64, !llvm.ptr, !llvm.ptr) -> i32
    func.call @write_buffered(%6, %11, %23) : (i32, !llvm.ptr, i32) -> ()
    func.return
  ^bb6:  // pred: ^bb2
    %24 = llvm.call @snprintf(%11, %4, %7, %12) vararg(!llvm.func<i32 (ptr, i64, ptr, ...)>) : (!llvm.ptr, i64, !llvm.ptr, !llvm.ptr) -> i32
    func.call @write_buffered(%6, %11, %24) : (i32, !llvm.ptr, i32) -> ()
    func.return
  }
  func.func @LyTraceback_PrintMessage(%arg0: i64, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i64, %arg4: i64, %arg5: i64) {
    %0 = arith.constant 2 : i32
    %1 = llvm.mlir.addressof @".traceback_header" : !llvm.ptr
    %2 = llvm.mlir.addressof @g_traceback_size : !llvm.ptr
    %3 = arith.constant 0 : i64
    %4 = arith.constant 1 : i64
    func.call @write_cstr(%0, %1) : (i32, !llvm.ptr) -> ()
    %5 = llvm.load %2 {alignment = 8 : i64} : !llvm.ptr -> i64
    cf.br ^bb1(%5 : i64)
  ^bb1(%6: i64):  // 2 preds: ^bb0, ^bb2
    %7 = arith.cmpi eq, %6, %3 : i64
    cf.cond_br %7, ^bb3, ^bb2
  ^bb2:  // pred: ^bb1
    %8 = arith.subi %6, %4 : i64
    %9 = func.call @frame_at(%8) : (i64) -> !llvm.ptr
    func.call @print_trace_frame(%9) : (!llvm.ptr) -> ()
    cf.br ^bb1(%8 : i64)
  ^bb3:  // pred: ^bb1
    func.call @print_exception_summary(%arg0, %arg2, %arg3, %arg4, %arg5) : (i64, !llvm.ptr, i64, i64, i64) -> ()
    func.return
  }
  func.func private @exception_base_class_id(%arg0: i64) -> i64 {
    %0 = arith.constant 59 : i64
    %1 = arith.constant 60 : i64
    %2 = arith.constant 50 : i64
    %3 = arith.constant 5 : i64
    %4 = arith.constant 0 : i64
    cf.switch %arg0 : i64, [
      default: ^bb5,
      50: ^bb1,
      62: ^bb1,
      51: ^bb2,
      52: ^bb2,
      53: ^bb2,
      56: ^bb2,
      57: ^bb2,
      58: ^bb2,
      59: ^bb2,
      60: ^bb2,
      54: ^bb3,
      55: ^bb3,
      61: ^bb4
    ]
  ^bb1:  // 2 preds: ^bb0, ^bb0
    func.return %3 : i64
  ^bb2:  // 8 preds: ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0
    func.return %2 : i64
  ^bb3:  // 2 preds: ^bb0, ^bb0
    func.return %1 : i64
  ^bb4:  // pred: ^bb0
    func.return %0 : i64
  ^bb5:  // pred: ^bb0
    func.return %4 : i64
  }
  func.func private @current_exception_class_id_unchecked() -> i64 {
    %0 = llvm.mlir.addressof @g_current_parts : !llvm.ptr
    %1 = arith.constant 0 : i32
    %2 = arith.constant 1 : i32
    %3 = arith.constant 2 : i32
    %4 = arith.constant 4 : i32
    %5 = llvm.mlir.zero : !llvm.ptr
    %6 = arith.constant 2 : i64
    %7 = llvm.getelementptr inbounds %0[%1, 0, 1] : (!llvm.ptr, i32) -> !llvm.ptr, !ExceptionParts
    %8 = llvm.getelementptr inbounds %0[%1, 0, 2] : (!llvm.ptr, i32) -> !llvm.ptr, !ExceptionParts
    %9 = llvm.getelementptr inbounds %0[%1, 0, 4] : (!llvm.ptr, i32) -> !llvm.ptr, !ExceptionParts
    %10 = llvm.load %7 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %11 = llvm.load %8 {alignment = 8 : i64} : !llvm.ptr -> i64
    %12 = llvm.load %9 {alignment = 8 : i64} : !llvm.ptr -> i64
    %13 = llvm.icmp "eq" %10, %5 : !llvm.ptr
    cf.cond_br %13, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    %14 = arith.muli %12, %6 : i64
    %15 = arith.addi %11, %14 : i64
    %16 = llvm.getelementptr %10[%15] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %17 = llvm.load %16 {alignment = 8 : i64} : !llvm.ptr -> i64
    func.return %17 : i64
  ^bb2:  // pred: ^bb0
    llvm.call @rt_abort() : () -> ()
    llvm.unreachable
  }
  func.func private @end_native_catch_if_active() {
    %0 = llvm.mlir.addressof @g_native_catch_active : !llvm.ptr
    %1 = arith.constant false
    %2 = llvm.load %0 {alignment = 4 : i64} : !llvm.ptr -> i1
    cf.cond_br %2, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    llvm.call @__cxa_end_catch() : () -> ()
    llvm.store %1, %0 {alignment = 4 : i64} : i1, !llvm.ptr
    cf.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    func.return
  }
  llvm.func @LyEH_ThrowException(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: !llvm.ptr, %arg11: !llvm.ptr, %arg12: i64, %arg13: i64, %arg14: i64) {
    %0 = llvm.mlir.addressof @g_current_exception : !llvm.ptr
    %1 = llvm.mlir.addressof @g_current_parts : !llvm.ptr
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(1 : i32) : i32
    %4 = llvm.mlir.constant(2 : i32) : i32
    %5 = llvm.mlir.constant(3 : i32) : i32
    %6 = llvm.mlir.constant(4 : i32) : i32
    %7 = llvm.mlir.constant(true) : i1
    %8 = llvm.mlir.constant(1 : i64) : i64
    %9 = llvm.mlir.addressof @_ZTI17LyPythonException : !llvm.ptr
    %10 = llvm.mlir.zero : !llvm.ptr
    %11 = llvm.load %0 {alignment = 4 : i64} : !llvm.ptr -> i1
    llvm.cond_br %11, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    %12 = llvm.getelementptr inbounds %1[%2, 0, 0] : (!llvm.ptr, i32) -> !llvm.ptr, !ExceptionParts
    %13 = llvm.getelementptr inbounds %1[%2, 0, 1] : (!llvm.ptr, i32) -> !llvm.ptr, !ExceptionParts
    %14 = llvm.getelementptr inbounds %1[%2, 0, 2] : (!llvm.ptr, i32) -> !llvm.ptr, !ExceptionParts
    %15 = llvm.getelementptr inbounds %1[%2, 0, 3] : (!llvm.ptr, i32) -> !llvm.ptr, !ExceptionParts
    %16 = llvm.getelementptr inbounds %1[%2, 0, 4] : (!llvm.ptr, i32) -> !llvm.ptr, !ExceptionParts
    %17 = llvm.getelementptr inbounds %1[%2, 1, 0] : (!llvm.ptr, i32) -> !llvm.ptr, !ExceptionParts
    %18 = llvm.getelementptr inbounds %1[%2, 1, 1] : (!llvm.ptr, i32) -> !llvm.ptr, !ExceptionParts
    %19 = llvm.getelementptr inbounds %1[%2, 1, 2] : (!llvm.ptr, i32) -> !llvm.ptr, !ExceptionParts
    %20 = llvm.getelementptr inbounds %1[%2, 1, 3] : (!llvm.ptr, i32) -> !llvm.ptr, !ExceptionParts
    %21 = llvm.getelementptr inbounds %1[%2, 1, 4] : (!llvm.ptr, i32) -> !llvm.ptr, !ExceptionParts
    %22 = llvm.getelementptr inbounds %1[%2, 2, 0] : (!llvm.ptr, i32) -> !llvm.ptr, !ExceptionParts
    %23 = llvm.getelementptr inbounds %1[%2, 2, 1] : (!llvm.ptr, i32) -> !llvm.ptr, !ExceptionParts
    %24 = llvm.getelementptr inbounds %1[%2, 2, 2] : (!llvm.ptr, i32) -> !llvm.ptr, !ExceptionParts
    %25 = llvm.getelementptr inbounds %1[%2, 2, 3] : (!llvm.ptr, i32) -> !llvm.ptr, !ExceptionParts
    %26 = llvm.getelementptr inbounds %1[%2, 2, 4] : (!llvm.ptr, i32) -> !llvm.ptr, !ExceptionParts
    llvm.store %arg0, %12 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %arg1, %13 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %arg2, %14 {alignment = 8 : i64} : i64, !llvm.ptr
    llvm.store %arg3, %15 {alignment = 8 : i64} : i64, !llvm.ptr
    llvm.store %arg4, %16 {alignment = 8 : i64} : i64, !llvm.ptr
    llvm.store %arg5, %17 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %arg6, %18 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %arg7, %19 {alignment = 8 : i64} : i64, !llvm.ptr
    llvm.store %arg8, %20 {alignment = 8 : i64} : i64, !llvm.ptr
    llvm.store %arg9, %21 {alignment = 8 : i64} : i64, !llvm.ptr
    llvm.store %arg10, %22 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %arg11, %23 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %arg12, %24 {alignment = 8 : i64} : i64, !llvm.ptr
    llvm.store %arg13, %25 {alignment = 8 : i64} : i64, !llvm.ptr
    llvm.store %arg14, %26 {alignment = 8 : i64} : i64, !llvm.ptr
    llvm.store %7, %0 {alignment = 4 : i64} : i1, !llvm.ptr
    %27 = llvm.call @__cxa_allocate_exception(%8) : (i64) -> !llvm.ptr
    llvm.call @__cxa_throw(%27, %9, %10) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.unreachable
  ^bb2:  // pred: ^bb0
    llvm.call @rt_abort() : () -> ()
    llvm.unreachable
  }
  llvm.func @LyEH_BeginCatch(%arg0: !llvm.ptr) {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.addressof @g_current_exception : !llvm.ptr
    %2 = llvm.mlir.constant(true) : i1
    %3 = llvm.mlir.addressof @g_native_catch_active : !llvm.ptr
    %4 = llvm.icmp "eq" %arg0, %0 : !llvm.ptr
    %5 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i1
    %6 = llvm.xor %5, %2 : i1
    %7 = llvm.or %4, %6 : i1
    llvm.cond_br %7, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    %8 = llvm.call @__cxa_begin_catch(%arg0) : (!llvm.ptr) -> !llvm.ptr
    llvm.store %2, %3 {alignment = 4 : i64} : i1, !llvm.ptr
    llvm.return
  ^bb2:  // pred: ^bb0
    llvm.call @rt_abort() : () -> ()
    llvm.unreachable
  }
  func.func @LyEH_ClassIdMatches(%arg0: i64, %arg1: i64) -> i1 {
    %0 = arith.constant 0 : i64
    %1 = arith.constant true
    %2 = arith.constant false
    cf.br ^bb1(%arg0 : i64)
  ^bb1(%3: i64):  // 2 preds: ^bb0, ^bb3
    %4 = arith.cmpi eq, %3, %0 : i64
    cf.cond_br %4, ^bb5, ^bb2
  ^bb2:  // pred: ^bb1
    %5 = arith.cmpi eq, %3, %arg1 : i64
    cf.cond_br %5, ^bb4, ^bb3
  ^bb3:  // pred: ^bb2
    %6 = func.call @exception_base_class_id(%3) : (i64) -> i64
    cf.br ^bb1(%6 : i64)
  ^bb4:  // pred: ^bb2
    func.return %1 : i1
  ^bb5:  // pred: ^bb1
    func.return %2 : i1
  }
  func.func @LyEH_CurrentExceptionClassId() -> i64 {
    %0 = llvm.mlir.addressof @g_current_exception : !llvm.ptr
    %1 = arith.constant 0 : i64
    %2 = llvm.load %0 {alignment = 4 : i64} : !llvm.ptr -> i1
    cf.cond_br %2, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %3 = func.call @current_exception_class_id_unchecked() : () -> i64
    func.return %3 : i64
  ^bb2:  // pred: ^bb0
    func.return %1 : i64
  }
  func.func @LyEH_CurrentExceptionMatches(%arg0: i64) -> i1 {
    %0 = func.call @LyEH_CurrentExceptionClassId() : () -> i64
    %1 = func.call @LyEH_ClassIdMatches(%0, %arg0) : (i64, i64) -> i1
    func.return %1 : i1
  }
  func.func @LyEH_DiscardCurrentExceptionIfMatches(%arg0: i64) -> i1 {
    %0 = arith.constant false
    %1 = arith.constant true
    %2 = func.call @LyEH_CurrentExceptionMatches(%arg0) : (i64) -> i1
    cf.cond_br %2, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    func.call @LyEH_DiscardCurrentException() : () -> ()
    func.return %1 : i1
  ^bb2:  // pred: ^bb0
    func.return %0 : i1
  }
  func.func @LyEH_DiscardCurrentException() {
    %0 = arith.constant false
    %1 = llvm.mlir.addressof @g_current_exception : !llvm.ptr
    %2 = llvm.mlir.addressof @g_current_parts : !llvm.ptr
    %3 = arith.constant 0 : i8
    %4 = arith.constant 120 : i64
    func.call @end_native_catch_if_active() : () -> ()
    llvm.store %0, %1 {alignment = 4 : i64} : i1, !llvm.ptr
    "llvm.intr.memset"(%2, %3, %4) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
    func.call @LyTraceback_Clear() : () -> ()
    func.return
  }
  llvm.func @LyEH_RethrowCurrent() {
    %0 = llvm.mlir.addressof @g_current_exception : !llvm.ptr
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.mlir.addressof @_ZTI17LyPythonException : !llvm.ptr
    %3 = llvm.mlir.zero : !llvm.ptr
    %4 = llvm.load %0 {alignment = 4 : i64} : !llvm.ptr -> i1
    llvm.cond_br %4, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    func.call @end_native_catch_if_active() : () -> ()
    %5 = llvm.call @__cxa_allocate_exception(%1) : (i64) -> !llvm.ptr
    llvm.call @__cxa_throw(%5, %2, %3) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.unreachable
  ^bb2:  // pred: ^bb0
    llvm.call @rt_abort() : () -> ()
    llvm.unreachable
  }
  func.func @LyEH_TakeCurrentDescriptor(%arg0: !llvm.ptr) -> i1 {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.addressof @g_current_exception : !llvm.ptr
    %2 = arith.constant false
    %3 = llvm.mlir.addressof @g_current_parts : !llvm.ptr
    %4 = arith.constant 120 : i64
    %5 = arith.constant 0 : i8
    %6 = arith.constant true
    %7 = llvm.icmp "ne" %arg0, %0 : !llvm.ptr
    %8 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i1
    %9 = arith.andi %7, %8 : i1
    cf.cond_br %9, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    func.call @end_native_catch_if_active() : () -> ()
    "llvm.intr.memcpy"(%arg0, %3, %4) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    "llvm.intr.memset"(%3, %5, %4) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
    llvm.store %2, %1 {alignment = 4 : i64} : i1, !llvm.ptr
    func.return %6 : i1
  ^bb2:  // pred: ^bb0
    func.return %2 : i1
  }
  llvm.func @LyRunPythonMain(%arg0: !llvm.ptr) -> i32 attributes {personality = @__gxx_personality_v0} {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(2 : i32) : i32
    %3 = llvm.mlir.addressof @".native_exception" : !llvm.ptr
    %4 = llvm.mlir.constant(0 : i32) : i32
    %5 = llvm.mlir.constant(4 : i32) : i32
    %6 = llvm.mlir.constant(3 : i32) : i32
    %7 = llvm.mlir.constant(2 : i64) : i64
    llvm.call @LyRt_InstallStackGuard() : () -> ()
    %8 = llvm.icmp "eq" %arg0, %0 : !llvm.ptr
    llvm.cond_br %8, ^bb3, ^bb1
  ^bb1:  // pred: ^bb0
    llvm.invoke %arg0() to ^bb2 unwind ^bb4 : !llvm.ptr, () -> ()
  ^bb2:  // pred: ^bb1
    llvm.return %4 : i32
  ^bb3:  // pred: ^bb0
    llvm.return %1 : i32
  ^bb4:  // pred: ^bb1
    %9 = llvm.landingpad (catch %0 : !llvm.ptr) : !llvm.struct<(ptr, i32)>
    %10 = llvm.extractvalue %9[0] : !llvm.struct<(ptr, i32)> 
    %11 = llvm.call @__cxa_begin_catch(%10) : (!llvm.ptr) -> !llvm.ptr
    %12 = llvm.alloca %1 x !ExceptionParts {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %13 = func.call @LyEH_TakeCurrentDescriptor(%12) : (!llvm.ptr) -> i1
    llvm.cond_br %13, ^bb6, ^bb5
  ^bb5:  // pred: ^bb4
    func.call @write_cstr(%2, %3) : (i32, !llvm.ptr) -> ()
    func.call @LyTraceback_Clear() : () -> ()
    llvm.call @__cxa_end_catch() : () -> ()
    llvm.return %1 : i32
  ^bb6:  // pred: ^bb4
    %14 = llvm.getelementptr inbounds %12[%4, 0, 1] : (!llvm.ptr, i32) -> !llvm.ptr, !ExceptionParts
    %15 = llvm.getelementptr inbounds %12[%4, 0, 2] : (!llvm.ptr, i32) -> !llvm.ptr, !ExceptionParts
    %16 = llvm.getelementptr inbounds %12[%4, 0, 4] : (!llvm.ptr, i32) -> !llvm.ptr, !ExceptionParts
    %17 = llvm.getelementptr inbounds %12[%4, 2, 1] : (!llvm.ptr, i32) -> !llvm.ptr, !ExceptionParts
    %18 = llvm.getelementptr inbounds %12[%4, 2, 2] : (!llvm.ptr, i32) -> !llvm.ptr, !ExceptionParts
    %19 = llvm.getelementptr inbounds %12[%4, 2, 3] : (!llvm.ptr, i32) -> !llvm.ptr, !ExceptionParts
    %20 = llvm.getelementptr inbounds %12[%4, 2, 4] : (!llvm.ptr, i32) -> !llvm.ptr, !ExceptionParts
    %21 = llvm.load %14 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %22 = llvm.load %15 {alignment = 8 : i64} : !llvm.ptr -> i64
    %23 = llvm.load %16 {alignment = 8 : i64} : !llvm.ptr -> i64
    %24 = llvm.mul %23, %7 : i64
    %25 = llvm.add %22, %24 : i64
    %26 = llvm.getelementptr %21[%25] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %27 = llvm.load %26 {alignment = 8 : i64} : !llvm.ptr -> i64
    %28 = llvm.load %17 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %29 = llvm.load %18 {alignment = 8 : i64} : !llvm.ptr -> i64
    %30 = llvm.load %19 {alignment = 8 : i64} : !llvm.ptr -> i64
    %31 = llvm.load %20 {alignment = 8 : i64} : !llvm.ptr -> i64
    func.call @LyTraceback_PrintMessage(%27, %0, %28, %29, %30, %31) : (i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    func.call @LyTraceback_Clear() : () -> ()
    llvm.call @__cxa_end_catch() : () -> ()
    llvm.return %1 : i32
  }
}
