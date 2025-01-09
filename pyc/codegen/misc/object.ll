; ModuleID = 'pyc/codegen/misc/object.c'
source_filename = "pyc/codegen/misc/object.c"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx15.0.0"

%struct.TypeObject = type { %struct.PyObject, ptr, i64, i64, ptr, ptr }
%struct.PyObject = type { ptr, i64, ptr }
%struct.PyMethodTable = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }
%struct.PyStringObject = type { %struct.PyObject, ptr, i64 }
%struct.PyIntObject = type { %struct.PyObject, i64 }

@.str = private unnamed_addr constant [15 x i8] c"<object at %p>\00", align 1
@PyInt_Type = global %struct.TypeObject { %struct.PyObject { ptr @PyType_Type, i64 1, ptr @int_method_table }, ptr @.str.2, i64 32, i64 0, ptr null, ptr null }, align 8
@default_method_table = global %struct.PyMethodTable { ptr @default_eq, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr @default_str, ptr null, ptr null, ptr null, ptr null, ptr @default_dealloc, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null }, align 8
@int_method_table = global %struct.PyMethodTable { ptr @default_eq, ptr null, ptr null, ptr null, ptr null, ptr null, ptr @int_add, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr @default_str, ptr null, ptr null, ptr null, ptr null, ptr @default_dealloc, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null }, align 8
@string_method_table = global %struct.PyMethodTable { ptr @default_eq, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr @string_str, ptr null, ptr null, ptr null, ptr null, ptr @default_dealloc, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null }, align 8
@.str.1 = private unnamed_addr constant [5 x i8] c"type\00", align 1
@PyType_Type = global %struct.TypeObject { %struct.PyObject { ptr null, i64 1, ptr @default_method_table }, ptr @.str.1, i64 64, i64 0, ptr null, ptr null }, align 8
@.str.2 = private unnamed_addr constant [4 x i8] c"int\00", align 1
@.str.3 = private unnamed_addr constant [4 x i8] c"str\00", align 1
@PyString_Type = global %struct.TypeObject { %struct.PyObject { ptr @PyType_Type, i64 1, ptr @string_method_table }, ptr @.str.3, i64 40, i64 0, ptr null, ptr null }, align 8

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define i32 @default_eq(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  store ptr %1, ptr %4, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr %4, align 8
  %7 = icmp eq ptr %5, %6
  %8 = zext i1 %7 to i32
  ret i32 %8
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define ptr @default_str(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  %3 = alloca [128 x i8], align 1
  store ptr %0, ptr %2, align 8
  %4 = getelementptr inbounds [128 x i8], ptr %3, i64 0, i64 0
  %5 = load ptr, ptr %2, align 8
  %6 = call i32 (ptr, i64, i32, i64, ptr, ...) @__snprintf_chk(ptr noundef %4, i64 noundef 128, i32 noundef 0, i64 noundef 128, ptr noundef @.str, ptr noundef %5)
  %7 = getelementptr inbounds [128 x i8], ptr %3, i64 0, i64 0
  %8 = call ptr @PyString_FromString(ptr noundef %7)
  ret ptr %8
}

declare i32 @__snprintf_chk(ptr noundef, i64 noundef, i32 noundef, i64 noundef, ptr noundef, ...) #1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define ptr @PyString_FromString(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca i64, align 8
  store ptr %0, ptr %3, align 8
  %6 = call ptr @PyObject_New(ptr noundef @PyString_Type)
  store ptr %6, ptr %4, align 8
  %7 = load ptr, ptr %4, align 8
  %8 = icmp eq ptr %7, null
  br i1 %8, label %9, label %10

9:                                                ; preds = %1
  store ptr null, ptr %2, align 8
  br label %38

10:                                               ; preds = %1
  %11 = load ptr, ptr %3, align 8
  %12 = call i64 @strlen(ptr noundef %11) #5
  store i64 %12, ptr %5, align 8
  %13 = load i64, ptr %5, align 8
  %14 = add i64 %13, 1
  %15 = call ptr @malloc(i64 noundef %14) #6
  %16 = load ptr, ptr %4, align 8
  %17 = getelementptr inbounds %struct.PyStringObject, ptr %16, i32 0, i32 1
  store ptr %15, ptr %17, align 8
  %18 = load ptr, ptr %4, align 8
  %19 = getelementptr inbounds %struct.PyStringObject, ptr %18, i32 0, i32 1
  %20 = load ptr, ptr %19, align 8
  %21 = icmp eq ptr %20, null
  br i1 %21, label %22, label %24

22:                                               ; preds = %10
  %23 = load ptr, ptr %4, align 8
  call void @free(ptr noundef %23)
  store ptr null, ptr %2, align 8
  br label %38

24:                                               ; preds = %10
  %25 = load ptr, ptr %4, align 8
  %26 = getelementptr inbounds %struct.PyStringObject, ptr %25, i32 0, i32 1
  %27 = load ptr, ptr %26, align 8
  %28 = load ptr, ptr %3, align 8
  %29 = load ptr, ptr %4, align 8
  %30 = getelementptr inbounds %struct.PyStringObject, ptr %29, i32 0, i32 1
  %31 = load ptr, ptr %30, align 8
  %32 = call i64 @llvm.objectsize.i64.p0(ptr %31, i1 false, i1 true, i1 false)
  %33 = call ptr @__strcpy_chk(ptr noundef %27, ptr noundef %28, i64 noundef %32) #5
  %34 = load i64, ptr %5, align 8
  %35 = load ptr, ptr %4, align 8
  %36 = getelementptr inbounds %struct.PyStringObject, ptr %35, i32 0, i32 2
  store i64 %34, ptr %36, align 8
  %37 = load ptr, ptr %4, align 8
  store ptr %37, ptr %2, align 8
  br label %38

38:                                               ; preds = %24, %22, %9
  %39 = load ptr, ptr %2, align 8
  ret ptr %39
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define void @default_dealloc(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds %struct.PyObject, ptr %3, i32 0, i32 1
  %5 = load i64, ptr %4, align 8
  %6 = add i64 %5, -1
  store i64 %6, ptr %4, align 8
  %7 = icmp eq i64 %6, 0
  br i1 %7, label %8, label %10

8:                                                ; preds = %1
  %9 = load ptr, ptr %2, align 8
  call void @free(ptr noundef %9)
  br label %10

10:                                               ; preds = %8, %1
  ret void
}

declare void @free(ptr noundef) #1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define ptr @int_add(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca i64, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  %9 = load ptr, ptr %4, align 8
  %10 = getelementptr inbounds %struct.PyObject, ptr %9, i32 0, i32 0
  %11 = load ptr, ptr %10, align 8
  %12 = icmp ne ptr %11, @PyInt_Type
  br i1 %12, label %18, label %13

13:                                               ; preds = %2
  %14 = load ptr, ptr %5, align 8
  %15 = getelementptr inbounds %struct.PyObject, ptr %14, i32 0, i32 0
  %16 = load ptr, ptr %15, align 8
  %17 = icmp ne ptr %16, @PyInt_Type
  br i1 %17, label %18, label %19

18:                                               ; preds = %13, %2
  store ptr null, ptr %3, align 8
  br label %31

19:                                               ; preds = %13
  %20 = load ptr, ptr %4, align 8
  store ptr %20, ptr %6, align 8
  %21 = load ptr, ptr %5, align 8
  store ptr %21, ptr %7, align 8
  %22 = load ptr, ptr %6, align 8
  %23 = getelementptr inbounds %struct.PyIntObject, ptr %22, i32 0, i32 1
  %24 = load i64, ptr %23, align 8
  %25 = load ptr, ptr %7, align 8
  %26 = getelementptr inbounds %struct.PyIntObject, ptr %25, i32 0, i32 1
  %27 = load i64, ptr %26, align 8
  %28 = add nsw i64 %24, %27
  store i64 %28, ptr %8, align 8
  %29 = load i64, ptr %8, align 8
  %30 = call ptr @PyInt_FromLong(i64 noundef %29)
  store ptr %30, ptr %3, align 8
  br label %31

31:                                               ; preds = %19, %18
  %32 = load ptr, ptr %3, align 8
  ret ptr %32
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define ptr @PyInt_FromLong(i64 noundef %0) #0 {
  %2 = alloca ptr, align 8
  %3 = alloca i64, align 8
  %4 = alloca ptr, align 8
  store i64 %0, ptr %3, align 8
  %5 = call ptr @PyObject_New(ptr noundef @PyInt_Type)
  store ptr %5, ptr %4, align 8
  %6 = load ptr, ptr %4, align 8
  %7 = icmp eq ptr %6, null
  br i1 %7, label %8, label %9

8:                                                ; preds = %1
  store ptr null, ptr %2, align 8
  br label %14

9:                                                ; preds = %1
  %10 = load i64, ptr %3, align 8
  %11 = load ptr, ptr %4, align 8
  %12 = getelementptr inbounds %struct.PyIntObject, ptr %11, i32 0, i32 1
  store i64 %10, ptr %12, align 8
  %13 = load ptr, ptr %4, align 8
  store ptr %13, ptr %2, align 8
  br label %14

14:                                               ; preds = %9, %8
  %15 = load ptr, ptr %2, align 8
  ret ptr %15
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define ptr @string_str(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  ret ptr %3
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define ptr @PyObject_New(ptr noundef %0) #0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = getelementptr inbounds %struct.TypeObject, ptr %5, i32 0, i32 2
  %7 = load i64, ptr %6, align 8
  %8 = call ptr @malloc(i64 noundef %7) #6
  store ptr %8, ptr %4, align 8
  %9 = load ptr, ptr %4, align 8
  %10 = icmp eq ptr %9, null
  br i1 %10, label %11, label %12

11:                                               ; preds = %1
  store ptr null, ptr %2, align 8
  br label %31

12:                                               ; preds = %1
  %13 = load ptr, ptr %3, align 8
  %14 = load ptr, ptr %4, align 8
  %15 = getelementptr inbounds %struct.PyObject, ptr %14, i32 0, i32 0
  store ptr %13, ptr %15, align 8
  %16 = load ptr, ptr %4, align 8
  %17 = getelementptr inbounds %struct.PyObject, ptr %16, i32 0, i32 1
  store i64 1, ptr %17, align 8
  %18 = load ptr, ptr %3, align 8
  %19 = icmp eq ptr %18, @PyType_Type
  br i1 %19, label %20, label %21

20:                                               ; preds = %12
  br label %26

21:                                               ; preds = %12
  %22 = load ptr, ptr %3, align 8
  %23 = getelementptr inbounds %struct.TypeObject, ptr %22, i32 0, i32 0
  %24 = getelementptr inbounds %struct.PyObject, ptr %23, i32 0, i32 2
  %25 = load ptr, ptr %24, align 8
  br label %26

26:                                               ; preds = %21, %20
  %27 = phi ptr [ @default_method_table, %20 ], [ %25, %21 ]
  %28 = load ptr, ptr %4, align 8
  %29 = getelementptr inbounds %struct.PyObject, ptr %28, i32 0, i32 2
  store ptr %27, ptr %29, align 8
  %30 = load ptr, ptr %4, align 8
  store ptr %30, ptr %2, align 8
  br label %31

31:                                               ; preds = %26, %11
  %32 = load ptr, ptr %2, align 8
  ret ptr %32
}

; Function Attrs: allocsize(0)
declare ptr @malloc(i64 noundef) #2

; Function Attrs: nounwind
declare i64 @strlen(ptr noundef) #3

; Function Attrs: nounwind
declare ptr @__strcpy_chk(ptr noundef, ptr noundef, i64 noundef) #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.objectsize.i64.p0(ptr, i1 immarg, i1 immarg, i1 immarg) #4

attributes #0 = { noinline nounwind optnone ssp uwtable(sync) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+ccdp,+ccidx,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a,+zcm,+zcz" }
attributes #1 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+ccdp,+ccidx,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a,+zcm,+zcz" }
attributes #2 = { allocsize(0) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+ccdp,+ccidx,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a,+zcm,+zcz" }
attributes #3 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+ccdp,+ccidx,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a,+zcm,+zcz" }
attributes #4 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { nounwind }
attributes #6 = { allocsize(0) }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 15, i32 1]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 8, !"PIC Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 1}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"Homebrew clang version 19.1.6"}
