; 定数の定義
@Py_True = global i1 true, align 1
@Py_False = global i1 false, align 1
@Py_None = global ptr null, align 8


attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
; 外部関数の宣言
declare i32 @print(ptr noundef)
declare i32 @print_object(ptr noundef)
declare i32 @puts(ptr nocapture readonly) local_unnamed_addr
declare ptr @PyInt_FromLong(i64)
declare ptr @str(ptr noundef)
declare ptr @PyString_FromString(ptr noundef)

; 構造体定義
%struct.PyObject = type { ptr, i64, ptr }

%struct.PyMethodTable = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }

%struct.PyStringObject = type { %struct.PyObject, ptr, i64 }

%struct.PyIntObject = type { %struct.PyObject, i64 }

%struct.PyBaseExceptionObject = type { %struct.PyObject, ptr }


; Function definition
define dso_local ptr @fib(ptr noundef %n) #0 {
entry:
  ; Function body
  %t0 = call ptr @PyInt_FromLong(i64 1)
  %t1 = getelementptr %struct.PyIntObject, ptr %n, i32 0, i32 1
  %t2 = getelementptr %struct.PyIntObject, ptr %t0, i32 0, i32 1
  %t3 = load i64, ptr %t1
  %t4 = load i64, ptr %t2
  %t5 = icmp sle i64 %t3, %t4
  br i1 %t5, label %cmp.true.0, label %cmp.false.1
cmp.true.0:
  %t6 = call ptr @PyInt_FromLong(i64 1)
  br label %cmp.end.2
cmp.false.1:
  %t7 = call ptr @PyInt_FromLong(i64 0)
  br label %cmp.end.2
cmp.end.2:
  %t8 = phi ptr [ %t6, %cmp.true.0 ], [ %t7, %cmp.false.1 ]
  %t9 = getelementptr %struct.PyIntObject, ptr %t8, i32 0, i32 1
  %t10 = load i64, ptr %t9
  %t11 = trunc i64 %t10 to i1
  br i1 %t11, label %if.then.3, label %if.else.4
if.then.3:
  ret ptr %n
  br label %if.end.5
if.else.4:
  br label %if.end.5
if.end.5:
  %t12 = call ptr @PyInt_FromLong(i64 1)
  %t13 = getelementptr %struct.PyIntObject, ptr %n, i32 0, i32 1
  %t14 = getelementptr %struct.PyIntObject, ptr %t12, i32 0, i32 1
  %t15 = load i64, ptr %t13
  %t16 = load i64, ptr %t14
  %t17 = sub i64 %t15, %t16
  %t18 = call ptr @PyInt_FromLong(i64 %t17)
  %t19 = call ptr @fib(ptr %t18)
  %t20 = call ptr @PyInt_FromLong(i64 2)
  %t21 = getelementptr %struct.PyIntObject, ptr %n, i32 0, i32 1
  %t22 = getelementptr %struct.PyIntObject, ptr %t20, i32 0, i32 1
  %t23 = load i64, ptr %t21
  %t24 = load i64, ptr %t22
  %t25 = sub i64 %t23, %t24
  %t26 = call ptr @PyInt_FromLong(i64 %t25)
  %t27 = call ptr @fib(ptr %t26)
  %t28 = getelementptr %struct.PyIntObject, ptr %t19, i32 0, i32 1
  %t29 = getelementptr %struct.PyIntObject, ptr %t27, i32 0, i32 1
  %t30 = load i64, ptr %t28
  %t31 = load i64, ptr %t29
  %t32 = add i64 %t30, %t31
  %t33 = call ptr @PyInt_FromLong(i64 %t32)
  ret ptr %t33
}

define i32 @main(i32 %argc, i8** %argv) {
entry:
  %t34 = call ptr @PyInt_FromLong(i64 30)
  %t35 = call ptr @fib(ptr %t34)
  %t36 = call ptr @str(ptr %t35)
  %t37 = call i32 @print_object(ptr %t36)
  ret i32 0
}