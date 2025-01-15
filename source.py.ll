@.str.0 = private unnamed_addr constant [14 x i8] c"\48\65\6c\6c\6f\2c\20\77\6f\72\6c\64\21\00", align 1
@.str.1 = private unnamed_addr constant [43 x i8] c"\6d\75\6c\74\69\62\69\74\65\20\63\68\61\72\61\63\74\65\72\3a\20\e3\81\82\e3\81\84\e3\81\86\e3\81\88\e3\81\8a\2c\20\f0\9f\90\8d\00", align 1
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
  %t6 = zext i1 %t5 to i64
  %t7 = call ptr @PyInt_FromLong(i64 %t6)
  %t8 = getelementptr %struct.PyIntObject, ptr %t7, i32 0, i32 1
  %t9 = load i64, ptr %t8
  %t10 = trunc i64 %t9 to i1
  br i1 %t10, label %if.then.0, label %if.else.1
if.then.0:
  ret ptr %n
  br label %if.end.2
if.else.1:
  br label %if.end.2
if.end.2:
  %t11 = call ptr @PyInt_FromLong(i64 1)
  %t12 = getelementptr %struct.PyIntObject, ptr %n, i32 0, i32 1
  %t13 = getelementptr %struct.PyIntObject, ptr %t11, i32 0, i32 1
  %t14 = load i64, ptr %t12
  %t15 = load i64, ptr %t13
  %t16 = sub i64 %t14, %t15
  %t17 = call ptr @PyInt_FromLong(i64 %t16)
  %t18 = call ptr @fib(ptr %t17)
  %t19 = call ptr @PyInt_FromLong(i64 2)
  %t20 = getelementptr %struct.PyIntObject, ptr %n, i32 0, i32 1
  %t21 = getelementptr %struct.PyIntObject, ptr %t19, i32 0, i32 1
  %t22 = load i64, ptr %t20
  %t23 = load i64, ptr %t21
  %t24 = sub i64 %t22, %t23
  %t25 = call ptr @PyInt_FromLong(i64 %t24)
  %t26 = call ptr @fib(ptr %t25)
  %t27 = getelementptr %struct.PyIntObject, ptr %t18, i32 0, i32 1
  %t28 = getelementptr %struct.PyIntObject, ptr %t26, i32 0, i32 1
  %t29 = load i64, ptr %t27
  %t30 = load i64, ptr %t28
  %t31 = add i64 %t29, %t30
  %t32 = call ptr @PyInt_FromLong(i64 %t31)
  ret ptr %t32
}

define i32 @main(i32 %argc, i8** %argv) {
entry:
  %t33 = call ptr @PyString_FromString(ptr @.str.0)
  %t34 = call i32 @print_object(ptr %t33)
  %t35 = call ptr @PyString_FromString(ptr @.str.1)
  %t36 = call i32 @print_object(ptr %t35)
  %t37 = call ptr @PyInt_FromLong(i64 30)
  %t38 = call ptr @fib(ptr %t37)
  %t39 = call ptr @str(ptr %t38)
  %t40 = call i32 @print_object(ptr %t39)
  ret i32 0
}