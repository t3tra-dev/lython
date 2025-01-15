@.str.0 = private unnamed_addr constant [14 x i8] c"\48\65\6c\6c\6f\2c\20\77\6f\72\6c\64\21\00", align 1
@.str.1 = private unnamed_addr constant [43 x i8] c"\6d\75\6c\74\69\62\69\74\65\20\63\68\61\72\61\63\74\65\72\3a\20\e3\81\82\e3\81\84\e3\81\86\e3\81\88\e3\81\8a\2c\20\f0\9f\90\8d\00", align 1
; 定数の定義
@Py_True = global i1 true, align 1
@Py_False = global i1 false, align 1
@Py_None = global ptr null, align 8

; 外部関数の宣言
declare i32 @print(ptr noundef)
declare i32 @puts(ptr nocapture readonly) local_unnamed_addr

; 構造体定義
%struct.PyObject = type { ptr, i64, ptr }

%struct.PyMethodTable = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }

%struct.PyStringObject = type { %struct.PyObject, ptr, i64 }

%struct.PyIntObject = type { %struct.PyObject, i64 }

%struct.PyBaseExceptionObject = type { %struct.PyObject, ptr }


define i32 @main(i32 %argc, i8** %argv) {
entry:
  %t0 = getelementptr [14 x i8], ptr @.str.0, i64 0, i64 0
  %t1 = call i32 @print(ptr %t0)
  %t2 = getelementptr [43 x i8], ptr @.str.1, i64 0, i64 0
  %t3 = call i32 @print(ptr %t2)
  ret i32 0
}