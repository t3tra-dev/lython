@.str.0 = private unnamed_addr constant [14 x i8] c"\48\65\6c\6c\6f\2c\20\77\6f\72\6c\64\21\00", align 1
; 定数の定義
@Py_True = global i1 true, align 1
@Py_False = global i1 false, align 1
@Py_None = global ptr null, align 8

; 外部関数の宣言
declare ptr @PyInt_FromLong(i64 noundef)
declare ptr @PyString_FromString(ptr noundef)
declare i32 @puts(ptr nocapture readonly) local_unnamed_addr
declare void @raise_exception(ptr noundef)
declare ptr @PyObject_New(ptr noundef)
declare ptr @PyBaseException_New(ptr noundef, ptr noundef, ptr noundef)
declare ptr @baseexception_str(ptr noundef)
declare i32 @print(ptr noundef)

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
  ret i32 0
}