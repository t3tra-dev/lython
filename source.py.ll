@.str.0 = private unnamed_addr constant [14 x i8] c"\48\65\6c\6c\6f\2c\20\77\6f\72\6c\64\21\00", align 1
@.str.1 = private unnamed_addr constant [43 x i8] c"\6d\75\6c\74\69\62\69\74\65\20\63\68\61\72\61\63\74\65\72\3a\20\e3\81\82\e3\81\84\e3\81\86\e3\81\88\e3\81\8a\2c\20\f0\9f\90\8d\00", align 1
; 外部関数の宣言
declare ptr @PyInt_FromLong(i64 noundef)
declare ptr @PyString_FromString(ptr noundef)
declare i32 @puts(ptr nocapture readonly) local_unnamed_addr

; 構造体定義
%struct.PyObject = type { ptr, i64, ptr }

%struct.PyMethodTable = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }

%struct.PyStringObject = type { %struct.PyObject, ptr, i64 }

%struct.PyIntObject = type { %struct.PyObject, i64 }


define i32 @main(i32 %argc, i8** %argv) {
entry:
  %0 = call ptr @PyString_FromString(ptr noundef @.str.0)
  ; __str__メソッドの取得と呼び出し
  %mt.2 = getelementptr inbounds %struct.PyObject, ptr %0, i32 0, i32 2
  %mt_ptr.2 = load ptr, ptr %mt.2
  %strmethod.2 = getelementptr inbounds %struct.PyMethodTable, ptr %mt_ptr.2, i32 0, i32 19
  %strmethod_ptr.2 = load ptr, ptr %strmethod.2
  %str.1 = call ptr %strmethod_ptr.2(ptr noundef %0)
  %strptr.2 = getelementptr inbounds %struct.PyStringObject, ptr %str.1, i32 0, i32 1
  %str.3 = load ptr, ptr %strptr.2
  %puts.3 = call i32 @puts(ptr noundef %str.3)
  %3 = call ptr @PyString_FromString(ptr noundef @.str.1)
  ; __str__メソッドの取得と呼び出し
  %mt.5 = getelementptr inbounds %struct.PyObject, ptr %3, i32 0, i32 2
  %mt_ptr.5 = load ptr, ptr %mt.5
  %strmethod.5 = getelementptr inbounds %struct.PyMethodTable, ptr %mt_ptr.5, i32 0, i32 19
  %strmethod_ptr.5 = load ptr, ptr %strmethod.5
  %str.4 = call ptr %strmethod_ptr.5(ptr noundef %3)
  %strptr.5 = getelementptr inbounds %struct.PyStringObject, ptr %str.4, i32 0, i32 1
  %str.6 = load ptr, ptr %strptr.5
  %puts.6 = call i32 @puts(ptr noundef %str.6)
  %6 = call ptr @PyInt_FromLong(i64 noundef 1)
  %7 = call ptr @PyInt_FromLong(i64 noundef 1)
  ; 左オペランドのメソッドテーブルを取得
    %mt.9 = getelementptr inbounds %struct.PyObject, ptr %6, i32 0, i32 2
    %mt_ptr.9 = load ptr, ptr %mt.9
    ; __add__メソッドのポインタを取得
    %method.9 = getelementptr inbounds %struct.PyMethodTable, ptr %mt_ptr.9, i32 0, i32 6
    %8 = load ptr, ptr %method.9
  %9 = call ptr %8(ptr noundef %6, ptr noundef %7)
  %isnull.11 = icmp eq ptr %9, null
    br i1 %isnull.11, label %fallback.10, label %end.10

    fallback.10:
    ; 右オペランドの__radd__を呼び出す
    %rmt.11 = getelementptr inbounds %struct.PyObject, ptr %7, i32 0, i32 2
    %rmt_ptr.11 = load ptr, ptr %rmt.11
    %rmethod.11 = getelementptr inbounds %struct.PyMethodTable, ptr %rmt_ptr.11, i32 0, i32 11
    %rmethod_ptr.11 = load ptr, ptr %rmethod.11
    %rresult.11 = call ptr %rmethod_ptr.11(ptr noundef %7, ptr noundef %6)
    br label %end.10

    end.10:
    %final.11 = phi ptr [ %9, %entry ], [ %rresult.11, %fallback.10 ]
  ret i32 0
}