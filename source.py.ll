@.str.0 = private unnamed_addr constant [14 x i8] c"Hello, world!\00", align 1
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
  %3 = call ptr @PyInt_FromLong(i64 noundef 1)
  %4 = call ptr @PyInt_FromLong(i64 noundef 1)
  ; 左オペランドのメソッドテーブルを取得
    %mt.6 = getelementptr inbounds %struct.PyObject, ptr %3, i32 0, i32 2
    %mt_ptr.6 = load ptr, ptr %mt.6
    ; __add__メソッドのポインタを取得
    %method.6 = getelementptr inbounds %struct.PyMethodTable, ptr %mt_ptr.6, i32 0, i32 6
    %5 = load ptr, ptr %method.6
  %6 = call ptr %5(ptr noundef %3, ptr noundef %4)
  %isnull.8 = icmp eq ptr %6, null
    br i1 %isnull.8, label %fallback.7, label %end.7

    fallback.7:
    ; 右オペランドの__radd__を呼び出す
    %rmt.8 = getelementptr inbounds %struct.PyObject, ptr %4, i32 0, i32 2
    %rmt_ptr.8 = load ptr, ptr %rmt.8
    %rmethod.8 = getelementptr inbounds %struct.PyMethodTable, ptr %rmt_ptr.8, i32 0, i32 11
    %rmethod_ptr.8 = load ptr, ptr %rmethod.8
    %rresult.8 = call ptr %rmethod_ptr.8(ptr noundef %4, ptr noundef %3)
    br label %end.7

    end.7:
    %final.8 = phi ptr [ %6, %entry ], [ %rresult.8, %fallback.7 ]
  ret i32 0
}