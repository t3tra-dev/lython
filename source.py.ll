@.str.0 = private unnamed_addr constant [14 x i8] c"\48\65\6c\6c\6f\2c\20\77\6f\72\6c\64\21\00", align 1
@.str.1 = private unnamed_addr constant [2 x i8] c"\61\00", align 1
@.str.2 = private unnamed_addr constant [2 x i8] c"\62\00", align 1
@.str.3 = private unnamed_addr constant [2 x i8] c"\63\00", align 1
@.str.4 = private unnamed_addr constant [5 x i8] c"\6b\65\79\31\00", align 1
@.str.5 = private unnamed_addr constant [5 x i8] c"\6b\65\79\32\00", align 1
@.str.6 = private unnamed_addr constant [5 x i8] c"\6b\65\79\33\00", align 1
@.str.7 = private unnamed_addr constant [5 x i8] c"\6b\65\79\31\00", align 1
; ========== オブジェクトシステム定数 ========== 
@Py_None = external global %struct.PyObject
@Py_True = external global %struct.PyObject
@Py_False = external global %struct.PyObject
@PyUnicode_Type = external global %struct.PyObject
@PyList_Type = external global %struct.PyObject
@PyDict_Type = external global %struct.PyObject
@PyBool_Type = external global %struct.PyObject

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" }
; ========== オブジェクトシステム関数宣言 ========== 
; オブジェクトシステムの初期化
declare void @PyObject_InitSystem()
declare void @PyObject_Init()
; 基本オブジェクト操作
declare ptr @PyObject_New(ptr)
declare ptr @PyObject_NewVar(ptr, i64)
declare ptr @PyObject_Repr(ptr)
declare ptr @PyObject_Str(ptr)
declare i32 @PyObject_Compare(ptr, ptr)
declare ptr @PyObject_RichCompare(ptr, ptr, i32)
declare i32 @PyObject_RichCompareBool(ptr, ptr, i32)
declare i64 @PyObject_Hash(ptr)
declare i32 @PyObject_IsTrue(ptr)
declare ptr @PyObject_GetAttrString(ptr, ptr)
declare i32 @PyObject_SetAttrString(ptr, ptr, ptr)
declare i32 @PyObject_HasAttrString(ptr, ptr)
; 型チェック・変換関数
declare i32 @PyType_IsSubtype(ptr, ptr)
declare void @PyType_Ready(ptr)
; 数値関連関数
declare ptr @PyInt_FromI32(i32)
declare i32 @PyInt_AsI32(ptr)
; ブール値関連
declare ptr @PyBool_FromLong(i32)
; Unicode文字列関連
declare ptr @PyUnicode_FromString(ptr)
declare ptr @PyUnicode_FromStringAndSize(ptr, i64)
declare ptr @PyUnicode_FromFormat(ptr, ...)
declare ptr @PyUnicode_Concat(ptr, ptr)
declare i64 @PyUnicode_GetLength(ptr)
declare ptr @PyUnicode_AsUTF8(ptr)
declare i32 @PyUnicode_Compare(ptr, ptr)
declare i32 @PyUnicode_CompareWithASCIIString(ptr, ptr)
declare i64 @PyUnicode_Hash(ptr)
; リスト関連
declare ptr @PyList_New(i64)
declare i64 @PyList_Size(ptr)
declare ptr @PyList_GetItem(ptr, i64)
declare i32 @PyList_SetItem(ptr, i64, ptr)
declare i32 @PyList_Insert(ptr, i64, ptr)
declare i32 @PyList_Append(ptr, ptr)
declare ptr @PyList_GetSlice(ptr, i64, i64)
declare i32 @PyList_SetSlice(ptr, i64, i64, ptr)
declare i32 @PyList_Sort(ptr)
declare i32 @PyList_Reverse(ptr)
; 辞書関連
declare ptr @PyDict_New()
declare ptr @PyDict_GetItem(ptr, ptr)
declare i32 @PyDict_SetItem(ptr, ptr, ptr)
declare i32 @PyDict_DelItem(ptr, ptr)
declare void @PyDict_Clear(ptr)
declare i32 @PyDict_Next(ptr, ptr, ptr, ptr)
declare ptr @PyDict_Keys(ptr)
declare ptr @PyDict_Values(ptr)
declare ptr @PyDict_Items(ptr)
declare i64 @PyDict_Size(ptr)
declare ptr @PyDict_GetItemString(ptr, ptr)
declare i32 @PyDict_SetItemString(ptr, ptr, ptr)
; GC関連
declare void @GC_init()

; 比較演算子の定数
; define i32 @Py_LT() { ret i32 0 }
; define i32 @Py_LE() { ret i32 1 }
; define i32 @Py_EQ() { ret i32 2 }
; define i32 @Py_NE() { ret i32 3 }
; define i32 @Py_GT() { ret i32 4 }
; define i32 @Py_GE() { ret i32 5 }

; 参照カウント操作
declare void @Py_INCREF(ptr)
declare void @Py_DECREF(ptr)
declare void @Py_XINCREF(ptr)
declare void @Py_XDECREF(ptr)
; 数値演算関連
declare ptr @PyNumber_Add(ptr, ptr)
declare ptr @PyNumber_Subtract(ptr, ptr)
declare ptr @PyNumber_Multiply(ptr, ptr)
declare ptr @PyNumber_TrueDivide(ptr, ptr)
declare ptr @PyNumber_FloorDivide(ptr, ptr)
declare ptr @PyNumber_Remainder(ptr, ptr)
declare ptr @PyNumber_Power(ptr, ptr, ptr)
declare ptr @PyNumber_Negative(ptr)
declare ptr @PyNumber_Positive(ptr)
declare ptr @PyNumber_Absolute(ptr)
declare ptr @PyNumber_Invert(ptr)
declare ptr @PyNumber_Lshift(ptr, ptr)
declare ptr @PyNumber_Rshift(ptr, ptr)
declare ptr @PyNumber_And(ptr, ptr)
declare ptr @PyNumber_Xor(ptr, ptr)
declare ptr @PyNumber_Or(ptr, ptr)
; その他のユーティリティ関数
declare void @print(ptr)
declare ptr @int2str(i32)
declare ptr @str2str(ptr)
declare ptr @create_string(ptr)
; ========== オブジェクトシステム構造体定義 ========== 
; 基本オブジェクト構造体
%struct.PyObject = type { i64, ptr }
; 可変長オブジェクト構造体
%struct.PyVarObject = type { %struct.PyObject, i64 }
; Unicode文字列オブジェクト構造体
%struct.PyUnicodeObject = type { %struct.PyObject, i64, i64, i32, ptr }
; リストオブジェクト構造体
%struct.PyListObject = type { %struct.PyObject, i64, ptr }
; 辞書エントリ構造体
%struct.PyDictEntry = type { ptr, ptr, i64 }
; 辞書オブジェクト構造体
%struct.PyDictObject = type { %struct.PyObject, i64, i64, ptr, i64, i64 }


define i32 @main(i32 %argc, i8** %argv) {
entry:
  call void @GC_init()
  call void @PyObject_InitSystem()
  %t0 = call ptr @PyUnicode_FromString(ptr @.str.0)
  call void @print(ptr %t0)
  %t2 = load ptr, ptr @Py_None
  call void @Py_INCREF(ptr %t2)
  call void @Py_DECREF(ptr %t2)
  %t3 = call ptr @PyList_New(i64 6)
  %t4 = call ptr @PyInt_FromI32(i32 0)
  call void @Py_INCREF(ptr %t4)
  call i32 @PyList_SetItem(ptr %t3, i64 0, ptr %t4)
  %t5 = call ptr @PyInt_FromI32(i32 1)
  call void @Py_INCREF(ptr %t5)
  call i32 @PyList_SetItem(ptr %t3, i64 1, ptr %t5)
  %t6 = call ptr @PyInt_FromI32(i32 2)
  call void @Py_INCREF(ptr %t6)
  call i32 @PyList_SetItem(ptr %t3, i64 2, ptr %t6)
  %t7 = call ptr @PyUnicode_FromString(ptr @.str.1)
  call void @Py_INCREF(ptr %t7)
  call i32 @PyList_SetItem(ptr %t3, i64 3, ptr %t7)
  %t8 = call ptr @PyUnicode_FromString(ptr @.str.2)
  call void @Py_INCREF(ptr %t8)
  call i32 @PyList_SetItem(ptr %t3, i64 4, ptr %t8)
  %t9 = call ptr @PyUnicode_FromString(ptr @.str.3)
  call void @Py_INCREF(ptr %t9)
  call i32 @PyList_SetItem(ptr %t3, i64 5, ptr %t9)
  %listvar = alloca ptr
  store ptr %t3, ptr %listvar
  call void @Py_INCREF(ptr %t3)
  %t10 = call ptr @PyDict_New()
  %t11 = call ptr @PyUnicode_FromString(ptr @.str.4)
  %t12 = call ptr @PyInt_FromI32(i32 1)
  call i32 @PyDict_SetItem(ptr %t10, ptr %t11, ptr %t12)
  %t13 = call ptr @PyUnicode_FromString(ptr @.str.5)
  %t14 = call ptr @PyInt_FromI32(i32 2)
  call i32 @PyDict_SetItem(ptr %t10, ptr %t13, ptr %t14)
  %t15 = call ptr @PyUnicode_FromString(ptr @.str.6)
  %t16 = call ptr @PyInt_FromI32(i32 3)
  call i32 @PyDict_SetItem(ptr %t10, ptr %t15, ptr %t16)
  %dictvar = alloca ptr
  store ptr %t10, ptr %dictvar
  call void @Py_INCREF(ptr %t10)
  call void @Py_INCREF(ptr %listvar)
  %t17 = call ptr @PyList_GetItem(ptr %listvar, i64 3)
  call void @Py_INCREF(ptr %t17)
  %t18 = call ptr @PyObject_Str(ptr %t17)
  call void @print(ptr %t18)
  %t19 = load ptr, ptr @Py_None
  call void @Py_INCREF(ptr %t19)
  call void @Py_DECREF(ptr %t19)
  call void @Py_INCREF(ptr %dictvar)
  %t20 = call ptr @PyUnicode_FromString(ptr @.str.7)
  %t21 = call ptr @PyDict_GetItem(ptr %dictvar, ptr %t20)
  call void @Py_DECREF(ptr %t20)
  ; キーが存在するかチェック
  call void @Py_INCREF(ptr %t21)
  %t22 = call ptr @PyObject_Str(ptr %t21)
  call void @print(ptr %t22)
  %t24 = load ptr, ptr @Py_None
  call void @Py_INCREF(ptr %t24)
  call void @Py_DECREF(ptr %t24)
  ret i32 0
}