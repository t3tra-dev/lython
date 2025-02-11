@.str.0 = private unnamed_addr constant [14 x i8] c"\48\65\6c\6c\6f\2c\20\77\6f\72\6c\64\21\00", align 1
@.str.1 = private unnamed_addr constant [2 x i8] c"\61\00", align 1
@.str.2 = private unnamed_addr constant [2 x i8] c"\62\00", align 1
@.str.3 = private unnamed_addr constant [2 x i8] c"\63\00", align 1
@.str.4 = private unnamed_addr constant [5 x i8] c"\6b\65\79\31\00", align 1
@.str.5 = private unnamed_addr constant [5 x i8] c"\6b\65\79\32\00", align 1
@.str.6 = private unnamed_addr constant [5 x i8] c"\6b\65\79\33\00", align 1
@.str.7 = private unnamed_addr constant [5 x i8] c"\6b\65\79\31\00", align 1
attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" }
; ========== External runtime declarations ========== 
declare void @print(ptr)

declare ptr @int2str(i32)
declare ptr @str2str(ptr)

declare ptr @create_string(ptr)

declare ptr @PyInt_FromI32(i32)
declare i32 @PyInt_AsI32(ptr)
declare ptr @PyDict_New(i32)
declare i32 @PyDict_SetItem(ptr, ptr, ptr)
declare ptr @PyDict_GetItem(ptr, ptr)
declare ptr @PyList_New(i32)
declare i32 @PyList_Append(ptr, ptr)
declare ptr @PyList_GetItem(ptr, i32)
%struct.String = type { i64, ptr } ; // length + data pointer


define i32 @main(i32 %argc, i8** %argv) {
entry:
  %t0 = call ptr @create_string(ptr @.str.0)
  call void @print(ptr %t0)
  %t1 = add i32 0, 0 ; discard return
  %t2 = call ptr @PyList_New(i32 8)
  %t3 = call ptr @PyInt_FromI32(i32 0)
  call i32 @PyList_Append(ptr %t2, ptr %t3)
  %t4 = call ptr @PyInt_FromI32(i32 1)
  call i32 @PyList_Append(ptr %t2, ptr %t4)
  %t5 = call ptr @PyInt_FromI32(i32 2)
  call i32 @PyList_Append(ptr %t2, ptr %t5)
  %t6 = call ptr @create_string(ptr @.str.1)
  call i32 @PyList_Append(ptr %t2, ptr %t6)
  %t7 = call ptr @create_string(ptr @.str.2)
  call i32 @PyList_Append(ptr %t2, ptr %t7)
  %t8 = call ptr @create_string(ptr @.str.3)
  call i32 @PyList_Append(ptr %t2, ptr %t8)
  %listvar = bitcast ptr %t2 to ptr ; assignment to listvar
  %t9 = call ptr @PyDict_New(i32 8)
  %t10 = call ptr @create_string(ptr @.str.4)
  %t12 = call ptr @PyInt_FromI32(i32 1)
  call i32 @PyDict_SetItem(ptr %t9, ptr %t10, ptr %t12)
  %t13 = call ptr @create_string(ptr @.str.5)
  %t15 = call ptr @PyInt_FromI32(i32 2)
  call i32 @PyDict_SetItem(ptr %t9, ptr %t13, ptr %t15)
  %t16 = call ptr @create_string(ptr @.str.6)
  %t18 = call ptr @PyInt_FromI32(i32 3)
  call i32 @PyDict_SetItem(ptr %t9, ptr %t16, ptr %t18)
  %dictvar = bitcast ptr %t9 to ptr ; assignment to dictvar
  %t19 = call ptr @PyList_GetItem(ptr %listvar, i32 3)
  call void @print(ptr %t19)
  %t20 = add i32 0, 0 ; discard return
  %t21 = call ptr @PyDict_GetItem(ptr %dictvar, ptr @.str.7)
  %t23 = call i32 @PyInt_AsI32(ptr %t21)
  %t22 = call ptr @int2str(i32 %t23)
  call void @print(ptr %t22)
  %t24 = add i32 0, 0 ; discard return
  ret i32 0
}