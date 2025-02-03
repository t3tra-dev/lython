@.str.0 = private unnamed_addr constant [14 x i8] c"\48\65\6c\6c\6f\2c\20\77\6f\72\6c\64\21\00", align 1
attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" }
; ========== External runtime declarations ========== 
declare ptr @malloc(i64)
declare void @free(ptr)
declare void @print(ptr)

declare ptr @int2str(i32)
declare ptr @str2str(ptr)

declare ptr @create_string(ptr)

%struct.String = type { i64, ptr } ; // length + data pointer


define i32 @main(i32 %argc, i8** %argv) {
entry:
  %t0 = call ptr @create_string(ptr @.str.0)
  call void @print(ptr %t0)
  %t1 = add i32 0, 0 ; discard return
  ret i32 0
}