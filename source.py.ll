@.str.0 = private unnamed_addr constant [14 x i8] c"\48\65\6c\6c\6f\2c\20\77\6f\72\6c\64\21\00", align 1
@.str.1 = private unnamed_addr constant [43 x i8] c"\6d\75\6c\74\69\62\69\74\65\20\63\68\61\72\61\63\74\65\72\3a\20\e3\81\82\e3\81\84\e3\81\86\e3\81\88\e3\81\8a\2c\20\f0\9f\90\8d\00", align 1
attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" }
; ========== External runtime declarations ========== 
declare ptr @malloc(i64)
declare void @free(ptr)
declare void @print(ptr)

declare ptr @int2str(i32)
declare ptr @str2str(ptr)

declare ptr @create_string(ptr)

%struct.String = type { i64, ptr } ; // length + data pointer


; Function definition: fib
define i32 @fib(i32 %n) #0 {
entry:
  %t0 = icmp sle i32 %n, 1
  %t1 = zext i1 %t0 to i32
  %t2 = icmp ne i32 %t1, 0
  br i1 %t2, label %if.then.0, label %if.else.1
if.then.0:
  ret i32 %n
  br label %if.end.2
if.else.1:
  br label %if.end.2
if.end.2:
  %t3 = sub i32 %n, 1
  %t4 = call i32 @fib(i32 %t3)
  %t5 = sub i32 %n, 2
  %t6 = call i32 @fib(i32 %t5)
  %t7 = add i32 %t4, %t6
  ret i32 %t7
}

define i32 @main(i32 %argc, i8** %argv) {
entry:
  %t8 = call ptr @create_string(ptr @.str.0)
  call void @print(ptr %t8)
  %t9 = add i32 0, 0 ; discard return
  %t10 = call ptr @create_string(ptr @.str.1)
  call void @print(ptr %t10)
  %t11 = add i32 0, 0 ; discard return
  %t12 = call i32 @fib(i32 30)
  %t13 = call ptr @int2str(i32 %t12)
  call void @print(ptr %t13)
  %t14 = add i32 0, 0 ; discard return
  ret i32 0
}