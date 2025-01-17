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
  %t8 = call i32 @fib(i32 35)
  %t9 = call ptr @int2str(i32 %t8)
  call void @print(ptr %t9)
  %t10 = add i32 0, 0 ; discard return
  ret i32 0
}