attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" }
; ========== External runtime declarations ========== 
declare ptr @malloc(i64)
declare void @free(ptr)
declare i32 @puts(ptr)
declare void @print_i32(i32)
declare void @print_string(ptr)
declare ptr @create_string(ptr)

%struct.String = type { i64, ptr } ; // length + data pointer


; Function definition: fib
define i32 @fib(i32 %n) #0 {
entry:
  %t0 = add i32 0, 1
  %t1 = icmp sle i32 %n, %t0
  %t2 = zext i1 %t1 to i32
  %t3 = icmp ne i32 %t2, 0
  br i1 %t3, label %if.then.0, label %if.else.1
if.then.0:
  ret i32 %n
  br label %if.end.2
if.else.1:
  br label %if.end.2
if.end.2:
  %t4 = add i32 0, 1
  %t5 = sub i32 %n, %t4
  %t6 = call i32 @fib(i32 %t5)
  %t7 = add i32 0, 2
  %t8 = sub i32 %n, %t7
  %t9 = call i32 @fib(i32 %t8)
  %t10 = add i32 %t6, %t9
  ret i32 %t10
}

define i32 @main(i32 %argc, i8** %argv) {
entry:
  %t11 = add i32 0, 35
  %t12 = call i32 @fib(i32 %t11)
  call void @print_i32(i32 %t12)
  %t13 = add i32 0, 0
  ret i32 0
}