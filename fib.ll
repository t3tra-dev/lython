@.str = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define i32 @fib(i32 noundef %0) #0 {
  ; 基本ケースの直接比較と返却
  %2 = icmp sle i32 %0, 1
  br i1 %2, label %base, label %recurse

base:
  ret i32 %0

recurse:
  %3 = sub nsw i32 %0, 1
  %4 = call i32 @fib(i32 noundef %3)
  %5 = sub nsw i32 %0, 2
  %6 = call i32 @fib(i32 noundef %5)
  %7 = add nsw i32 %4, %6
  ret i32 %7
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define i32 @main() #0 {
  %1 = call i32 @fib(i32 noundef 35)
  %2 = call i32 (ptr, ...) @printf(ptr noundef @.str, i32 noundef %1)
  ret i32 0
}

declare i32 @printf(ptr noundef, ...) #1
