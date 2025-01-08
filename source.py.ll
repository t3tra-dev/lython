@.str.0 = private unnamed_addr constant [14 x i8] c"Hello, World!\00", align 1
declare i32 @puts(i8* nocapture readonly) local_unnamed_addr

define i32 @main(i32 %argc, i8** %argv) {
entry:
  %puts = call i32 @puts(i8* getelementptr inbounds ([14 x i8], [14 x i8]* @.str.0, i64 0, i64 0))
  %0 = add i32 1, 1
  ret i32 0
}