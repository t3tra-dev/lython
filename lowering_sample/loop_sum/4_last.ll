define i32 @sum_list(ptr %xs, i64 %n) {
entry:
  %sum = alloca i32
  store i32 0, ptr %sum
  br label %loop

loop:                                            ; preds = %body, %entry
  %i = phi i64 [ 0, %entry ], [ %i.next, %body ]
  %cmp = icmp slt i64 %i, %n
  br i1 %cmp, label %body, label %exit

body:                                            ; preds = %loop
  %elem_ptr = getelementptr i32, ptr %xs, i64 %i
  %x = load i32, ptr %elem_ptr
  %s_old = load i32, ptr %sum
  %s_new = add nsw i32 %s_old, %x
  store i32 %s_new, ptr %sum
  %i.next = add i64 %i, 1
  br label %loop

exit:                                            ; preds = %loop
  %ret = load i32, ptr %sum
  ret i32 %ret
}
