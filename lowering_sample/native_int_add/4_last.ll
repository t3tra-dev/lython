define i32 @add(i32 %0, i32 %1) {
entry:
  %2 = add nsw i32 %0, %1
  ret i32 %2
}
