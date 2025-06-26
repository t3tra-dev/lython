define ptr @greet(ptr %name) {
entry:
  %cst = load ptr, ptr @"str.Hello"
  %0   = call ptr @PyStr_Concat(ptr %cst, ptr %name)
  ret ptr %0
}
