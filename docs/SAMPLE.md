## サンプルコードから生成される IR

```python
@native
def add(x: i32, y: i32) -> i32:    # ---- TypedAST ----
    return x + y
```

```mlir
#dialect p
p.native.func @add(%arg0: !p.i32, %arg1: !p.i32) -> !p.i32 {
  %0 = p.add(%arg0, %arg1) : (!p.i32, !p.i32) -> !p.i32
  p.return %0 : !p.i32
}
```

`lower-py-to-llvm` 直後:

```llvm
define i32 @add(i32 %0, i32 %1) local_unnamed_addr {
entry:
  %2 = add nsw i32 %0, %1
  ret i32 %2
}
```
