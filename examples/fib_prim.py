from lyrt import from_prim, native, to_prim
from lyrt.prim import i32  # pyright: ignore[reportMissingModuleSource]

p1 = to_prim(1, i32)
p2 = to_prim(2, i32)
p35 = to_prim(35, i32)


@native(gc="none")
def fib(n: i32) -> i32:
    if n <= p1:
        return n
    return fib(n - p1) + fib(n - p2)


print(from_prim(fib(p35)))
