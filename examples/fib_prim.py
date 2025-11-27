from lyrt import from_prim, native, to_prim
from lyrt.prim import i8  # pyright: ignore[reportMissingModuleSource]

p1 = to_prim(1, i8)
p2 = to_prim(2, i8)
p35 = to_prim(35, i8)


@native(gc="none")
def fib(n: i8) -> i8:
    if n <= p1:
        return n
    return fib(n - p1) + fib(n - p2)


print(from_prim(fib(p35)))
