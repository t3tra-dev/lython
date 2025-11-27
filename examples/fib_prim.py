from lyrt import from_prim, native, to_prim
from lyrt.prim import i8  # pyright: ignore[reportMissingModuleSource]

prim_one = to_prim(1, i8)
prim_two = to_prim(2, i8)


@native(gc="none")
def fib(n: i8) -> i8:
    if n <= prim_one:
        return n
    return fib(n - prim_one) + fib(n - prim_two)


print(from_prim(fib(to_prim(35, i8))))
