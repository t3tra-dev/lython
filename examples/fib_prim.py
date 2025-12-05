from lyrt import from_prim, native
from lyrt.prim import Int

p1 = Int[32](1)
p2 = Int[32](2)
p35 = Int[32](35)


@native(gc="none")
def fib(n: Int[32]) -> Int[32]:
    if n <= p1:
        return n
    return fib(n - p1) + fib(n - p2)


print(from_prim(fib(p35)))
