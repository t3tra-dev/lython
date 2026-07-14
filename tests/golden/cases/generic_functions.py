from typing import Callable


def ident[T](x: T) -> T:
    return x


def first[A, B](a: A, b: B) -> A:
    return a


def rec[T](x: T, n: int) -> T:
    if n == 0:
        return x
    return rec(x, n - 1)


def apply(f: Callable[[int], int], v: int) -> int:
    return f(v)


print(ident(5))
print(ident("hello"))
print(first(7, "x"))
print(first("y", 8))
print(rec(99, 3))
print(rec("deep", 2))

g: Callable[[str], str] = ident
print(g("bound"))
print(apply(ident, 41))
