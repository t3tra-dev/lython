# What: a generator defined inside a function, capturing that function's
# locals -- an int, a str, a list, two at once beside a parameter, a name
# rebound after the generator was made (it sees the new value, as a cell),
# and one written through `nonlocal`.
#
# Every one was refused: the captures are entry arguments the callable type
# does not list, and the state machine built the resume clone's lanes from
# the callable type alone ("generator resume clone entry was not seeded as a
# primitive-i64 callable").
#
# Why execution: the rebinding cases are the cell semantics, which only the
# values show.
from typing import Iterator


def ints(n: int) -> list[int]:
    def gen() -> Iterator[int]:
        for i in range(3):
            yield i + n
    return list(gen())


def strs(p: str) -> list[str]:
    def gen() -> Iterator[str]:
        for s in ["a", "b"]:
            yield p + s
    return list(gen())


def lists(xs: list[int]) -> list[int]:
    def gen() -> Iterator[int]:
        for x in xs:
            yield x * 2
    return list(gen())


def two(a: int, b: str) -> list[str]:
    def gen(k: int) -> Iterator[str]:
        for i in range(k):
            yield b * (a + i)
    return list(gen(2))


def rebound_after(n: int) -> list[int]:
    def gen() -> Iterator[int]:
        yield n
        yield n + 1
    g = gen()
    n = 100
    return list(g)


def mutated_list() -> list[int]:
    xs = [1]
    def gen() -> Iterator[int]:
        for x in xs:
            yield x
    g = gen()
    xs.append(2)
    return list(g)


def interleaved(n: int) -> list[int]:
    def gen() -> Iterator[int]:
        yield n
        yield n
    g = gen()
    out = [next(g)]
    n = 7
    out.append(next(g))
    return out


print(ints(10), strs("p"), lists([1, 2]), two(1, "z"))
print(rebound_after(1), mutated_list(), interleaved(3))
def counter() -> list[int]:
    total = 0
    def gen() -> Iterator[int]:
        nonlocal total
        for i in range(3):
            total += i
            yield total
    out = list(gen())
    out.append(total)
    return out


def method_style() -> list[str]:
    prefix = "x"
    def gen(items: list[str]) -> Iterator[str]:
        for it in items:
            yield prefix + it
    first = list(gen(["a"]))
    prefix = "y"
    return first + list(gen(["b"]))


print(counter(), method_style())
