# WHAT: a method with type parameters of its own -- `def m[T](self, a: T,
# b: T)`, a bounded `T: SupportsEq`, two parameters, a return of `T`, a
# keyword-only parameter, a call through a subclass -- runs with T read as
# the type each call passes.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the body is specialized
# at each call site, and which `==` and which `repr` the specialization
# reaches (int, str, list, dict) shows only in what it prints. The method's
# body was emitted with T itself, which has no methods, so `a == b` was
# refused while the same def at module level compiled -- the shape of a
# statically typed assertEqual.
from typing import Protocol


class SupportsEq(Protocol):
    def __eq__(self, other: object) -> bool: ...


class Case:
    def assert_equal[T](self, first: T, second: T, msg: str | None = None) -> None:
        if not first == second:
            standard = "%r != %r" % (first, second)
            raise AssertionError(standard if msg is None else standard + " : " + msg)

    def same[T: SupportsEq](self, a: T, b: T, *, strict: bool = False) -> bool:
        if strict:
            return a == b and type(a) is type(b)
        return a == b

    def first[T](self, xs: list[T]) -> T:
        return xs[0]

    def pair[K, V](self, k: K, v: V) -> tuple[K, V]:
        return (k, v)


class Sub(Case):
    pass


c = Case()
c.assert_equal(1, 1)
c.assert_equal("a", "a")
c.assert_equal([1, 2], [1, 2])
for a, b in [(1, 2)]:
    try:
        c.assert_equal(a, b)
    except AssertionError as e:
        print(e)
try:
    c.assert_equal({"a": 1}, {"a": 2}, "dicts")
except AssertionError as e:
    print(e)
print(c.same(1, 1), c.same("a", "b"), c.same(1, 1, strict=True))
print(c.first([3, 4]) + 1, c.first(["x"]).upper(), c.pair(1, "a"), c.pair("k", [1.5]))
s = Sub()
print(s.first([[1], [2]]), s.same([1], [1]))
