# A subclass instance goes where its base is declared and stays the class it
# is: a function returning the subclass passed to a method's Callable
# parameter of the base, a lambda returning it under a Callable annotation of
# the base, a `U | None` yielded under `Iterator[M | None]`, base and
# subclass instances yielded directly by one generator, and a union field
# tested with isinstance against a class below one of its members.
import unittest
from typing import Callable, Iterator


class E:
    pass


class F(E):
    pass


def check(v: int) -> F | None:
    return F() if v < 0 else None


def make(v: int) -> F:
    return F()


class Runner:
    def run(self, f: Callable[[int], E | None]) -> str:
        return type(f(-1)).__name__ + type(f(1)).__name__

    def run_plain(self, f: Callable[[int], E]) -> str:
        return type(f(0)).__name__


class M:
    pass


class U(M):
    pass


def optional_subclass() -> Iterator[M | None]:
    u: U | None = U()
    yield u
    yield None


def both() :
    yield M()
    yield U()


class A:
    def __init__(self) -> None:
        self.tag = "a"


class AA(A):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.extra = n


class B:
    pass


class Holder:
    def __init__(self, k: int) -> None:
        self.one: A | B = AA(k) if k == 0 else (A() if k == 1 else B())

    def extra(self) -> int:
        if isinstance(self.one, AA):
            return self.one.extra + 10
        return -1


class ACallableOfTheBase(unittest.TestCase):
    def test_a_function_passed_to_a_method(self) -> None:
        self.assertEqual(Runner().run(check), "FNoneType")
        self.assertEqual(Runner().run_plain(make), "F")

    def test_a_lambda_under_the_annotation(self) -> None:
        f: Callable[[int], E] = lambda v: make(v)
        self.assertEqual(type(f(-1)).__name__, "F")
        g: Callable[[int], E | None] = lambda v: check(v)
        self.assertEqual(type(g(-1)).__name__, "F")


class AGeneratorOfTheBase(unittest.TestCase):
    def test_an_optional_subclass_yielded(self) -> None:
        self.assertEqual([type(x).__name__ for x in optional_subclass()],
                         ["U", "NoneType"])

    def test_base_and_subclass_yielded_directly(self) -> None:
        self.assertEqual([type(m).__name__ for m in both()], ["M", "U"])


class AFieldTestedForASubclass(unittest.TestCase):
    def test_a_class_below_one_member(self) -> None:
        self.assertEqual([Holder(k).extra() for k in range(3)], [10, -1, -1])
        h = Holder(0)
        if isinstance(h.one, AA):
            self.assertEqual(h.one.extra, 0)
        else:
            self.fail("an AA was held")


if __name__ == "__main__":
    unittest.main()
