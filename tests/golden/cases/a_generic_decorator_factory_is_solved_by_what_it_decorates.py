# WHAT: a decorator factory generic in a type parameter only its result
# names -- `def skipIf[F](cond, reason) -> Callable[[F], F]`, unittest's
# shape -- decorates a function, a method here and a method of the
# importing module, with the parameter solved by what it decorates, and a
# factory that also takes a generic argument is solved by both.
#
# WHY THIS IS RUN: the factories run when the module and the classes are
# made, and the methods run what they returned -- the defaults through the
# decorator that hands the function back, the product through the one that
# wraps it.
from a_generic_decorator_factory_is_solved_by_what_it_decorates_lib import (
    Lib,
    notes,
    skipIf,
    tagged,
    times,
)


class Case:
    @skipIf(False, "never")
    def test_one(self) -> None:
        print("one")

    @tagged(42)
    @skipIf(True, "always")
    def test_two(self, k: int = 1) -> int:
        return k + 1

    @times(3)
    def seven(self) -> int:
        return 7


@tagged("module")
def plain(x: int) -> int:
    return x * 10


print(notes)
c = Case()
c.test_one()
print(c.test_two(), c.test_two(5), c.seven(), plain(2))
lib = Lib()
print(lib.check(), lib.check(1), lib.size())
