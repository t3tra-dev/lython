# WHAT: generic decorator factories that return another factory's result
# unapplied (`skipUnless` returning `skipIf(...)`), and one that returns a
# generic identity function as the decorator; applied to methods of two
# classes, with the conditions that skip and that do not.
#
# WHY THIS IS RUN: which method body runs is decided by the conditions at
# run time. The returned call's type parameter was named only by its result,
# which the factory's own return annotation determines, and it was refused
# ("generic in a type parameter no argument names").
from collections.abc import Callable


def _id[T](obj: T) -> T:
    return obj


def skip[T](reason: str) -> Callable[[Callable[[T], None]], Callable[[T], None]]:
    def decorator(f: Callable[[T], None]) -> Callable[[T], None]:
        def skipped(self: T) -> None:
            print("skipped:", reason)
        return skipped
    return decorator


def skipIf[T](condition: bool, reason: str) -> Callable[[Callable[[T], None]], Callable[[T], None]]:
    if condition:
        return skip(reason)
    return _id


def skipUnless[T](condition: bool, reason: str) -> Callable[[Callable[[T], None]], Callable[[T], None]]:
    return skipIf(not condition, reason)


class A:
    @skipUnless(False, "why")
    def first(self) -> None:
        print("first ran")

    @skipUnless(True, "never")
    def second(self) -> None:
        print("second ran")


class B:
    @skipIf(True, "b skipped")
    def only(self) -> None:
        print("only ran")


A().first()
A().second()
B().only()
