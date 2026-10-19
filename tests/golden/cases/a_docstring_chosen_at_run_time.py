# Where the program's value decides whose docstring is read, the run time
# does: a class value answers the class it holds (a class's docstring is its
# own, never its base's), an instance its class's, a method its class's
# definition of it, a function value the function it was made for -- a
# lambda's is None, a decorated function's is what the decorator returned --
# and `getattr(x, name).__doc__` with the name computed whatever the name
# finds, which is how unittest describes a test. All were refused.
# Why run it: each answer depends on what the value holds at run time.
from collections.abc import Callable


def skip[T](reason: str) -> Callable[[Callable[[T], None]], Callable[[T], None]]:
    def decorator(item: Callable[[T], None]) -> Callable[[T], None]:
        def wrapper(self: T) -> None:
            print("skipped:", reason)
        return wrapper
    return decorator


def keep[T](item: Callable[[T], None]) -> Callable[[T], None]:
    return item


class Base:
    """Base."""

    def describe(self) -> None:
        """Base.describe."""

    def plain(self) -> None:
        pass


class Derived(Base):
    def describe(self) -> None:
        """Derived.describe."""

    @skip("not today")
    def test_skipped(self) -> None:
        """Hidden by the wrapper skip returns."""

    @keep
    def test_kept(self) -> None:
        """Kept: keep returns the function itself."""


class Leaf(Derived):
    """Leaf."""


def first() -> int:
    """First."""
    return 1


def second() -> int:
    return 2


def make(n: int) -> Callable[[], int]:
    def inner() -> int:
        """Made by make."""
        return n
    return inner


def traced(f: Callable[[], int]) -> Callable[[], int]:
    def wrapper() -> int:
        """The wrapper's."""
        return f()
    return wrapper


@traced
def decorated() -> int:
    """The decorated function's own."""
    return 3


classes: list[type[Base]] = [Base, Derived, Leaf]
print([c.__doc__ for c in classes])
print([c.describe.__doc__ for c in classes])
objects: list[Base] = [Base(), Derived(), Leaf()]
print([o.__doc__ for o in objects], [o.describe.__doc__ for o in objects])
functions: list[Callable[[], int]] = [first, second, make(1), lambda: 4, decorated]
print([f.__doc__ for f in functions])
bound: Callable[[], None] = Leaf().describe
print(bound.__doc__)
leaf = Leaf()
for name in ["describe", "plain", "test_skipped", "test_kept"]:
    print(name, repr(getattr(leaf, name).__doc__))
try:
    getattr(leaf, "missing").__doc__
except AttributeError as e:
    print("AttributeError:", e)
