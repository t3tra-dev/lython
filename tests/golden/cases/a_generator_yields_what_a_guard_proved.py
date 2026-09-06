# What: a generator whose yield expression only types under a guard. Every
# yield is typed by a walk with no flow facts, so `x.tag()` inside
# `if isinstance(x, B)` inferred None -- `tag` is not on the base -- and the
# generator was refused as "annotated Iterator[str] but yields literal<None>",
# a sentence about the annotation for a program whose annotation is right.
# `yield v * 2` over an `int | None` was the shape that worked, and only
# because `__mul__` still infers something; a method or attribute lookup fails
# outright.
#
# Running it is what shows the guard held: the unguarded elements must produce
# nothing at all, and the guarded ones the subclass's answer. The `else` arm is
# here because the narrowing must NOT stand there, and the `and` chain because
# a guard with a second condition is the ordinary spelling.
from typing import Iterator


class Shape:
    pass


class Named(Shape):
    def tag(self) -> str:
        return "named"


def tags(xs: list[Shape]) -> Iterator[str]:
    for x in xs:
        if isinstance(x, Named):
            yield x.tag()


def uppers(xs: "list[str | None]") -> Iterator[str]:
    for v in xs:
        if v is not None:
            yield v.upper()


def chained(xs: "list[str | None]") -> Iterator[str]:
    for v in xs:
        if v is not None and len(v) > 0:
            yield v.upper()


def both_arms(xs: "list[str | None]") -> Iterator[str]:
    for v in xs:
        if v is not None:
            yield v.upper()
        else:
            yield "none"


print("isinstance", list(tags([Shape(), Named(), Shape()])))
print("is not None", list(uppers(["a", None, "b"])))
print("chained", list(chained(["a", None, "", "b"])))
print("both arms", list(both_arms(["a", None])))
