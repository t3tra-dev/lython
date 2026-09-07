# What: a generator whose yield expression only types under a guard. Every
# yield is typed by a walk with no flow facts, so `x.tag()` inside
# `if isinstance(x, B)` inferred None -- `tag` is not on the base -- and the
# generator was refused as "annotated Iterator[str] but yields literal<None>",
# a sentence about the annotation for a program whose annotation is right.
# `yield v * 2` over an `int | None` was the shape that worked, and only
# because `__mul__` still infers something; a method or attribute lookup fails
# outright.
#
# ⭐ AND THE ITERABLE, not only the yield. A guard over what the loop WALKS is
# the same fact, and the walk did not carry it two ways:
#
#     if self.rows is None:
#         return
#     for v in self.rows:
#         yield v * 2
#     # annotated Iterator[int] but yields builtins.object
#
# -- because the fact was applied to the guard's BODY only, so the code after
# an early `return` saw the union; and because the subject was a field PATH,
# which the walk's guard reader took only as a bare name. Both halves are here:
# the early-return spelling, the `if ... is not None:` spelling with the loop
# inside it, and the same through a local.
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



class Rows:
    def __init__(self) -> None:
        self.rows: "list[int] | None" = None

    def after_an_early_return(self) -> Iterator[int]:
        if self.rows is None:
            return
        for v in self.rows:
            yield v * 2

    def inside_the_guard(self) -> Iterator[int]:
        if self.rows is not None:
            for v in self.rows:
                yield v + 1

    def through_a_local(self) -> Iterator[int]:
        rows = self.rows
        if rows is None:
            return
        for v in rows:
            yield v * 3


holder = Rows()
print(list(holder.after_an_early_return()), list(holder.inside_the_guard()))
print(list(holder.through_a_local()))
holder.rows = [1, 2, 3]
print(list(holder.after_an_early_return()), list(holder.inside_the_guard()))
print(list(holder.through_a_local()), sum(holder.after_an_early_return()))
