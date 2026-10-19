# A generator keeps `str | None` and `dict | None` across a `yield` inside a
# `try` or a `with`. The yield returns from the resume function with the
# handler still armed, so the value was live on that block's exception edge
# only, and nothing released the frame's reference before the exit
# ("owned resource ... reaches function exit without release").
# Why run it: the value read after the resume, and the one left in an
# abandoned generator, are what a misplaced release would corrupt.
import contextlib


def kept(flag: bool):
    held: str | None = ("a" * 3 + str(flag)) if flag else None
    with contextlib.nullcontext():
        yield 1
    print("held", held)


def kept_in_try(flag: bool):
    table: dict[str, int] | None = {"a": 1} if flag else None
    try:
        yield 2
    finally:
        print("finally", table)
    print("table", table)


for x in kept(True):
    print("x", x)
for x in kept(False):
    print("x", x)
for x in kept_in_try(True):
    print("x", x)
abandoned = kept_in_try(False)
print("next", next(abandoned))
abandoned.close()


def quiet(flag: bool):
    held: str | None = ("b" * 5 + str(flag)) if flag else None
    with contextlib.nullcontext():
        yield 1
    if held is not None:
        yield len(held)


total = 0
for i in range(3000):
    for x in quiet(i % 2 == 0):
        total += x
print(total)
