# WHAT: `b.f = v` where `f` is `str | None` / `int | None` / `Box | None` and
# `b` arrived as a parameter. The caller reads the field back afterwards, so
# what is checked is that the write crossed the frame -- and, for the payload
# that is more than one lane, that the absent arm still answers when asked for
# its shape.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: both failures are values.
# An optional field that stays on the inline lane splice compiles, runs, and
# prints the value the CALLEE last saw, which for a caller-owned object is the
# one it had before the call; and a `str` payload's second lane is a word read
# out of the entity, so the absent arm prints a length taken from whatever the
# stand-in header is followed by. Neither is visible before the value is read.
class Box:
    s: "str | None"
    i: "int | None"
    n: "Box | None"

    def __init__(self) -> None:
        self.s = None
        self.i = None
        self.n = None


def put_s(b: Box, v: "str | None") -> None:
    b.s = v


def put_i(b: Box, v: "int | None") -> None:
    b.i = v


def link(b: Box, v: "Box | None") -> None:
    b.n = v


def show(b: Box) -> str:
    s = b.s
    i = b.i
    n = b.n
    return (("-" if s is None else s + "/" + str(len(s)))
            + "|" + ("-" if i is None else str(i + 1))
            + "|" + ("-" if n is None else "node"))


b = Box()
print(show(b))
put_s(b, "hello")
put_i(b, 41)
print(show(b))
put_s(b, "")
put_i(b, -1)
print(show(b))
put_s(b, "日本語")
put_i(b, 1 << 40)
print(show(b))
link(b, Box())
print(show(b))
put_s(b, None)
put_i(b, None)
link(b, None)
print(show(b))

# The replaced payload has to be released on every rewrite, and the field has
# to still read back correctly after many of them.
k = 0
while k < 500:
    put_s(b, "x" * (k % 7))
    put_i(b, k)
    k += 1
print(show(b))
