# WHAT: an optional field written through a parameter. The caller has to see
# the write -- the callee holds its own copy of the receiver's lanes, so a
# store that lands in those lanes is invisible one frame up.
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

# rewritten many times: the old payload has to be released each time
i = 0
while i < 500:
    put_s(b, "x" * (i % 7))
    put_i(b, i)
    i += 1
print(show(b))
