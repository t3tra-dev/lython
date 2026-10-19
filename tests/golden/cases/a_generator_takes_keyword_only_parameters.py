# WHAT: a generator function, a generator method and a nested generator take
# keyword-only parameters -- given or left to their defaults, of every kind of
# value, beside `*args` -- and hold them across their yields, including in a
# generator dropped or closed before it is exhausted.
#
# WHY THIS IS RUN: what a generator yields is produced on each resume, from
# the arguments its frame keeps.
from collections.abc import Iterator


class Box:
    def __init__(self, v: int) -> None:
        self.v = v

    def items(self, n: int, *, step: int = 1, label: str = "b") -> Iterator[str]:
        for i in range(0, n, step):
            yield label + str(self.v + i)


def pair(a: int, *, by: int) -> Iterator[int]:
    yield a + by
    yield a * by


def mixed(*args: int, scale: float = 1.0, tag: str | None = None,
          extra: list[int] | None = None, flag: bool = False,
          box: Box | None = None) -> Iterator[str]:
    for a in args:
        out = str(a * scale)
        if tag is not None:
            out = tag + out
        if extra:
            out += str(extra)
        if flag:
            out += "!"
        if box is not None:
            out += "#" + str(box.v)
        yield out


def outer(base: int) -> list[int]:
    def gen(k: int, *, by: int = 2) -> Iterator[int]:
        for i in range(k):
            yield base + i * by
    return list(gen(3)) + list(gen(2, by=10))


print(list(pair(2, by=3)))
print(list(Box(5).items(3)), list(Box(1).items(5, step=2, label="x")))
print(list(mixed(1, 2)))
print(list(mixed(3, scale=0.5, tag="t", extra=[9], flag=True, box=Box(7))))
print(outer(100))
g = mixed(4, 5, tag="p")
print(next(g))
g = mixed()
h = Box(2).items(4, label="q")
print(next(h), next(h))
h.close()
print("done")
