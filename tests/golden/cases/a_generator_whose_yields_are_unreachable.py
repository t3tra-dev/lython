# What: a function whose every yield is unreachable (`if False: yield`, a yield
# after `return`) is still a generator: its body runs at the first resume, not
# at the call, it yields nothing, and a value it returns reaches `yield from`.
# Why run: when the body runs is the answer.
from collections.abc import Iterator, Generator


def empty() -> Iterator[str]:
    print("ran")
    if False:
        yield "x"


def counted(n: int) -> Generator[int, None, str]:
    print("start", n)
    if False:
        yield n
    return "done" + str(n)


class C:
    def none(self) -> Iterator[int]:
        return
        yield 1


def outer() -> Generator[int, None, None]:
    r = yield from counted(3)
    print("got", r)


g = empty()
print("made")
print(list(g), list(C().none()))
for v in counted(2):
    print("never", v)
try:
    next(counted(5))
except StopIteration as e:
    print("stop")
print(list(outer()))
print(next(empty(), "default"))
