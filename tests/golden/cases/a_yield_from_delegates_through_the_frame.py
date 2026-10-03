# `yield from` a generator the delegating body cannot take the body of -- a
# parameter, a recursive call, a method of the object itself -- resumes the
# delegate through its frame. The lines show PEP 380 holding: the return
# value arrives, a throw reaches the delegate's handler first, and closing the
# delegating generator runs the delegate's `finally`.
from typing import Generator, Iterator


class Tree:
    def __init__(self, v: int) -> None:
        self.v = v
        self.kids: list["Tree"] = []

    def walk(self) -> Iterator[int]:
        yield self.v
        for k in self.kids:
            yield from k.walk()


def count(n: int) -> Iterator[int]:
    yield n
    if n > 0:
        yield from count(n - 1)


def inner(n: int) -> Generator[int, None, str]:
    try:
        for i in range(n):
            yield i
    except ValueError:
        print("inner caught")
        yield -1
    finally:
        print("inner finally")
    return "r" + str(n)


def outer(g: Generator[int, None, str]) -> Generator[int, None, None]:
    r = yield from g
    print("outer got", r)


tree = Tree(1)
mid = Tree(2)
mid.kids.append(Tree(3))
tree.kids.append(mid)
tree.kids.append(Tree(4))
print(list(tree.walk()))
print(list(count(3)))
for v in outer(inner(2)):
    print(v)
o = outer(inner(3))
print(next(o))
print(o.throw(ValueError("x")))
o.close()
print("closed")
