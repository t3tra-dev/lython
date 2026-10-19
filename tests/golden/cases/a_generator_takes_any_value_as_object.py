# A generator whose parameter is `object` is handed values of other classes
# -- a str, an int, a list, an instance -- and reads them after
# it has been suspended. The argument was persisted into the frame as the
# lanes it arrived with, which are not the box the lane is ("generator
# argument 0 (builtins.object) has no matching physical span to persist"),
# and a box that did arrive was kept by pointer with only its contents
# retained. The frame now keeps a box of its own.
# Why run it: the value has to outlive the caller's reference, through a
# suspension, and the frame's box has to be dropped exactly once -- including
# by a generator that is abandoned before it finishes.


class A:
    def __repr__(self) -> str:
        return "A()"


def describe(first: object, second: object):
    yield 1
    print(type(first).__name__, type(second).__name__, first is second)
    yield 2


def quiet(first: object, second: object):
    yield 1
    yield 3 if first is second else 2


for value in describe("text", [1, 2]):
    print(value)
for value in describe(3, A()):
    print(value)
token = A()
for value in describe(token, token):
    print(value)
abandoned = describe("left", "behind")
print(next(abandoned))
total = 0
for i in range(2000):
    for value in quiet("x" * 10 + str(i), A()):
        total += value
    for value in quiet(token, token):
        total += value
print(total)
