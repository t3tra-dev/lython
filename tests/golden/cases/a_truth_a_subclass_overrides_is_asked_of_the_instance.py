# WHAT: a truth test on a value whose static class has a subclass that adds
# or overrides `__bool__` or `__len__` answers with the class the instance
# was made as -- through a function returning the base, a `list[Base]`, and
# a `while` condition.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: which subclass is behind
# the reference is a run-time fact. Each test here was refused ("'__bool__'
# is overridden by a subclass of 'A', so this call cannot be resolved from
# the static type of the receiver").
class A:
    pass


class B(A):
    def __bool__(self) -> bool:
        return False


class C(A):
    def __len__(self) -> int:
        return 0


def mk(n: int) -> A:
    if n == 0:
        return B()
    if n == 1:
        return C()
    return A()


for n in range(3):
    x = mk(n)
    print(n, bool(x), "T" if x else "F", not x)


class P:
    def __bool__(self) -> bool:
        return True


class Q(P):
    def __bool__(self) -> bool:
        return False


ps: list[P] = [P(), Q(), P()]
print([bool(p) for p in ps], [type(p).__name__ for p in ps if not p])


class Countdown:
    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n


class Stuck(Countdown):
    def __len__(self) -> int:
        return 0


def drain(c: Countdown) -> int:
    steps = 0
    while c:
        c.n -= 1
        steps += 1
    return steps


print(drain(Countdown(3)), drain(Stuck(3)))
