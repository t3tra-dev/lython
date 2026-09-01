# What: a method body is inlined at its call site, so a helper defined inside
# one has to be a real function of its own -- and only calling it shows that
# its `return` landed in its own body rather than in the caller's.
class Counter:
    def __init__(self, base: int) -> None:
        self.base = base

    def scaled(self, n: int) -> int:
        def times(k: int) -> int:
            return k * n

        return times(self.base)

    def summed(self, n: int) -> int:
        def rec(k: int) -> int:
            if k <= 0:
                return 0
            return k + rec(k - 1)

        return rec(n) + self.base

    def looped(self, n: int) -> int:
        def pick(k: int) -> int:
            for candidate in range(k):
                if candidate > 1:
                    return candidate
            return -1

        return pick(n)

    def with_a_lambda(self) -> int:
        f = lambda k: k + self.base
        return f(1)


c = Counter(3)
print(c.scaled(4), c.summed(3), c.looped(5), c.with_a_lambda())


class Wrapper:
    def build(self) -> "list[int]":
        def make(k: int) -> int:
            return k * k

        return [make(i) for i in range(4)]


print(Wrapper().build())
