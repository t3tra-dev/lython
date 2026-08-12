# Why execution: the values are what the dispatch has to produce. `-v` over a
# class that defines __neg__ died in the lowering as "runtime manifest has no
# V.__neg__ method" (py.neg resolves its target against the manifest, and a
# source class is not in it) and `abs(v)` reached the builtin's numeric
# overloads as "!py.overload<...> is not callable with these arguments" --
# while __len__ and __bool__ on the same class both dispatch.
from typing import Iterator


class Vec:
    def __init__(self, x: int) -> None:
        self.x = x

    def __neg__(self) -> "Vec":
        return Vec(-self.x)

    def __pos__(self) -> "Vec":
        return Vec(self.x)

    def __invert__(self) -> "Vec":
        return Vec(~self.x)

    def __abs__(self) -> int:
        return abs(self.x)

    def __len__(self) -> int:
        return self.x

    def __bool__(self) -> bool:
        return self.x != 0

    def __int__(self) -> int:
        return self.x

    def __float__(self) -> float:
        return float(self.x)

    def __round__(self) -> int:
        return self.x

    def __reversed__(self) -> Iterator[int]:
        return iter([self.x, self.x + 1])


def main() -> None:
    v = Vec(3)
    print((-v).x, (+v).x, (~v).x, abs(Vec(-4)), abs(v))
    print(len(v), bool(v), bool(Vec(0)))
    if Vec(1):
        print("truthy")
    print(abs(-3), abs(-3.5), abs(3))
    print((-(-v)).x)
    # the one-argument builtins whose whole job is to call a dunder
    print(int(v), float(v), round(v), list(reversed(v)))


main()
