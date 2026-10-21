BASE: int = 10


def make(tag: str) -> list[int]:
    print("eval", tag)
    return [BASE]


def f(xs: list[int] = make("f")) -> int:
    xs.append(1)
    return len(xs)


def g[T](x: T, xs: list[int] = [7]) -> int:
    xs.append(1)
    return len(xs)


class B:
    def add(self, into: list[int] = make("B.add"), *, more: list[int] = make("B.more")) -> int:
        into.append(1)
        more.append(2)
        return len(into) * 10 + len(more)
