def f(x: int = 2 * 3) -> int:
    return x
print(f())
print(f(10))
def g(s: str = "a" + "b") -> str:
    return s
print(g())
print(g("z"))
def h(v: float = 1.5 * 2.0) -> float:
    return v
print(h())
def take(xs: list[int] = [1, 2]) -> int:
    return len(xs)
print(take())
print(take([5, 6, 7]))
def opt(v: int | None = None) -> int:
    if v is None:
        return -1
    return v
print(opt())
print(opt(9))
def tup(t: tuple[int, int] = (1, 2)) -> int:
    return t[0] + t[1]
print(tup())
