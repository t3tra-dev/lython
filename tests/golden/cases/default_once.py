count: int = 0
def bump() -> int:
    global count
    count = count + 1
    return count
def f(x: int = bump()) -> int:
    return x
print(f())
print(f())
print(f(10))
print(count)
def g(s: str = "a" + "b") -> str:
    return s
print(g())
def take(xs: list[int] = [1, 2]) -> int:
    return len(xs)
print(take())
print(take())
