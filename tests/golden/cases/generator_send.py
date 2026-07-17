from collections.abc import Generator

def accumulate(start: int) -> Generator[int, int, None]:
    total = start
    while True:
        got = yield total
        total = total + got

g = accumulate(10)
print(next(g))
print(g.send(5))
print(g.send(7))

def spread(base: int) -> Generator[int, int, None]:
    a = yield base
    b = yield base + 1
    yield a * 100 + b

h = spread(1)
print(next(h))
print(h.send(3))
print(h.send(4))

fresh = accumulate(0)
try:
    fresh.send(5)
except TypeError as e:
    print("TypeError:", e)
