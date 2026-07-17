from collections.abc import Generator

def inner():
    yield 10
    yield 20

def outer():
    yield 1
    yield from inner()
    yield 2

for v in outer():
    print(v)

def scored() -> Generator[int, int, int]:
    a = yield 100
    b = yield 200
    return a + b

def relay() -> Generator[int, int, None]:
    total = yield from scored()
    yield total

r = relay()
print(next(r))
print(r.send(3))
print(r.send(4))

def inner_cleanup():
    try:
        yield 1
    finally:
        print("inner cleanup")

def outer_cleanup():
    try:
        yield from inner_cleanup()
    finally:
        print("outer cleanup")

g = outer_cleanup()
print(next(g))
g.close()
print("closed")
