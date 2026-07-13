# Default arguments: literal defaults of every kind (including bytes and
# None in optional unions) plus expression defaults, which compile into a
# zero-argument provider function evaluated at each call that omits the
# argument (a documented deviation from CPython's once-per-def evaluation).
def a(x: int = 1) -> int:
    return x

def b(s: str = "hi") -> str:
    return s

def c(f: float = 1.5) -> float:
    return f

def d(v: bool = True) -> bool:
    return v

def e(n: int | None = None) -> int:
    if n is None:
        return -1
    return n

def g(bs: bytes = b"raw") -> bytes:
    return bs

print(a(), a(2))
print(b(), b("yo"))
print(c(), c(2.5))
print(d(), d(False))
print(e(), e(5))
print(g(), g(b"x"))
class Point:
    def __init__(self, x: int = 0, y: int = 0) -> None:
        self.x = x
        self.y = y


class Canvas:
    def __init__(self) -> None:
        self.total = 0

    def mark(self, p: Point = Point(3, 4)) -> int:
        self.total = self.total + p.x + p.y
        return self.total


def norm(p: Point = Point(1, 2)) -> int:
    return p.x + p.y


def tags(items: list[int] = [10, 20]) -> int:
    n = 0
    for item in items:
        n = n + item
    return n


def greet(prefix: str = "hi " + "there") -> str:
    return prefix


print(norm())
print(norm(Point(5, 6)))
c = Canvas()
print(c.mark())
print(c.mark(Point(1, 1)))
print(tags())
print(tags([1]))
print(greet())
