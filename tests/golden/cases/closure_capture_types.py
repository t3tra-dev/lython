# Closure capture of each capture-eligible kind: int (lazy unboxed local),
# str, list, dict, user class instance, read through def and lambda.
class Point:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y


def reads() -> None:
    n: int = 41
    s: str = "hello"
    xs: list[int] = [1, 2, 3]
    d: dict[str, int] = {"a": 1, "b": 2}
    p: Point = Point(3, 4)

    def show() -> None:
        print(n + 1)
        print(s + "!")
        print(len(xs), xs[0], xs[-1])
        print(d["a"] + d["b"])
        print(p.x * p.y)

    show()
    f = lambda: n + s.count("l") + len(xs)
    print(f())


def deep() -> None:
    xs: list[int] = [7, 8]

    def mid() -> int:
        def inner() -> int:
            return xs[0] + xs[1]
        return inner()

    print(mid())


reads()
deep()
