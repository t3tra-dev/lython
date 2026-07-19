# nonlocal (R6 shared cells): counter, multi-name accumulation, augmented
# assignment in loops, rebinding a container, bool flag, live view through a
# non-declaring closure, and two-level nesting.
def counter() -> None:
    n: int = 0

    def inc() -> int:
        nonlocal n
        n += 1
        return n

    print(inc())
    print(inc())
    print(n)


def accumulate() -> None:
    total: int = 0
    words: str = ""

    def add(x: int, w: str) -> None:
        nonlocal total, words
        total += x
        words += w

    add(1, "a")
    add(2, "b")
    add(3, "c")
    print(total)
    print(words)


def loop_driven() -> None:
    n: int = 0

    def inc() -> None:
        nonlocal n
        n += 2

    for i in range(5):
        inc()
    print(n)


def rebind_container() -> None:
    xs: list[int] = [1]

    def swap() -> None:
        nonlocal xs
        xs = [7, 8, 9]

    swap()
    print(len(xs), xs[0], xs[2])


def flags() -> None:
    seen: bool = False

    def mark() -> None:
        nonlocal seen
        seen = True

    mark()
    if seen:
        print("seen")


def live_view() -> None:
    n: int = 10
    f = lambda: n * 2

    def bump() -> None:
        nonlocal n
        n = 21

    bump()
    print(f())


def deep() -> None:
    x: int = 1

    def mid() -> None:
        def inner() -> None:
            nonlocal x
            x = 42

        inner()

    mid()
    print(x)


counter()
accumulate()
loop_driven()
rebind_container()
flags()
live_view()
deep()
