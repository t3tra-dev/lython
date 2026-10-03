# Generators resumed where the function that created them is not known: one
# returned out of another function, one taken as a parameter, one closed and
# thrown into through a parameter. Each resume goes through the frame, which
# names its own body; the `finally` and `except` lines show the body really
# ran, which no earlier stage can see.
from typing import Generator, Iterator


def inner() -> Iterator[int]:
    yield 1
    yield 2


def outer() -> Iterator[int]:
    return inner()


for v in outer():
    print(v)


def words(prefix: str, n: int) -> Iterator[str]:
    try:
        for i in range(n):
            yield prefix + str(i)
    finally:
        print("closed", prefix)


def letters(text: str) -> Iterator[str]:
    for c in text:
        yield c


def take(source: Iterator[str], k: int) -> list[str]:
    out: list[str] = []
    for v in source:
        out.append(v)
        if len(out) == k:
            break
    return out


print(take(words("w", 5), 2))
print(take(letters("xyz"), 5))


def guarded() -> Generator[int, None, None]:
    try:
        yield 1
        yield 2
    except ValueError:
        print("caught")
        yield 10
    finally:
        print("fin")


def poke(g: Generator[int, None, None]) -> None:
    print(next(g))
    print(g.throw(ValueError("v")))
    g.close()


poke(guarded())


def shut(g: Generator[int, None, None]) -> None:
    print(next(g))
    g.close()


shut(guarded())
