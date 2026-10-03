# Handlers that read a local defined before their `try`, with a yield inside
# it: the `finally` after the suspension, an `except` the body raises into
# after resuming, and a `with` whose exit runs after two yields and on
# close(). Each runs after a resume, which only execution shows.
from collections.abc import Generator
from typing import Iterator


def worker() -> Generator[int, None, None]:
    held: list[int] = [1, 2, 3]
    try:
        yield len(held)
    finally:
        print(len(held))


def catches() -> Iterator[int]:
    m = [1, 2]
    try:
        yield 1
        raise ValueError("x")
    except ValueError:
        print(m)
        yield 2


class Ctx:
    def __enter__(self) -> int:
        return 5

    def __exit__(self, a: object, b: object, c: object) -> bool:
        print("exit")
        return False


def managed() -> Iterator[int]:
    with Ctx() as base:
        yield base
        yield base + 1


for v in worker():
    print(v)
print(list(catches()))
print(list(managed()))
g = managed()
print(next(g))
g.close()
print("closed")
