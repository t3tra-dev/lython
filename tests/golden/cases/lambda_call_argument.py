from typing import Callable


def apply(f: Callable[[int], int], v: int) -> int:
    return f(v)


def twice(f: Callable[[int], int], v: int) -> int:
    return f(f(v))


print(apply(lambda x: x + 1, 41))
print(twice(lambda x: x * 2, 10))
