from typing import Callable

notes: list[str] = []


def skipIf[F](condition: bool, reason: str) -> Callable[[F], F]:
    def decorator(f: F) -> F:
        if condition:
            notes.append("skip: " + reason)
        return f

    return decorator


def tagged[T, F](tag: T) -> Callable[[F], F]:
    def decorator(f: F) -> F:
        notes.append("tag " + str(tag))
        return f

    return decorator


def times[T](k: int) -> Callable[[Callable[[T], int]], Callable[[T], int]]:
    def decorator(f: Callable[[T], int]) -> Callable[[T], int]:
        def wrapper(x: T) -> int:
            return f(x) * k

        return wrapper

    return decorator


class Lib:
    @skipIf(True, "lib reason")
    def check(self, n: int = 4) -> int:
        return n * 2

    @times(10)
    def size(self) -> int:
        return 3
