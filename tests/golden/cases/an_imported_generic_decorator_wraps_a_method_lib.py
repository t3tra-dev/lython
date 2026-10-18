from typing import Callable


class skip:
    def __init__(self, reason: str) -> None:
        self.reason = reason

    def __call__[T](self, f: Callable[[T], None]) -> Callable[[T], None]:
        def wrapper(x: T) -> None:
            print("skipped:", self.reason)

        return wrapper


def logged[T](f: Callable[[T], None]) -> Callable[[T], None]:
    def wrapper(x: T) -> None:
        print("before")
        f(x)
        print("after")

    return wrapper
