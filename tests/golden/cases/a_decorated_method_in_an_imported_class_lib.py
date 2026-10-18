from typing import Callable

registry: list[str] = []


def register[T](f: Callable[[T], None]) -> Callable[[T], None]:
    registry.append("registered")
    return f


def twice[T](f: Callable[[T], None]) -> Callable[[T], None]:
    def wrapper(x: T) -> None:
        f(x)
        f(x)

    return wrapper


class Base:
    def __init__(self, tag: str) -> None:
        self.tag = tag

    def helper(self) -> str:
        return "helper:" + self.tag

    @twice
    def run(self) -> None:
        print("run", self.helper())

    @register
    def setup(self) -> None:
        print("setup", self.tag)

    def plain(self) -> None:
        print("plain", self.tag)
