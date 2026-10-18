# WHAT: a method a decorator wraps runs on the instance it is called on:
# `self.helper()` in its body is a subclass's `helper` for a subclass's
# instance, whether the call names the subclass or its base.
#
# WHY THIS IS RUN: the body runs through what the decorator returned, called
# with the instance, so which `helper` it reaches is the instance's class at
# run time. The body assumed its own class and ran the base's.
from typing import Callable


def twice[T](f: Callable[[T], None]) -> Callable[[T], None]:
    def wrapper(x: T) -> None:
        f(x)
        f(x)

    return wrapper


class Base:
    def __init__(self, tag: str) -> None:
        self.tag = tag

    def helper(self) -> str:
        return "base " + self.tag

    @twice
    def run(self) -> None:
        print("run", self.helper())


class Sub(Base):
    def helper(self) -> str:
        return "sub " + self.tag


Base("b").run()
Sub("s").run()
items: list[Base] = [Base("x"), Sub("y")]
for item in items:
    item.run()
