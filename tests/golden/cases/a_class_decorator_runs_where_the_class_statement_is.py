# WHAT: class decorators -- a function taking and returning the class, a
# factory's result, two stacked (innermost first) -- run where the class
# statement runs, between the module's other statements, and the name still
# names the class afterwards.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: a class decorator was
# refused outright; what it now does is a call at the statement's place in
# the module's flow, and that order is seen only in what is printed.
from collections.abc import Callable

registry: list[str] = []


def register[T](cls: type[T]) -> type[T]:
    registry.append(cls.__name__)
    print("register", cls.__name__)
    return cls


def tagged[T](tag: str) -> Callable[[type[T]], type[T]]:
    def apply(cls: type[T]) -> type[T]:
        registry.append(tag + ":" + cls.__name__)
        print("tag", tag, cls.__name__)
        return cls
    return apply


print("before")


@register
class Plain:
    def hello(self) -> str:
        return "plain"


print("between")


@tagged("outer")
@register
class Stacked:
    pass


print(registry)
print(Plain().hello(), Stacked.__name__)
