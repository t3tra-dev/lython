# A generic class's specialization is an internal contract named
# "<class>$specN", and that name must never surface: the synthesized dataclass
# repr, the exception name in a traceback, and a class object reached through an
# instantiation all have to read as the class the source wrote.
from dataclasses import dataclass


@dataclass
class Point[T]:
    x: T
    y: T


# A dataclass has no __init__ of its own, so its annotated fields ARE the
# constructor whose parameter types determine T.
print(Point(1, 2))
print(Point("a", "b"))

located: Point[int] = Point(3, 4)
print(located.x, located.y)
print(located == Point(3, 4), located == Point(3, 5))


class Tagged[T]:
    label = "tag"

    def __init__(self, value: T) -> None:
        self.value = value

    def get(self) -> T:
        return self.value


# `C[int]` is the instantiation's class object, so its class attributes read
# through it.
print(Tagged[int].label, Tagged[str].label)
print(Tagged(5).get(), Tagged("six").get())
