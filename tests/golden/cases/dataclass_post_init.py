# Why execution: the hook simply did not run, and nothing said so. The compiler
# exited 0 and printed the object's other fields correctly, so only the missing
# side effect -- and the field `__post_init__` was supposed to derive -- tells
# the two apart.
#
# CPython's dataclasses appends this call to the end of the generated
# __init__; a class that writes its own __init__ gets no generated one and so
# no call, which is the last case here.
from dataclasses import dataclass


@dataclass
class Announcing:
    x: int

    def __post_init__(self) -> None:
        print("post_init ran")


@dataclass
class Rectangle:
    width: int
    height: int
    area: int = 0

    def __post_init__(self) -> None:
        self.area = self.width * self.height


@dataclass
class Plain:
    x: int


@dataclass
class OwnInit:
    x: int = 0

    def __init__(self) -> None:
        self.x = 5

    def __post_init__(self) -> None:
        print("must not run")


def main() -> None:
    print(Announcing(1).x)
    print(Rectangle(3, 4).area)
    print(Plain(7).x)
    print(OwnInit().x)


main()
