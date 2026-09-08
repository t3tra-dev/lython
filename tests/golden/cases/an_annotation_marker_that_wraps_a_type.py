# `ClassVar[int]` and `Final[int]` are `int` with a note on them: both mark how
# a binding may be USED, not what it holds. Neither had an annotation arm, and
# neither name could even be imported -- so `count: ClassVar[int] = 0`, the way
# a class counter is spelled, stopped at "unsupported import 'typing.ClassVar'".
#
# Why execution: a marker dropped from the wrong side would compile and then
# answer with the wrong storage -- a ClassVar read through an instance has to
# see what the CLASS holds, and the counters below are the only thing that says
# which one was read.
#
# ⛔ Neither is enforced. `Final` is an immutability claim this type system does
# not check, and a ClassVar assigned through an instance would bind an instance
# attribute in CPython; both are the deviation `@overload` and `@final` already
# carry, and both are what a CHECKER is for.
from typing import ClassVar, Final

MAX: Final[int] = 10
NAME: Final[str] = "cfg"


class Registry:
    count: ClassVar[int] = 0
    names: ClassVar["list[str]"] = []
    limit: ClassVar[int] = MAX

    def __init__(self, name: str) -> None:
        self.name: str = name
        Registry.count += 1
        Registry.names.append(name)

    def over(self) -> bool:
        return Registry.count > Registry.limit


class Sub(Registry):
    tag: ClassVar[str] = "sub"

    def describe(self) -> str:
        return self.tag + ":" + self.name


def main() -> None:
    a = Registry("a")
    b = Sub("b")
    print(Registry.count, Registry.names)
    print(a.count, a.over(), Registry.limit)
    print(b.describe(), Sub.tag)
    print(MAX + 1, NAME + "!", len(NAME))


main()
