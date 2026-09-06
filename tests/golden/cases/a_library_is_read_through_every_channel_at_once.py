# What: one program reading a library through every channel that had to learn
# to cross a file boundary, all at once. Each line has its own golden; this one
# exists because the fixes SHARE state -- the class attribute slots, the
# `__main__`-start initializers, the dispatcher's candidate scan and the
# declaration maps are one set of tables, and a combination is where a fix that
# is right alone stops being right.
#
# Every line needs running, and each one names which channel it is: the hook
# order, the attribute a subclass redeclares, the overridden method, the
# staticmethod, the property WRITE followed by the read, the generator whose
# yield only types under a guard, the enum members, and the module-level
# constants.
from typing import Iterator

import a_module_of_everything_a_library_declares
from a_module_of_everything_a_library_declares import Color, Shape


class Square(Shape, tag="sq"):
    kind: str = "square"

    @property
    def size(self) -> int:
        return self._size * 10

    @size.setter
    def size(self, n: int) -> None:
        self._size = n + 1

    @staticmethod
    def sides() -> int:
        return 4

    def name(self) -> str:
        return "square"


def guarded(xs: list[Shape]) -> Iterator[str]:
    for x in xs:
        if isinstance(x, Square):
            yield x.name()


shapes: list[Shape] = [Shape(1), Square(2)]
for s in shapes:
    s.size = 5

print("hook", a_module_of_everything_a_library_declares.Shape.seen)
print("attribute", [s.kind for s in shapes])
print("method", [s.name() for s in shapes])
print("static", [s.sides() for s in shapes])
print("property", [s.size for s in shapes])
print("guarded", list(guarded(shapes)))
print("enum", Color.RED.name, Color.BLUE.value)
print("constants", a_module_of_everything_a_library_declares.NAMES, a_module_of_everything_a_library_declares.SCALE(3))
