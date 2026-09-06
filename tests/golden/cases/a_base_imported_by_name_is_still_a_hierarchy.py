# What: a base class brought in with `from module import Shape` and subclassed
# here. Every one of these lines answered the BASE's body before -- silently,
# with no diagnostic -- because the hierarchy was recorded under the SPELLING
# the source wrote ("Shape") while every question about it is asked under the
# contract name ("a_module_of_importable_shapes.Shape"). The dotted spelling of
# the same program was already right, so this is what the two disagreed about.
#
# Six spellings of the one question, because the dispatch has six doors and the
# hierarchy is what all six read: a call, a base method calling an override, a
# dunder, a property, a class attribute, a method read as a value, and a
# staticmethod. Each needs a base-typed receiver, which the list provides.
from a_module_of_importable_shapes import Shape


class Local(Shape):
    kind: str = "local"

    def name(self) -> str:
        return "local"

    def __len__(self) -> int:
        return 7

    @property
    def area(self) -> int:
        return 9

    @staticmethod
    def sides() -> int:
        return 4


items: list[Shape] = [Shape(2), Local(6)]

print("call", [i.name() for i in items])
print("through the base", [i.describe() for i in items])
print("dunder", [len(i) for i in items])
print("property", [i.area for i in items])
print("attribute", [i.kind for i in items])
print("static", [i.sides() for i in items])

values: list[str] = []
for i in items:
    m = i.name
    values.append(m())
print("as a value", values)
