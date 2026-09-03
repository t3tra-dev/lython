# `__name__` on a function, which had no arm at all: it type-checked and then
# died in the lowering as "attr.get object type has no class schema", while
# `C.__name__` beside it has been a compile-time string since it was written.


def compute(a: int, b: int) -> int:
    return a + b


class Shape:
    def area(self) -> int:
        return 1


print(compute.__name__, Shape.area.__name__, Shape.__name__)
print(compute.__name__ + "/" + str(compute(1, 2)))
print([f for f in [compute.__name__, Shape.area.__name__]])
