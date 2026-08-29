# Classes and functions that construct a class defined LOWER in the file: the
# iterable/iterator pair, a factory function above its product, and a class
# whose method builds two later classes. The annotation always resolved -- only
# the members were missing -- so the refusal was
# "static type RangeIter does not provide manifest method '__init__'".
#
# Golden and not an emit test: the repair reorders when each class's contract
# and BODIES are emitted, and the thing that would go wrong quietly is a field
# read against a layout registered in a different order. Printing the values
# is what says the reorder moved nothing but the order.
class Counted:
    def __init__(self, stop: int) -> None:
        self.stop = stop

    def __iter__(self) -> "CountedIter":
        return CountedIter(self.stop)

    def boxed(self) -> "Box":
        return Box(self.stop, label(self.stop))


class CountedIter:
    def __init__(self, stop: int) -> None:
        self.i = 0
        self.stop = stop

    def __iter__(self) -> "CountedIter":
        return self

    def __next__(self) -> int:
        if self.i >= self.stop:
            raise StopIteration
        value = self.i
        self.i += 1
        return value


def label(n: int) -> str:
    return "n=" + str(n)


def first_box() -> "Box":
    return Box(1, "one")


class Box:
    def __init__(self, value: int, name: str) -> None:
        self.value = value
        self.name = name

    def __repr__(self) -> str:
        return "Box(" + str(self.value) + ", " + self.name + ")"


counted = Counted(4)
print([v for v in counted])
total = 0
for v in counted:
    total += v
print(total, len([v for v in counted]))
print(counted.boxed(), first_box())
print(counted.boxed().value + first_box().value)
