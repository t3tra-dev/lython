# A list held in a field or passed as a parameter grows by `append` when the
# argument is computed through a class value -- a class attribute read off
# `cls` -- in a method called in a loop, beside appends in the arms of an
# `if` that later reads see.
import unittest


class Base:
    tag = "base"


class Sub(Base):
    tag = "sub"


class Factory:
    def __init__(self) -> None:
        self.log: list[str] = []
        self.sizes: list[int] = [7]

    def note(self, cls: type[Base]) -> None:
        self.log.append(cls.tag)
        self.sizes.append(len(self.log))

    def note_in_arms(self, cls: type[Base], k: int) -> str:
        if k % 2 == 0:
            self.log.append(cls.tag + "!")
        else:
            self.log.append(cls.tag)
        return str(len(self.log)) + self.log[-1]


def extend(xs: list[str], cls: type[Base]) -> int:
    xs.append(cls.tag)
    xs.append(cls.tag.upper())
    return len(xs)


class AFieldListGrows(unittest.TestCase):
    def test_in_a_method_called_in_a_loop(self) -> None:
        factory = Factory()
        seen: list[str] = []
        for i in range(3):
            factory.note(Base if i % 2 == 0 else Sub)
            seen.append(factory.note_in_arms(Sub, i))
        self.assertEqual(factory.log,
                         ["base", "sub!", "sub", "sub", "base", "sub!"])
        self.assertEqual(factory.sizes, [7, 1, 3, 5])
        self.assertEqual(seen, ["2sub!", "4sub", "6sub!"])

    def test_a_parameter(self) -> None:
        ys: list[str] = ["z"]
        lengths = [extend(ys, k) for k in (Base, Sub)]
        self.assertEqual(lengths, [3, 5])
        self.assertEqual(ys, ["z", "base", "BASE", "sub", "SUB"])


if __name__ == "__main__":
    unittest.main()
