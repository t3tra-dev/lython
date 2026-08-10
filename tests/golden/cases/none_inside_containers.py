# Why execution: the failure was a SIGSEGV, and the fix is about the VALUE
# printed. A box carrying None has class id 0, which is the box's "no class"
# reading rather than a class -- every dispatch arm rebuilds its operand from
# the box's handle words, where slot 4 points at the entity, and a box with no
# class has nothing there. The class-0 arm dereferenced that null. Only running
# tells `[None]` from a crash, and from `[<garbage>]` if the arm were ever
# pointed at the wrong operand again.
from typing import Optional


def main() -> None:
    print([None])
    print((None,))
    print({"k": None})
    print([1, None, 2])
    values: list[Optional[int]] = [None, 3]
    print(values)


main()
