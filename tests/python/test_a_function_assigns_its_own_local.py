# A name a function assigns is the function's own local, also where the module
# binds the name too and the assignment is in a statement another arm of which
# raises: read after the statement, it is the value the arm assigned -- over a
# module global of the same type, a builtin, a class, and a local of the
# enclosing function. A name declared `global` or `nonlocal` is still the
# outer one.
import difflib
import unittest
from collections.abc import Iterator

tag = "module"
count = 0


class Box:
    pass


def after_if(which: int) -> str:
    if which == 1:
        tag = "one"
    elif which == 2:
        tag = "two"
    else:
        raise ValueError(which)
    return tag


def after_match(which: int) -> str:
    match which:
        case 1:
            tag = "one"
        case _:
            raise ValueError(which)
    return tag


def after_try(text: str) -> str:
    try:
        tag = text.upper()
    except ValueError:
        raise
    return tag


def in_a_generator(which: int) -> Iterator[str]:
    if which == 1:
        tag = "one"
    else:
        raise ValueError(which)
    yield tag


def shadows_a_builtin(n: int) -> int:
    if n > 0:
        len = n * 2
    else:
        raise ValueError(n)
    return len


def shadows_a_class(n: int) -> int:
    if n > 0:
        Box = n + 1
    else:
        raise ValueError(n)
    return Box


def shadows_an_enclosing_local() -> tuple[str, str]:
    tag = "outer"

    def inner(which: int) -> str:
        if which == 1:
            tag = "inner"
        else:
            raise ValueError(which)
        return tag
    return inner(1), tag


def writes_the_enclosing_local() -> tuple[str, str]:
    tag = "outer"

    def inner(which: int) -> str:
        nonlocal tag
        if which == 1:
            tag = "inner"
        else:
            raise ValueError(which)
        return tag
    first = inner(1)
    return first, tag


def writes_the_global(n: int) -> int:
    global count
    if n > 0:
        count = n
    else:
        raise ValueError(n)
    return count


class LocalTest(unittest.TestCase):

    def test_over_a_global_of_the_same_type(self) -> None:
        self.assertEqual((after_if(1), after_if(2), after_match(1),
                          after_try("x"), list(in_a_generator(1))),
                         ("one", "two", "one", "X", ["one"]))
        self.assertEqual(tag, "module")

    def test_over_a_builtin_and_a_class(self) -> None:
        self.assertEqual((shadows_a_builtin(3), shadows_a_class(3)), (6, 4))
        self.assertEqual(len("ab"), 2)

    def test_over_an_enclosing_functions_local(self) -> None:
        self.assertEqual(shadows_an_enclosing_local(), ("inner", "outer"))

    def test_global_and_nonlocal_still_write_outside(self) -> None:
        self.assertEqual(writes_the_enclosing_local(), ("inner", "inner"))
        self.assertEqual((writes_the_global(5), count), (5, 5))

    def test_the_stdlib_beside_a_global_spelled_like_its_local(self) -> None:
        # difflib.restore binds a local `tag` in the arms of an if whose last
        # arm raises.
        self.assertEqual(list(difflib.restore(["- a", "+ b", "  c"], 2)),
                         ["b", "c"])


if __name__ == "__main__":
    unittest.main()
