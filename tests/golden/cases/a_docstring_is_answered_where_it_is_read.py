# `__doc__` of a function, a class, a method and the module is the docstring
# each was written with, cleaned as CPython 3.14's compiler cleans it -- tabs
# expanded, the first line's leading spaces dropped, the common indentation of
# the later lines taken off -- or None when there is none. Every one was
# refused: "'__doc__' is not available: docstrings are not retained at run
# time".
# Why run it: the cleaning is the compiler's, and only the printed text shows
# what it kept.
"""   The module's docstring.

        Indented further.
    Indented once.
"""


class Shape:
    """A shape.

       Its second line, indented less than the class body is."""

    def area(self) -> float:
        """\tThe area, after a tab.
\tA tab-indented line.
        A space-indented line.
        """
        return 0.0

    @staticmethod
    def unit() -> int:
        "A unit."
        return 1

    @classmethod
    def named(cls) -> str:
        'The class name.'
        return cls.__name__

    @property
    def sides(self) -> int:
        """The number of sides."""
        return 0


class Plain:
    x: int = 1


def documented(a: int) -> int:
    """Return a.

    Args:
        a: anything.
    """
    return a


def undocumented() -> None:
    pass


def after_a_statement() -> None:
    x = 1
    "a string after a statement is not a docstring"
    print(x)


def concatenated() -> None:
    ("two " "parts")


def formatted() -> None:
    f"""an f-string is not a docstring {1}"""


async def coroutine() -> None:
    """A coroutine's."""


print(repr(__doc__))
print(repr(documented.__doc__), repr(undocumented.__doc__))
print(repr(after_a_statement.__doc__), repr(concatenated.__doc__))
print(repr(formatted.__doc__), repr(coroutine.__doc__))
print(repr(Shape.__doc__), repr(Plain.__doc__))
print(repr(Shape.area.__doc__))
print(repr(Shape.unit.__doc__), repr(Shape.named.__doc__), repr(Shape.sides.__doc__))
print(repr(Shape().area.__doc__), repr(Plain().__doc__))
doc = documented.__doc__
print(doc.splitlines()[0] if doc else "-")
