# Helper module for generic_import.py. Compiled on its own too (the golden
# glob makes every cases/*.py a case), where it declares everything and prints
# nothing -- generic functions are registered, not emitted, until a use site
# demands an instantiation.

LABEL: str = "item"
WIDTH: int = 3


def _pad(text: str) -> str:
    # Module-level constant read from a module-level helper.
    while len(text) < WIDTH:
        text = text + "."
    return text


def describe(text: str) -> str:
    # Module-level helper AND constant read from another module function.
    return _pad(text) + ":" + LABEL


def first[T](values: list[T]) -> T:
    return values[0]


def pair_up[A, B](left: A, right: B) -> list[str]:
    out: list[str] = []
    out.append(str(left))
    out.append(str(right))
    return out


def count_matches[T](values: list[T], wanted: T) -> int:
    total = 0
    for value in values:
        if value == wanted:
            total = total + 1
    return total


class Tagger:
    def __init__(self, name: str) -> None:
        self.name = name

    def tagged(self) -> str:
        # A method of an imported class reading its own module's globals.
        return describe(self.name)


# Module-level alias (of a non-generic function; an alias of a GENERIC one is
# only resolvable through an import, since a bare reference in a main module
# has no ground context -- stdlib_bisect covers that through `bisect`).
label_of = describe
