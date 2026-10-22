# WHAT: a parameter typed `Sequence[str]` left to its default `''`, in a
# function, in an `__init__`, and in a generic class built inside a
# generator method -- SequenceMatcher(isjunk) in difflib's Differ.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the default had no
# contract of its own ("default argument has no concrete runtime contract"),
# then no specialization to select, and in the generator a protocol view
# whose iteration had no evidence. What the default holds is seen in the
# list it is read into.
from collections.abc import Callable, Iterator, Sequence


def chars(a: Sequence[str] = '') -> list[str]:
    return list(a)


class Holder:
    def __init__(self, a: Sequence[str] = 'xy') -> None:
        self.a = list(a)


class Matcher[T]:
    def __init__(self, junk: Callable[[T], bool] | None = None,
                 a: Sequence[T] = '', b: Sequence[T] = '') -> None:
        self.junk = junk
        self.a: list[T] = list(a)
        self.b: list[T] = list(b)

    def set_b(self, b: Sequence[T]) -> None:
        self.b = list(b)


def blank(s: str) -> bool:
    return s == " "


class User:
    def __init__(self, junk: Callable[[str], bool] | None) -> None:
        self.junk = junk

    def walk(self, words: list[str]) -> Iterator[str]:
        m: Matcher[str] = Matcher(self.junk)
        for w in words:
            m.set_b(w)
            yield "".join(m.b) + str(len(m.a))


print(chars("ab"), chars(), Holder("q").a, Holder().a)
print(list(User(blank).walk(["cd", "ef"])))
