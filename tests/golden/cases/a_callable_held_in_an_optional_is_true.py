# WHAT: the truth of a value holding a function, a lambda and a bound method,
# alone and in `Callable | None` -- the `if isjunk:` difflib's SequenceMatcher
# writes before calling an optional filter -- and a lambda passed where a
# `Callable | None` is expected.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the test was refused
# ("does not provide manifest method '__bool__'"), and so was the lambda
# ("lambda requires a Callable annotation"); what has to hold now is
# that the filter is called when one is given and skipped when it is not,
# which only the values the calls return show.
from collections.abc import Callable


def is_blank(ch: str) -> bool:
    return ch == " "


class Counter:
    def count_junk(self, text: str,
                   junk: Callable[[str], bool] | None) -> int:
        if junk:
            return len([c for c in text if junk(c)])
        return -1


def plain(f: Callable[[str], bool]) -> str:
    return "true" if f else "false"


class Filter:
    def __init__(self, ws: str) -> None:
        self.ws = ws

    def test(self, ch: str) -> bool:
        return ch in self.ws


counter = Counter()
print(counter.count_junk("a b c", is_blank), counter.count_junk("a b c", None))
print(counter.count_junk("a\tb c", Filter("\t").test),
      counter.count_junk("abc", lambda c: c > "a"))
print(plain(is_blank), not is_blank)
