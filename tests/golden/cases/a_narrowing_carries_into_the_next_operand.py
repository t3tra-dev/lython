# WHAT: `A and B` where A proves something about a name B reads -- the spelling
# the Optional idiom is written in, and the one that could not be written here
# until now. Also `or` (which proves the FALSE side), `not`, and a chain of
# three.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the failure this replaces
# was a diagnostic, but the repair is not "it compiles" -- it is WHICH operand
# sees the narrowing and on which side of the test. An `and` that carried the
# false-narrowing, or an `or` that carried the true one, still compiles and
# answers the wrong question. Every line below is the answer for an argument
# that takes the short-circuit and one that does not.
from typing import Optional


def at_least(n: int, limit: Optional[int]) -> bool:
    return limit is not None and n >= limit


def short_or_none(s: Optional[str]) -> bool:
    return s is None or len(s) < 3


def both(s: Optional[str], t: Optional[str]) -> bool:
    return s is not None and t is not None and len(s) + len(t) > 3


def not_none_and_long(s: Optional[str]) -> bool:
    return not (s is None) and len(s) > 1


print(at_least(3, 2), at_least(1, 2), at_least(1, None))
print(short_or_none(None), short_or_none("ab"), short_or_none("abcd"))
print(both("ab", "cd"), both("a", "b"), both(None, "cd"), both("ab", None))
print(not_none_and_long("abc"), not_none_and_long("a"), not_none_and_long(None))

# The narrowing must not outlive the expression: `s` is a union again here.
def still_a_union(s: Optional[str]) -> str:
    ok = s is not None and len(s) > 1
    if s is None:
        return "none"
    return s + ("!" if ok else "?")


print(still_a_union("ab"), still_a_union("a"), still_a_union(None))
