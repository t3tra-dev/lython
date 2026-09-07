# Why execution: what the guard proves decides the TYPE of the value the body
# uses, so the proof is only visible in what that value is then allowed to do.
# Two compound guards were each answered with one fact too few:
#
#     if v is not None and isinstance(v, str):
#         return v          # cannot adapt !py.union<int, str> return value
#
# -- the first fact took the slot for the name and the second was dropped, so
# `v` narrowed to `int | str` and stopped there. An `and` evaluates left to
# right and each operand is proved under the ones before it, so the LATER fact
# is the refinement.
#
#     assert isinstance(v, str) and isinstance(w, str)
#     return v + w          # str.__add__ has no signature accepting (int | str)
#
# -- `assert T` is `if not T: raise`, and the walk asked its single-fact
# spelling about the whole conjunction, so it narrowed the first subject and
# left the second reading the union. `not X` proves X's facts with the sides
# swapped, however many there are.
#
# ⛔ Two facts that CONTRADICT each other keep the first, which is the
# conservative answer: `isinstance(v, str) and isinstance(v, int)` proves
# nothing about the second, and the body below is unreachable in CPython too.


def refined(v: "int | str | None") -> str:
    if v is not None and isinstance(v, str):
        return v
    return "?"


def refined_further(v: "int | str | None") -> int:
    if v is not None and isinstance(v, str) and len(v) > 1:
        return len(v)
    return -1


def two_subjects(v: "int | str", w: "int | str") -> str:
    assert isinstance(v, str) and isinstance(w, str)
    return v + w


def negated_pair(v: "int | str", w: "int | str") -> str:
    if not (isinstance(v, str) and isinstance(w, str)):
        return "?"
    return v + "-" + w


def double_negated(v: "int | str") -> str:
    if not (not isinstance(v, str)):
        return v
    return "?"


def contradictory(v: "int | str") -> str:
    if isinstance(v, str) and isinstance(v, int):
        return "both"
    return "neither"


def main() -> None:
    print(refined("a"), refined(1), refined(None))
    print(refined_further("ab"), refined_further("a"), refined_further(None))
    print(two_subjects("x", "y"))
    print(negated_pair("x", "y"), negated_pair(1, "y"))
    print(double_negated("a"), double_negated(1))
    print(contradictory("a"), contradictory(1))


main()
