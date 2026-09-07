# Why execution: the narrowing decides the TYPE of a value that is then used
# for what only that type provides -- `v.upper()`, `len(v)`, returning it from
# a `-> str` function -- so the printed answers are the proof the exit edge
# carried the fact. The program did not compile: "cannot adapt
# !py.union<!py.contract<"builtins.int">, !py.contract<"builtins.str">> return
# value to callable return ABI 0 of f".
#
# A loop leaves when its test is FALSE, so its exit edge proves the negative
# exactly as an `if`'s fall-through does. The `if` spelling has always worked,
# which is what made this read as a union defect rather than a loop one.
#
# ⛔ Still refused, and both are refusals rather than wrong answers: a `break`
# leaves with the test still TRUE, and an `else` clause puts a block between
# the exit edge and the code after it. Neither is narrowed here.
from typing import Iterator


def to_text(flag: bool) -> str:
    v: "int | str" = 1 if flag else "a"
    while isinstance(v, int):
        v = "done"
    return v


def to_int(flag: bool) -> int:
    v: "int | None" = 1 if flag else None
    while v is None:
        v = 5
    return v


def measured(flag: bool) -> int:
    v: "int | str" = 1 if flag else "ab"
    while isinstance(v, int):
        v = "four"
    return len(v)


def down_a_tower() -> str:
    v: "int | float | str" = 1
    while isinstance(v, int):
        v = 2.5
    while isinstance(v, float):
        v = "settled"
    return v


def through_a_continue() -> str:
    v: "int | str" = 1
    seen = 0
    while isinstance(v, int):
        seen += 1
        if seen < 3:
            continue
        v = "after " + str(seen)
    return v


def nested() -> str:
    outer: "int | str" = 1
    while isinstance(outer, int):
        inner: "int | str" = 2
        while isinstance(inner, int):
            inner = "in"
        outer = inner
    return outer


def yielded() -> Iterator[str]:
    v: "int | str" = 1
    while isinstance(v, int):
        v = "y"
    yield v


def main() -> None:
    print(to_text(True), to_text(False))
    print(to_int(True), to_int(False))
    print(measured(True), measured(False))
    print(down_a_tower(), nested(), through_a_continue())
    print(list(yielded()))
    at_module: "int | str" = 1
    while isinstance(at_module, int):
        at_module = "m"
    print(at_module.upper())


main()
