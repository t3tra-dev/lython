# Why execution: the answer is the TRACEBACK, which only running produces. The
# StopIteration that ends a hand-resumed generator carried an extra frame --
# and the line it named belonged to a call that had SUCCEEDED, so the traceback
# read as though the failure had two call sites.
#
# The generator body has already returned when the exhaustion StopIteration is
# raised, so CPython's traceback shows the caller's frame and nothing else.
# The raise lives in a function built ONCE, at whichever resume site
# materialized the clone, so the frame it pushed named that site forever: with
# `next(a)` once and then `next(b)` to exhaustion, the stale frame was on A --
# a different generator.
#
# ⛔ The directory is stripped from each `File` line, for the reason the other
# traceback goldens give: the recorded name is the absolute path.
import traceback
from typing import Iterator


def counted() -> Iterator[int]:
    yield 1
    yield 2


def valued() -> Iterator[str]:
    yield "a"


def frames(text: str) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("File "):
            out.append(stripped[stripped.rfind(", line"):])
    return out


def drain_by_hand() -> None:
    gen = counted()
    print(next(gen), next(gen))
    try:
        next(gen)
    except StopIteration:
        print("by hand", frames(traceback.format_exc()))


def drain_a_second_generator() -> None:
    first = counted()
    print(next(first))
    second = valued()
    print(next(second))
    try:
        next(second)
    except StopIteration:
        print("second", frames(traceback.format_exc()))


def drain_in_a_loop() -> None:
    total = 0
    for value in counted():
        total += value
    print("loop", total)
    fresh = counted()
    try:
        print("started", next(fresh))
    except StopIteration:
        print("unreachable")


def main() -> None:
    drain_by_hand()
    drain_a_second_generator()
    drain_in_a_loop()


main()
