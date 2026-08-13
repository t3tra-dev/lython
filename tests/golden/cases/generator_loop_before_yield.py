# A loop that finishes BEFORE the first yield. The generator's resume clone
# expands the loop's carried merge re-entrantly, and the inner expansion's
# drain used to finish a later argument of the same block while the outer one
# still had no edge operands -- so the later argument read a physical lane as
# its logical operand and the program was refused.
#
# Needs execution: the defect was a refusal, but the contract under it is
# which VALUE each merge carries, and only running the generator shows the
# accumulator arriving intact at the yield.
from typing import Iterator


def after_a_for(n: int) -> Iterator[int]:
    total = 0
    for i in range(n):
        total = total + i
    yield total


def two_yields_after_a_for(n: int) -> Iterator[int]:
    total = 0
    for i in range(n):
        total = total + i
    yield total
    yield total * 2


def after_a_while(n: int) -> Iterator[int]:
    total = 0
    i = 0
    while i < n:
        total = total + i
        i = i + 1
    yield total


def a_loop_between_two_yields(n: int) -> Iterator[int]:
    yield -1
    total = 0
    for i in range(n):
        total = total + i
    yield total


def two_loops_then_a_yield(n: int) -> Iterator[int]:
    total = 0
    for i in range(n):
        total = total + i
    for j in range(n):
        total = total + j * 10
    yield total


def a_loop_then_a_yielding_loop(n: int) -> Iterator[int]:
    base = 0
    for i in range(n):
        base = base + i
    for j in range(3):
        yield base + j


def two_carried_locals(n: int) -> Iterator[int]:
    total = 0
    count = 0
    for i in range(n):
        total = total + i
        count = count + 1
    yield total
    yield count


for v in after_a_for(5):
    print(v)
for v in two_yields_after_a_for(5):
    print(v)
for v in after_a_while(5):
    print(v)
for v in a_loop_between_two_yields(5):
    print(v)
# A SECOND `range` in the body: its class object is rematerialized in the
# block that builds it rather than threaded as a block argument, since a
# `!py.type<...>` has no runtime value to thread.
for v in two_loops_then_a_yield(5):
    print(v)
for v in a_loop_then_a_yielding_loop(5):
    print(v)
for v in two_carried_locals(5):
    print(v)

# An empty loop still reaches the yield, carrying the initial value.
for v in after_a_for(0):
    print(v)
for v in two_carried_locals(0):
    print(v)
