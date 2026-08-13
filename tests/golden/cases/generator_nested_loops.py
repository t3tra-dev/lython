# Nested loops inside a generator body. The affine ownership walk did not
# converge on this CFG -- it refused with "ownership CFG exploration exceeded
# 20000 states" -- so the program never compiled.
#
# Needs execution: the accumulator crosses two back edges and the frame, and
# the refusal said nothing about which value it would have carried.
#
# ⛔ This program LEAKS one range iterator (1 allocation / 56 B, bounded --
# the same figure at n=3, 10 and 40), so it is NOT in the leak gate. The leak
# is what the non-convergence was hiding, and it is recorded with its
# measurement in tests/probe/wb_generator_nested_loop_iterator_leak.py.
from typing import Iterator


def sum_of_products(n: int) -> Iterator[int]:
    total = 0
    for i in range(n):
        for j in range(2):
            total = total + i * j
    yield total


def two_yields(n: int) -> Iterator[int]:
    total = 0
    for i in range(n):
        for j in range(3):
            total = total + j
    yield total
    yield total * 2


def nested_then_flat(n: int) -> Iterator[int]:
    total = 0
    for i in range(n):
        for j in range(2):
            total = total + 1
    for k in range(n):
        total = total + 100
    yield total


for v in sum_of_products(3):
    print(v)
for v in sum_of_products(0):
    print(v)
for v in two_yields(4):
    print(v)
for v in nested_then_flat(3):
    print(v)
