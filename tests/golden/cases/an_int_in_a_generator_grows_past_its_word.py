# WHAT: ints inside a generator -- the locals it keeps across a yield, the
# value it yields, an argument -- answer as CPython when `+`, `-`, `*` and `<<`
# take them past 2**63 and when they come back under it: Fibonacci to the
# 100th term, factorials, a count down past -2**63, a big argument, a value
# made before a yield and read after it as an object (appended, printed, a
# dict key, hashed), one value kept in two locals, a local that leaves the
# word and comes back between yields, and a delegating generator; `//` and `%`
# by zero and a negative shift raise inside the generator and a `try` around
# the loop catches them.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: a generator's int is its
# i64 while that is the value, and its frame keeps the word and, past it, the
# object an operator's slow arm made; whether each crossing (the yield, the
# frame, the argument) brings back the value -- and not 0, or "int too large"
# -- is only seen by running it.
from typing import Iterator


def fibonacci_numbers() -> Iterator[int]:
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b


def factorials_up_to(n: int) -> Iterator[int]:
    f = 1
    for i in range(1, n + 1):
        f = f * i
        yield f


def count_down(n: int) -> Iterator[int]:
    x = -9223372036854775807
    for _ in range(n):
        x = x - 2
        yield x


def echo_argument(n: int) -> Iterator[int]:
    yield n
    yield n + 1
    yield n * n


def keep_across(n: int) -> Iterator[int]:
    small = n * 1000003
    big = n << 70
    twice = small
    yield 0
    held: list[int] = []
    held.append(small)
    held.append(big)
    names = {small: "small", big: "big"}
    print("before the yield:", small, big, names[n * 1000003], names[n << 70])
    print("same hash:", hash(small) == hash(n * 1000003), hash(big) == hash(n << 70))
    yield held[0] + held[1]
    yield small + twice
    yield (big >> 70) + (small - twice)


def carry_over(n: int, m: int) -> Iterator[int]:
    a = 0
    for _ in range(n):
        yield a
        a = (a + 7) * m * m * m // m // m // m % m


def raise_inside(n: int) -> Iterator[int]:
    x = 7
    yield x // n
    yield x % n
    yield x << n
    yield x >> n


def delegate(n: int) -> Iterator[int]:
    yield from factorials_up_to(n)
    yield -1


out: list[int] = []
for i, v in enumerate(fibonacci_numbers()):
    if i >= 100:
        break
    out.append(v)
print("fibonacci:", out[-1], out[93], out[92], len(out))
print("factorial:", list(factorials_up_to(25))[19:])
print("counted down:", list(count_down(3)))
print("echoed:", list(echo_argument(10**30)), list(echo_argument(5)), list(echo_argument(-(2**63))))
print("across a yield:", list(keep_across(3)))
print("across a yield:", list(keep_across(10**20)))
print("out and back:", sum(carry_over(1000, 1000003)), list(carry_over(4, 1000003)))
print("delegated:", list(delegate(22))[-3:])
for n in [2, 0, -1]:
    try:
        print("raised:", n, list(raise_inside(n)))
    except ZeroDivisionError as e:
        print("raised:", n, "ZeroDivisionError", e)
    except ValueError as e:
        print("raised:", n, "ValueError", e)
