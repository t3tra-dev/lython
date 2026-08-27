# WHAT: `//`, `%`, `<<`, `>>`, `&`, `|` and `^` on ints, over the operand pairs
# that separate a machine word's arithmetic from Python's -- negative operands,
# the i64 boundaries, a zero divisor, a negative shift count, and operands that
# do not fit a word at all.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: each of these now has TWO
# implementations. The fast one is native i64 in the emitted code, guarded by a
# validity bit; the slow one is the manifest bignum. Which one answers is a
# run-time decision on the operand values, so only running the program tells
# you whether the pair that took the fast path got Python's answer.
#
# ⛔ THE SIGNS ARE THE POINT. LLVM's `divsi`/`remsi` truncate toward zero and
# Python's `//`/`%` floor: `-7 // 2` is -4 here and -3 there, and `-7 % 2` is 1
# here and -1 there -- the remainder takes the DIVISOR's sign. Every row with a
# negative operand is a row that a straight translation of the machine
# instruction gets wrong.
#
# ⛔ AND THE TWO PAIRS THE FAST PATH MAY NOT ANSWER. A zero divisor is a
# ZeroDivisionError and `-2**63 // -1` does not fit, so both have to reach the
# boxed path; a negative shift count is a ValueError there. If the guard were
# missing the first two would trap rather than raise.


def div(a: int, b: int) -> int:
    return a // b


def mod(a: int, b: int) -> int:
    return a % b


def shl(a: int, b: int) -> int:
    return a << b


def shr(a: int, b: int) -> int:
    return a >> b


def sweep(n: int) -> int:
    """Every pair in range, on the raw lane: `sweep` takes and returns an int,
    so it gets the unboxed clone and `i` is a machine word inside it."""
    total = 0
    i = 0 - n
    while i <= n:
        if i != 0:
            total = total + (100 // i) + (100 % i) + (i // 7) + (i % 7)
            total = total + (i & 255) + (i | 3) + (i ^ 9)
            if i > 0:
                total = total + (i >> 2) + ((i % 40) << 3)
        i = i + 1
    return total


# ⛔ THE PAIRS ARE WRITTEN OUT, NOT ITERATED. An operand read out of a list is
# a boxed object with no raw lane, so a table-driven loop tests the bignum and
# NOTHING of the fast path -- measured: with floored division replaced by the
# machine's truncating one, the table-driven spelling of this golden stayed
# green. A literal, a local bound from one, and an int parameter all carry the
# lane; those are the three spellings below.
print(7 // 2, 7 % 2, -7 // 2, -7 % 2, 7 // -2, 7 % -2, -7 // -2, -7 % -2)
print(0 // 5, 0 % 5, -1 // 3, -1 % 3, 1 // -3, 1 % -3)
print(123456789 // 1000, 123456789 % 1000, -123456789 // 1000,
      -123456789 % 1000)

LO = -9223372036854775808
HI = 9223372036854775807
print(HI // 3, HI % 3, LO // 3, LO % 3, LO // -1, HI // -1, LO % -1)

a = -7
b = 2
print(a // b, a % b, a << 3, a >> 1, a & 12, a | 12, a ^ 12)

print(div(-7, 2), mod(-7, 2), div(HI, 3), mod(LO, 3), div(LO, -1))
print(shl(1, 62), shl(1, 63), shl(-1, 62), shl(255, 3), shl(HI, 1), shl(LO, 1))
print(shr(1, 62), shr(-1, 62), shr(-255, 3), shr(HI, 1), shr(LO, 1))
print(shl(1, 64), shl(1, 200), shr(1, 64), shr(-1, 200), shl(12345, 0),
      shr(12345, 0))
print(12 & 10, -12 & 10, 12 & -10, -12 & -10, 0 & -1, HI & LO)
print(12 | 10, -12 | 10, 12 | -10, -12 | -10, 0 | -1, HI | LO)
print(12 ^ 10, -12 ^ 10, 12 ^ -10, -12 ^ -10, 0 ^ -1, HI ^ LO)
print(sweep(60))

# bool keeps its class through `&`, `|` and `^`, and loses it through the rest.
print(True & True, True | False, True ^ True, True // True, True % True,
      True << 1, True >> 1)

# Operands no word can hold: the guard has to send these to the bignum.
BIG = 1 << 100
print(BIG % 7, -BIG // 7, BIG >> 90, (BIG + 1) & 255)
print((1 << 62) * 4 // 3)

try:
    print(div(7, 0))
except ZeroDivisionError as e:
    print("ZeroDivisionError:", str(e))
try:
    print(mod(7, 0))
except ZeroDivisionError as e:
    print("ZeroDivisionError:", str(e))
try:
    print(shr(7, -1))
except ValueError as e:
    print("ValueError:", str(e))
