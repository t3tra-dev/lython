# What this pins: a borrowed parameter rebound inside a NESTED loop. It was
# refused by the affine verifier -- "borrowed entry argument 0 of @f is
# returned with 2 retained ownership tokens; exactly one may be transferred",
# and the non-returning spelling with "reaches function exit with 1" -- which
# took `roman()` and every accumulator that consumes its own parameter.
#
# One merge-borrow lend is taken per loop entry edge and returned on the back
# edge THROUGH THE PRE-MERGE NAME. The walk counted the lend and could not
# credit the return, because the group had been remapped to the merge argument
# by then, so the balance ran one high per rename and a nested loop renames
# twice.
#
# Why this needs to run rather than assert on a diagnostic: the refusal it
# replaces was loud, but what the repair changes is a REFCOUNT count, and the
# two ways of getting that wrong are both silent. One credit too many frees the
# caller's argument while the caller still holds it; one too few leaks it per
# call. The values below only show the first; the second is why this case is in
# the leak gate, where 2000 calls have to come back at net zero.
#
# Every expected line is python3.14's.


# --- the minimal shape: two loops, the inner one consuming the parameter ---
def strip_tens(n: int) -> int:
    i = 0
    while i < 2:
        while n >= 10:
            n -= 10
        i += 1
    return n


print(strip_tens(25), strip_tens(9), strip_tens(100))


# --- the same, returning something else, which is the other arm ------------
def count_tens(n: int) -> str:
    out = ""
    i = 0
    while i < 2:
        while n >= 10:
            out += "X"
            n -= 10
        i += 1
    return out


print(count_tens(25), count_tens(9))


# --- roman numerals, which is why this matters ----------------------------
def roman(n: int) -> str:
    vals = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    syms = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV",
            "I"]
    out = ""
    i = 0
    while i < len(vals):
        while n >= vals[i]:
            out += syms[i]
            n -= vals[i]
        i += 1
    return out


print(roman(1994), roman(4), roman(2024), roman(0), roman(3999))


# --- a `for` over the table instead of an index ---------------------------
def digits(n: int) -> str:
    out = ""
    for base in [100, 10, 1]:
        count = 0
        while n >= base:
            n -= base
            count += 1
        out += str(count)
    return out


print(digits(305), digits(0), digits(999))


# --- three levels, and the parameter consumed at the innermost ------------
def drain(n: int) -> int:
    a = 0
    while a < 2:
        b = 0
        while b < 2:
            while n >= 3:
                n -= 3
            b += 1
        a += 1
    return n


print(drain(10), drain(2))


# --- called many times, so the leak gate has something to count -----------
k = 0
total = 0
while k < 200:
    total += strip_tens(25) + len(roman(1994))
    k += 1
print(total)
