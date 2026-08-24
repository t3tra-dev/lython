# What this pins: a comprehension whose target is spelled like an enclosing
# local leaves that local ALONE -- its value and its reference both.
#
# A comprehension has its own scope, so `[7 for i in range(3)]` neither reads
# nor writes the function's `i`. The desugaring emits a `for` statement, and
# `for i in ...:` legitimately rebinds the function's `i`: the loop carries the
# target as a block argument seeded with whatever `i` holds on entry and
# releases the incoming one each trip. Applied to a comprehension, the first
# trip released the ENCLOSING variable's value.
#
# Why this needs to run: the enclosing variable's value after the comprehension
# is a runtime value, and the failure is a use-after-release -- what it prints
# is whatever the freed box happens to hold. The compiler refused these
# programs while the affine verifier could see the hold, but the refusal moved
# with the shape (a later use of the name turned it into a different
# diagnostic) and vanished when the value was left unread, which is the case
# that would have executed wrongly.
#
# The first iterable is the one part evaluated in the ENCLOSING scope, so
# `[7 for i in range(i)]` reading the outer `i` is pinned here too.


def main() -> None:
    i: int = 1
    i = i + 1
    print([7 for i in range(3)], i)

    j: int = 4
    print([7 for j in range(j)], j)

    k: int = 1
    k = k + 1
    a: list[int] = [k for k in range(3)]
    b: list[int] = [k * 2 for k in range(3)]
    print(a, b, k)

    m: int = 2
    m = m + 1
    print({m: m * 2 for m in range(3)}, m)
    print(sorted({m for m in range(3)}), m)

    s: str = "hi"
    s = s + "!"
    print([s for s in ["p", "q"]], s)

    xs: list[int] = [10, 20]
    xs = xs + [30]
    print([xs for xs in range(2)], xs)

    p: int = 5
    q: int = 6
    p = p + q
    print([p + q for p in range(2) for q in range(2)], p, q)

    n: int = 3
    n = n + 1
    print([n for n in range(5) if n % 2 == 0], n)


main()
