# What this pins: `a, b = b, a + b` inside a loop, which is the fibonacci
# idiom and every loop-carried swap with it. It was refused -- "operand #0 does
# not dominate this use" at module scope, "released owned resource ... is used
# after release" inside a function -- because the general unpacking path
# materializes the right side as a TUPLE and then indexes it once per target,
# and in a loop that object's ownership placement has no answer. The same
# statement outside a loop worked, and so did the three-statement spelling with
# an explicit temporary.
#
# Both sides are written out here, so there is no tuple to build: Python
# evaluates the whole right side before assigning any target, which is what
# emitting every element first and then assigning IS.
#
# Why this needs to run rather than assert on a diagnostic: skipping the tuple
# changes evaluation ORDER if it is done wrong. Every right-hand element must
# be evaluated before any target is written, or `a, b = b, a + b` reads the new
# `a` when it computes `b`. The fibonacci numbers are what tell those apart --
# the wrong order gives 1 2 4 8 16, a doubling, not the sequence. The
# side-effect counter below pins the order directly.
#
# Every expected line is python3.14's.


# --- the idiom, at module scope and in a function -------------------------
a, b = 0, 1
i = 0
while i < 10:
    a, b = b, a + b
    i += 1
print(a, b)


def fib(n: int) -> int:
    x, y = 0, 1
    k = 0
    while k < n:
        x, y = y, x + y
        k += 1
    return x


print(fib(10), fib(30), fib(0), fib(1))


# --- a plain swap, and a three-way rotation -------------------------------
def rotate(n: int) -> str:
    p, q, r = "a", "b", "c"
    k = 0
    while k < n:
        p, q, r = q, r, p
        k += 1
    return p + q + r


print(rotate(0), rotate(1), rotate(2), rotate(3))


def reverse_pairs() -> str:
    out = ""
    for u, v in [(1, 2), (3, 4)]:
        u, v = v, u
        out = out + str(u) + str(v)
    return out


print(reverse_pairs())


# --- the ORDER: every right-hand element runs before any target is written
calls: list[int] = []


def note(v: int) -> int:
    calls.append(v)
    return v


m = 0
n = 0
m, n = note(1), note(2)
print(m, n, calls)


# --- the general path is still there for a non-literal right side ---------
xs = [7, 8]
c, d = xs
print(c, d)
t = (9, 10)
e, f = t
print(e, f)


# --- a target that is not a bare name goes through the tuple, and must -----
# The no-tuple path is restricted to bare names because the tuple is doing
# ownership work for a store target: `grid[0], grid[1] = grid[1], grid[0]`
# after the list already holds values leaks 2 allocations / 104 B without it.
# These lines are the control that the restricted path did not take them.
class Holder:
    def __init__(self) -> None:
        self.x: int = 0
        self.y: int = 0


h = Holder()
h.x, h.y = 5, 6
print(h.x, h.y)
h.x, h.y = h.y, h.x
print(h.x, h.y)

grid = [0, 0]
grid[0], grid[1] = 3, 4
print(grid)
grid[0], grid[1] = grid[1], grid[0]
print(grid)
