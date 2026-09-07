# WHAT: a mutually recursive nested pair where one member also captures a name
# of the ENCLOSING frame. That keeps the cell -- the group is not closed -- so
# the cycle stays: cell holds one function object, whose closure store holds
# the other, whose store holds the cell, and this runtime has no cycle
# collector.
#
# MEASURED 2026-09-07 with tests/probe/tools/leak_sweep.py: 5 allocs / 480 B
# per call of the enclosing function. The same pair WITHOUT the capture is
# reached by symbol and leaks nothing (golden:
# a_mutually_recursive_pair_inside_a_function).
#
# ⛔ Not a wrong answer -- the program prints CPython's -- so this is a probe
# and not a golden: a golden that leaks would put a known leaker in a corpus
# the leak sweep reads as clean.
#
# The fix is the one the closed group already takes, generalized: lift the
# group to real functions and pass the captured names as parameters. Until
# then, a member that reads the frame keeps the frame alive.
def scaled(n: int, base: int) -> int:
    def a(k: int) -> int:
        return base if k == 0 else b(k - 1)

    def b(k: int) -> int:
        return a(k - 1) + 1

    return a(n)


print(scaled(4, 10), scaled(2, 1))
