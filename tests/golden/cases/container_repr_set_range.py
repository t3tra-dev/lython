# Why execution: these are exact-text renderings produced at run time.
# `print({1, 2})` did not reach the screen at all -- the lowering reported
# "runtime manifest has no builtins.set.__repr__ method", and a set nested in
# a printed container aborted the process. frozenset and range fell back to
# the address form `<frozenset object at 0x...>`, which is a wrong answer
# rather than a refusal.
#
# Deviation, noted: set elements come out in the table's own order (insertion
# order here), not CPython's hash order. The values below are chosen where
# the two agree.
def main() -> None:
    print({1, 2, 3})
    print(set())
    print(frozenset([1, 2]))
    print(frozenset())
    print([{1}, {2}])
    print({1: {2}})
    print(repr({1, 2}), str({1, 2}))
    print(range(3))
    print(range(1, 5))
    print(range(1, 10, 2))
    print(range(5, 1, -1))
    print([range(2)])
    print(f"{range(3)} {frozenset([1])}")


main()
