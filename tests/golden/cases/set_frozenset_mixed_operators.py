# Why execution: the result KIND is what the conversion has to preserve, and
# it is only visible at run time (a frozenset has no add). CPython runs the
# LEFT operand's __or__, which builds its own kind, so `frozenset | set` is a
# frozenset and `set | frozenset` is a set. The two kinds are different
# headers here, so the mixed pair had no operator at all: `sorted(f | {3})`
# was refused for a value that is an ordinary frozenset at run time.
def main() -> None:
    f = frozenset([1, 2])
    s = {2, 3}
    print(sorted(f | s), sorted(s | f))
    print(sorted(f & s), sorted(s & f))
    print(sorted(f - s), sorted(s - f))
    print(sorted(f ^ s), sorted(s ^ f))
    # the left kind decides: only a set can absorb an update afterwards
    grown = s | f
    grown.add(9)
    print(sorted(grown))
    print(len(f | s), 1 in (f | s), 9 in (f | s))
    # same-kind pairs are untouched
    print(sorted({1} | {2}), sorted(frozenset([1]) | frozenset([2])))
    # and the int operators keep theirs
    print(5 | 2, 5 & 3, 5 ^ 1, 5 - 1)


main()
