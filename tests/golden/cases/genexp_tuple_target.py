# Why execution: the values are what the fused loops have to produce. A TUPLE
# target is what emitFor already binds -- `for a, b in zip(xs, ys)` is an
# ordinary loop -- so the fusion only had to know which names to scope, and
# the reducer's element type only had to bind them member-wise. Both rejected
# it: "generator expression target must be a simple name", and then an element
# type the fold could not see -- while the list comprehension spelling of the
# same thing worked.
def main() -> None:
    xs = [1, 2, 3]
    ys = [4, 5, 6]
    pairs = [(1, 2), (3, 4)]
    print(sum(a * b for a, b in zip(xs, ys)))
    print(max(a + b for a, b in pairs), min(a + b for a, b in pairs))
    print(any(a > 2 for a, b in pairs), all(b > 1 for a, b in pairs))
    print(list(a for a, b in pairs), [a for a, b in pairs])
    for v in (a + b for a, b in pairs):
        print(v)
    # the target names do not leak out of the genexpr
    a = 99
    print(sum(a * b for a, b in pairs), a)


main()
