# Why execution: the lazy builtin iterators synthesize their generator when
# they are EMITTED, so their type is not what inferExpr sees -- only running
# the materializing spellings shows they produce the same elements the loop
# spelling does. Every one of these was "builtins.object does not provide
# manifest method '__iter__'" while the same call bound to a name first
# worked.
def main() -> None:
    a = [1, 2, 3]
    b = ["x", "y", "z"]
    print(list(zip(a, b)))
    print(tuple(zip(a, b)))
    print(dict(zip(b, a)))
    print(list(enumerate(b)))
    print(list(enumerate(b, 1)))
    print(list(map(str, a)))
    print(list(filter(lambda v: v > 1, a)))
    print(list(reversed(a)))
    print(sorted(zip(b, a)))
    print([p for p in zip(a, b)])
    print({i: v for i, v in enumerate(b)})
    print(sum(v for v in map(int, ["1", "2"])))
    # pow(x, y) is the ** operator: CPython's builtin_pow passes Py_None for
    # the modulus, the same slot ** reaches. Only the three-argument form is
    # declared in the manifest, so this arity was refused.
    print(pow(2, 10), pow(2, 10, 1000), pow(2.0, 3), pow(2, -1))


main()
