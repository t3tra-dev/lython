# Why execution: the arity is a RUNTIME comparison against the source object
# (a tuple whose members share a type collapses to `tuple[T]`, and a list
# carries its length in the object), so only running it shows the raise.
# Every one of these silently bound the first N and dropped the rest.
#
# CPython's two messages differ in one detail: the tuple/list fast paths in
# ceval report what they got, the generic iterator path does not know it.
def main() -> None:
    ok = [1, 2]
    a, b = ok
    print(a, b)
    pair = (3, 4)
    c, d = pair
    print(c, d)
    for e, f in [(5, 6), (7, 8)]:
        print(e, f)
    try:
        g, h = [1, 2, 3]
        print(g, h)
    except ValueError as err:
        print(err)
    try:
        i, j = [1]
        print(i, j)
    except ValueError as err:
        print(err)
    try:
        k, m = (1, 2, 3)
        print(k, m)
    except ValueError as err:
        print(err)
    try:
        n, o = "xyz"
        print(n, o)
    except ValueError as err:
        print(err)
    try:
        p, q, r = "xy"
        print(p, q, r)
    except ValueError as err:
        print(err)
    try:
        for s, t in [(1, 2, 3)]:
            print(s, t)
    except ValueError as err:
        print(err)


main()
