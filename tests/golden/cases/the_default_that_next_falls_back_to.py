# What: `next(it, None)` -- the exhaustion check every hand-rolled scan is
# written with. The default and the element have to share one binding, so the
# result is Optional and has to be narrowed before it is used. Decoding both
# arms is the point: adding to the value proves the element came back as an
# int, and the None arm proves the fallback was reached rather than a zero.
def first(xs: "list[int]"):
    return next(iter(xs), None)


def describe(xs: "list[int]") -> str:
    got = first(xs)
    if got is None:
        return "empty"
    return str(got + 1)


print(describe([5, 6]), describe([]))
print(first([7]), first([]))

steps = iter([1, 2])
print(next(steps, None), next(steps, None), next(steps, None))

# A default of the element's own type keeps the plain element type, and the
# no-default spelling is untouched.
counts = iter([10, 20])
print(next(counts, 0) + 1, next(counts, 0) + 1, next(counts, 0) + 1)

words = iter(["a"])
print(next(words, "z"), next(words, "z"))

pairs = iter([(1, "a")])
print(next(pairs, None))

plain = iter([3, 4])
print(next(plain), next(plain))


def total(xs: "list[int]") -> int:
    it = iter(xs)
    running = 0
    for _ in range(len(xs) + 1):
        got = next(it, None)
        if got is None:
            break
        running += got
    return running


print(total([1, 2, 3]), total([]))
