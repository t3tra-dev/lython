# What: a type parameter that two arguments give different types is their
# union (`collect(0, 1.5, 2)` is a list of int and float), and each value keeps
# its own type; a wider class among them is the parameter, as before.
# Why run: the element types are read back at run time.
def collect[T](*items: T) -> list[T]:
    return list(items)


def first[T](a: T, b: T) -> T:
    return a


def pair[K, V](k: K, v: V, w: V) -> dict[K, V]:
    return {k: v}


xs = collect(0, 1.5, 2)
print(xs, len(xs), [type(x).__name__ for x in xs])
f = first(2.5, 1)
print(f, f + 1, max(1, 2.5), first(1, 2) + 1)
print(pair("a", 1, "b"), first(1.5, 3))
ys: list[int | float] = collect(3, 4.5)
