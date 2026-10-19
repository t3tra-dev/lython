# WHAT: the type parameter a generic function names only in `*args: T` or
# `**kwargs: V` is solved by what the call passes there -- spelled out, or as
# `f(*xs)` and `f(**d)` with a list, a `tuple[T, ...]` or a `dict[str, T]`,
# alone, after explicit arguments, beside keywords -- and the specialization
# collects those into its *args and **kwargs, with CPython's TypeError for a
# key the call also gives.
#
# WHY THIS IS RUN: how many members a sequence has and which keys a mapping
# holds are run-time facts.
def total[T](*args: T) -> list[T]:
    return list(args)


def keyed[V](**kwargs: V) -> list[str]:
    return [k + "=" + str(v) for k, v in kwargs.items()]


def first[T](head: T, *rest: T) -> tuple[T, int]:
    return head, len(rest)


def count[T](*args: T) -> int:
    return len(args)


def width[V](**kwargs: V) -> int:
    return len(kwargs)


xs = [1, 2, 3]
words = ("a", "b")
print(total(*xs), total(0, *xs), total(*words), total(*[]))
d = {"a": 1.5, "b": 2.5}
print(keyed(**d), keyed(z=0.5, **d), keyed(**d, y=3.5))
print(first(9, *xs), first("x", *words))
print(count(1, 2), count("a"), count(*xs), width(a=1), width(**d, c=0.5))
try:
    keyed(a=0.0, **d)
except TypeError as e:
    print("TypeError:", e)
