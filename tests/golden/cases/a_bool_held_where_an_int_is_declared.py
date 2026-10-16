# Why execution: the values are the point -- a bool stored where an int is
# declared must still print as True and still add as 1, and only the printed
# lines tell those apart from a wrong one.
#
# CPython's bool is an int object. Lython's bool object had a layout of its
# own, so int's primitives read past it once a container declared `int` held
# one: `xs[0] + 1` was refused in the lowering, `sum(xs)` aborted in the
# allocator, `max(xs)` segfaulted, `min(xs)` answered 2 where CPython answers
# True, and a set rebuilt into a list read True back as 1. The bool objects
# now carry the int layout; int's repr and empty-spec format print a bool as
# itself, and a slot keeps a bool's object rather than its value.
def show(xs: list[int]) -> None:
    print(xs, sum(xs), min(xs), max(xs), sorted(xs))
    print(xs.count(True), xs.index(1), True in xs, 0 in xs, xs + [7], xs[::-1], xs[1:])
    print([x + 1 for x in xs], [str(x) for x in xs], [x for x in xs if x != 0])


def lists() -> None:
    xs: list[int] = [5, 0, 2]
    xs[1] = True
    xs.append(False)
    xs.insert(0, True)
    show(xs)
    print(xs.pop(), xs.pop(0), xs)
    xs.remove(1)
    print(xs)
    xs.sort()
    print(xs)
    xs.reverse()
    print(xs)


def sets() -> None:
    s: set[int] = {4, 9}
    s.add(True)
    s.add(False)
    print(sorted(s), sum(s), min(s), max(s), len(s), 1 in s, list(s))
    s.discard(1)
    print(sorted(s))


def dicts() -> None:
    d: dict[str, int] = {"a": 2}
    d["t"] = True
    d["f"] = False
    print(d, sum(d.values()), max(d.values()), sorted(d.values()), list(d.items()))
    for k, v in d.items():
        print(k, v, v * 10)
    counts: dict[int, int] = {}
    counts[True] = 3
    counts[1] = counts[1] + 1
    print(counts)


def tuples() -> None:
    t: tuple[int, ...] = (1, True, 3)
    print(t, sum(t), sorted(t), t.count(1), max(t))


lists()
sets()
dicts()
tuples()


def more_maps() -> None:
    d: dict[str, int] = {"a": 1}
    d["b"] = True
    w = d["b"]
    print(d, w + 10, w * 2)
    s: set[int] = {3}
    s.add(True)
    print(sorted(s), sum(s))


more_maps()


def through_a_parameter(xs: list[int]) -> None:
    print(xs[0] + 1, xs[0], f"{xs[0]:>3}", f"{xs[0]}", format(xs[0], "d"))


def caller() -> None:
    xs: list[int] = [1, 2]
    xs[0] = True
    through_a_parameter(xs)


caller()
