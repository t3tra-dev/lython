# WHAT: `is` on values typed `object` is identity: the same instance, list,
# None, bool, small int or float passed twice is the same object; two
# instances or two equal lists are not; a value read back out of a
# `list[object]` is the object that was put in. And containers take a NaN to
# be itself, as CPython's identity test does: `nan in [nan]`, `[nan] ==
# [nan]`, `count`, `index` and a dict or set keyed by it.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: every crossing into
# `object` builds a box of its own, and identity is what the boxes hold, not
# the boxes; only running it shows which word each side compares.


class Box:
    pass


def same(a: object, b: object) -> bool:
    return a is b


def differ(a: object, b: object) -> bool:
    return a is not b


def is_none(a: object) -> bool:
    return a is None


def check(n: int, p: float) -> None:
    a = Box()
    b = Box()
    xs = [1]
    ys = [1]
    x = p * 2.0
    print(same(a, a), same(xs, xs), same(None, None), same(True, True))
    print(same(n, n), same(x, x), same(p, p), same(a, b), same(xs, ys))
    print(differ(a, b), differ(a, a), same(None, 0), same(0, False), same(1, True))
    print(is_none(None), is_none(a), is_none(0), is_none(False))
    items: list[object] = [a, xs, None, n]
    print(same(items[0], a), same(items[1], xs), same(items[2], None),
          same(items[0], b), items[1] is xs, items[0] is not b)


check(7, 1.5)


def nans(p: float) -> None:
    nan = float("nan") * p
    xs = [nan, 1.0]
    print(nan in xs, xs == [nan, 1.0], xs.count(nan), xs.index(nan))
    t = (nan, 2.0)
    print(nan in t, t == (nan, 2.0), t.count(nan), t.index(nan))
    d = {nan: 1}
    s = {nan}
    print(nan in d, d.get(nan, 0), nan in s, nan == nan, same(nan, nan))


nans(1.5)
