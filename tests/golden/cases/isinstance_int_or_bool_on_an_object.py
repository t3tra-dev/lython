# Why execution: an int test accepts a bool, and only the printed values tell
# the bool's own value (True + 1 == 2) from a misread one.
#
# `isinstance(v, int)` and `isinstance(v, bool)` on an `object` value now
# narrow the branch, as every other builtin target already did. They were
# withheld because a boxed bool had a layout of its own that an int view
# misread; bool objects carry the int layout now, so an int view of one is
# exact, and a bool target is unboxed to its truth bit.
def as_int(value: object) -> str:
    if isinstance(value, int):
        return str(value) + "|" + str(value + 1) + "|" + str(value * 2) + "|" + repr(value)
    return "not int"


def as_bool(value: object) -> str:
    if isinstance(value, bool):
        return ("yes" if value else "no") + "|" + str(not value) + "|" + str(value)
    return "not bool"


def describe(items: list[object]) -> list[str]:
    out: list[str] = []
    for v in items:
        if isinstance(v, bool):
            out.append("b:" + str(v))
        elif isinstance(v, int):
            out.append("i:" + str(v - 1))
        elif isinstance(v, str):
            out.append("s:" + v)
        else:
            out.append("?")
    return out


for v in [3, True, False, 10 ** 25, -7, "x", 2.5, None]:
    print(as_int(v), as_bool(v))
print(describe([1, True, "a", False, 0, 3.0]))
