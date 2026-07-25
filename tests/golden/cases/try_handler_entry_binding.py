# A local the try body rebinds is observed by the handler, by the finally
# body and by the continuation with the value it held AT THE RAISE POINT.
#
# The shapes below are the CPython truth table that rules out every
# single-static-binding design, so they are pinned individually:
#   * two raise points in one body make the same handler answer differently,
#   * a rebind AFTER the raise is never observed,
#   * a handler that only passes still sees the body's value afterwards.


def one_raise() -> int:
    n = 1
    try:
        n = 3
        raise ValueError()
    except ValueError:
        pass
    return n


print(one_raise())


def two_raise_points(which: int) -> int:
    n = 1
    try:
        n = 2
        if which == 0:
            raise ValueError()
        n = 3
        raise ValueError()
    except ValueError:
        return n
    return -1


print(two_raise_points(0))
print(two_raise_points(1))


def rebind_after_raise() -> int:
    x = 1
    try:
        x = 2
        raise ValueError()
        x = 3
    except ValueError:
        pass
    return x


print(rebind_after_raise())


def handler_reads_it() -> str:
    n = 1
    try:
        n = 3
        raise ValueError()
    except ValueError:
        return "handler saw " + str(n)
    return "no"


print(handler_reads_it())


def no_raise_at_all() -> int:
    n = 1
    try:
        n = 3
    except ValueError:
        pass
    return n


print(no_raise_at_all())


def carries_str() -> str:
    s = "a"
    try:
        s = "b"
        raise ValueError()
    except ValueError:
        pass
    return s


print(carries_str())


def carries_list() -> list[int]:
    xs = [1]
    try:
        xs = [1, 2, 3]
        raise ValueError()
    except ValueError:
        pass
    return xs


print(carries_list())


class Tag:
    def __init__(self, label: str) -> None:
        self.label: str = label


def carries_instance() -> str:
    t = Tag("before")
    try:
        t = Tag("inside")
        raise ValueError()
    except ValueError:
        pass
    return t.label


print(carries_instance())


def handler_rebinds_too() -> int:
    n = 1
    try:
        n = 3
        raise ValueError()
    except ValueError:
        n = n + 10
    return n


print(handler_rebinds_too())


# Several promoted locals of different contracts in one statement.
def several(kind: int) -> str:
    count = 0
    label = "start"
    try:
        count = 1
        label = "mid"
        if kind == 0:
            raise ValueError()
        count = 2
        label = "end"
        raise KeyError()
    except ValueError:
        pass
    except KeyError:
        pass
    return str(count) + label


print(several(0))
print(several(1))


# Module level, not only inside a function.
top = 1
try:
    top = 3
    raise RuntimeError()
except RuntimeError:
    pass
print(top)


# The finally body observes the same incarnation, and so does the continuation
# after it. (Module level: a callee whose body allocates inside a try/finally
# still runs twice, an unrelated defect tracked separately.)
mod_final = 1
try:
    mod_final = 3
finally:
    print("finally saw", mod_final)
print(mod_final)


# The else body runs only on normal completion, where the body's value is the
# one the continuation sees as well.
mod_else = 1
try:
    mod_else = 3
except ValueError:
    pass
else:
    print("else saw", mod_else)
print(mod_else)
