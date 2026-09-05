# What: a nested def that captures a local bound to a bool literal. The capture
# arrives as `builtins.bool` while the binding was `literal<True>`, and the
# declared capture type has to be the one that arrives. Runtime values, because
# the question is which value the nested function reads -- the same capture of
# an int, a str or a float has always worked, and so has a bool assigned twice.


def flagged(n: int) -> int:
    flag = True

    def pick(v: int) -> int:
        if flag:
            return v
        return -v

    return pick(n)


def returned() -> bool:
    flag = False

    def read() -> bool:
        return flag

    return read()


def mixed(n: int) -> str:
    base = 10
    name = "x"
    ratio = 1.5
    flag = True

    def describe(v: int) -> str:
        if flag:
            return name + str(v + base) + str(ratio)
        return "off"

    return describe(n)


class Gate:
    def open(self, n: int) -> int:
        allowed = True

        def check(v: int) -> int:
            return v if allowed else 0

        return check(n)


print(flagged(3), flagged(-3))
print(returned())
print(mixed(2))
print(Gate().open(7))
