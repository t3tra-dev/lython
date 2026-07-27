# A local an enclosing loop carries, rebound in an INNER except handler that then
# raises. The rebind used to be lost silently: the handler's value can only leave
# a `py.try` through a result lane, a lane is only written by a yield, and a
# handler that raises never yields -- so the enclosing handler published the value
# the name held before the statement. `depth` stayed "0" for every iteration.
#
# a1d26ba's three re-raise goldens rebind in the OUTER handler, which survives,
# so none of them covers this. The distinguishing statement is `depth = ...`
# inside `except KeyError`, one line above a `raise`.
#
# Both str and int are exercised: with str the lost value is owned, and the same
# defect then also shows up as a rejection (the pre-try lane is released by the
# rebind and forwarded by the handler); with int there is no token at all, which
# is what proves the loss is control flow and not ownership.
#
# Why this needs execution: the compiler accepted the program and printed a wrong
# answer, so only the answer distinguishes fixed from broken.
def strings(n: int) -> str:
    tag = "start"
    seen = "-"
    depth = "0"
    i = 0
    while i < n:
        try:
            try:
                if i % 5 == 0:
                    raise KeyError("k" + str(i))
                seen = seen[-6:] + str(i % 7)
            except KeyError as inner:
                depth = str(inner)[-3:]
                raise ValueError("wrapped " + depth)
        except ValueError as outer:
            tag = str(outer)[-5:]
        i += 1
    return tag + "/" + seen[-4:] + "/" + depth


def integers(n: int) -> int:
    d = 0
    i = 0
    while i < n:
        try:
            try:
                if i % 2 == 0:
                    raise KeyError("k")
                d = 7
            except KeyError:
                d = 5
                raise ValueError("w")
        except ValueError:
            pass
        i += 1
    return d


print(strings(0))
print(strings(1))
print(strings(200))
print(integers(1))
print(integers(2))
print(integers(3))
