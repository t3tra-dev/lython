# WHAT: a name a loop body, an `if` arm or a `try` body binds, read after that
# statement -- and read where the binding did not happen, which is
# `UnboundLocalError` in a function and `NameError` at module scope.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: both halves are runtime
# answers. The bound half is the VALUE the last executed binding left, and the
# unbound half is an exception class and message that a reader compares against
# CPython's word for word.
#
# ⛔ A LOOP TARGET is deliberately not one of these. `for i in xs` reuses the
# same spelling across loops over different element types, so one storage with
# one type is the wrong shape for it; reading a loop target after its loop
# still fails to resolve.
import sys


def last_of(xs: "list[int]") -> int:
    for v in xs:
        seen = v
    return seen


print(last_of([1, 2, 3]))
try:
    print(last_of([]))
except UnboundLocalError as e:
    print("UnboundLocalError:", e)


def classify(n: int) -> str:
    if n > 0:
        tag = "pos"
    elif n < 0:
        tag = "neg"
    return tag


print(classify(5), classify(-5))
try:
    print(classify(0))
except UnboundLocalError as e:
    print("UnboundLocalError:", e)


def guarded(c: bool) -> str:
    try:
        if c:
            raise ValueError("boom")
        got = "ok"
    except ValueError:
        got = "caught"
    return got


print(guarded(False), guarded(True))


def rebound(c: bool) -> int:
    if c:
        n = 1
    n = 5
    return n


print(rebound(False), rebound(True))


def counted(c: bool) -> int:
    while c:
        total = 9
        c = False
    return total


print(counted(True))


def closed_over(c: bool) -> str:
    if c:
        message = "hi"

    def read() -> str:
        return message

    return read()


print(closed_over(True))

for step in [10, 20]:
    running = step
print(running)

if len(sys.argv) < 0:
    never = "unreachable"
try:
    sys.stdout.write(never)
except NameError as e:
    print("NameError:", e)
