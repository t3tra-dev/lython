# WHAT: a function whose clone cannot say (a zero divisor) is re-run boxed by
# its caller, with the arguments the caller held only as i64s boxed for that
# call; when the re-run raises, the caller's `except` catches it and every
# argument boxed for it is released -- and so is a heap int an exception skips
# past after a `j = m` rebinding in the same frame.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the defect is a reference
# nobody releases on the exceptional path; only the leak gate, which runs the
# program, can see it (it sits in LYTHON_LEAK_GATE_CASES). The printed values
# pin that the boxed re-run and the exception still answer as CPython.


def inner(a: int, b: int) -> int:
    return a // b


def outer(xs: list[int]) -> int:
    t = 0
    for x in xs:
        try:
            t += inner(x * 1000, 0)
        except ZeroDivisionError:
            t += 1
    return t


def scan(out: list[str], extra: int) -> int:
    m = len(out) + extra
    j = 0
    while j < m:
        x = out[j]
        if x == "stop":
            j = m
            continue
        j = j + 1
    return j


print(outer(list(range(300))))
caught = 0
for i in range(50):
    try:
        scan(["a"], 1000 + i)
    except IndexError:
        caught += 1
print(caught, scan(["a", "stop"], 1000))
