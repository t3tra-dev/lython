# Constructs nested inside a loop BODY, each writing a loop-carried local that
# the code after the loop reads back.
#
# Why execution is needed: the local is threaded through the loop header as a
# block argument, so it exists under one name before the loop and another
# inside it. `tests/probe/wb_forloop_handler_local_unwind.py` documents three
# separate consumers that rename that pair forward only, one of which frees the
# cell while it is still live. Which incarnation a nested construct's write
# lands on is decided during lowering and is visible only in the value the
# program prints -- `tests/probe/wb_loopelse_only_write_lost.py` records a
# spelling that compiles clean, verifies clean, exits 0 and prints the
# PRE-LOOP value.
#
# Why these cells: `tests/probe/tools/nestgrid.py` reports all of them empty in
# all four corpora -- 87 golden cases contain a `for` and 48 a `while`, and
# none nests a comprehension, a `def`, a generator or a `lambda` in the body in
# a way that couples a local across the loop edge. All twelve are correct today
# (5/5 against CPython). The `for`/`while` bodies are used both ways round so a
# regression in either loop's block-argument handling is named by the line.
def for_nested_while() -> int:
    acc = 0
    for i in [1, 2]:
        n = 0
        while n < i:
            n += 1
            acc += n
    return acc


def while_nested_for() -> int:
    acc = 0
    n = 0
    while n < 2:
        n += 1
        for v in [10, 20]:
            acc += v
    return acc


def for_nested_try_finally() -> int:
    acc = 0
    for i in [1, 2]:
        try:
            acc += i
        finally:
            acc += 10
    return acc


def while_nested_try_finally() -> int:
    acc = 0
    n = 0
    while n < 2:
        n += 1
        try:
            acc += n
        finally:
            acc += 10
    return acc


def for_comprehensions() -> int:
    acc = 0
    for i in [1, 2]:
        acc += sum([v for v in [1, 2, 3]])
        acc += len({v % 2 for v in [1, 2, 3, 4]})
        acc += len({v: v * 2 for v in [1, 2, 3]})
        acc += sum(v for v in [4, 5])
    return acc


def while_comprehensions() -> int:
    acc = 0
    n = 0
    while n < 2:
        n += 1
        acc += sum([v for v in [1, 2, 3]])
        acc += sum(v for v in [4, 5])
    return acc


def for_nested_def() -> int:
    acc = 0
    for i in [1, 2]:
        def inner(x: int) -> int:
            return x * 3

        acc += inner(i)
    return acc


def for_nested_gen() -> int:
    acc = 0
    for i in [1, 2]:
        def counter() -> "object":
            yield 7
            yield 8

        acc += sum(counter())
    return acc


def for_nested_lambda() -> int:
    acc = 0
    for i in [1, 2]:
        nine = lambda: 9
        acc += nine()
    return acc


def for_continue() -> int:
    acc = 0
    for i in [1, 2, 3, 4]:
        if i % 2 == 0:
            continue
        acc += i
    return acc


print(for_nested_while())
print(while_nested_for())
print(for_nested_try_finally())
print(while_nested_try_finally())
print(for_comprehensions())
print(while_comprehensions())
print(for_nested_def())
print(for_nested_gen())
print(for_nested_lambda())
print(for_continue())
