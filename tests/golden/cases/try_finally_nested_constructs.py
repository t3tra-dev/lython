# Constructs nested inside a `finally` clause, each writing a local that the
# code after the `try` statement reads back.
#
# Why execution is needed: a `finally` clause runs on both the normal and the
# unwind path, so the refcount machinery has to place the same local's release
# on two paths that rejoin. Nothing before execution distinguishes a correct
# placement from a double release -- `tests/probe/wb_finally_loop_local_unwind.py`
# records the `for` spelling of exactly this shape compiling and verifying
# clean and then SIGSEGVing 5/5.
#
# Why these cells: `tests/probe/tools/nestgrid.py` reports `try.finally` in 16
# of 292 golden cases and `for` in 87, with no file nesting one in the other,
# and no file anywhere coupling a `finally` to a sibling region by a shared
# local. The nine cells below were all empty and are all correct today (5/5
# against CPython). `try.finally > for` and `> while` are deliberately absent:
# they crash, and a red golden is not something to commit.
def nested_try() -> int:
    acc = 0
    try:
        acc += 1
    finally:
        try:
            acc += 2
        except ZeroDivisionError:
            acc += 100
    return acc


def nested_try_finally() -> int:
    acc = 0
    try:
        acc += 1
    finally:
        try:
            acc += 2
        finally:
            acc += 4
    return acc


def comp_list() -> int:
    acc = 0
    try:
        acc += 1
    finally:
        acc += sum([v for v in [1, 2, 3]])
    return acc


def comp_set() -> int:
    acc = 0
    try:
        acc += 1
    finally:
        acc += len({v % 2 for v in [1, 2, 3, 4]})
    return acc


def comp_dict() -> int:
    acc = 0
    try:
        acc += 1
    finally:
        acc += len({v: v * 2 for v in [1, 2, 3]})
    return acc


def comp_gen() -> int:
    acc = 0
    try:
        acc += 1
    finally:
        acc += sum(v for v in [4, 5])
    return acc


def nested_def() -> int:
    acc = 0
    try:
        acc += 1
    finally:
        def inner(x: int) -> int:
            return x * 3

        acc += inner(4)
    return acc


def nested_gen() -> int:
    acc = 0
    try:
        acc += 1
    finally:
        def counter() -> "object":
            yield 7
            yield 8

        acc += sum(counter())
    return acc


def nested_lambda() -> int:
    acc = 0
    try:
        acc += 1
    finally:
        nine = lambda: 9
        acc += nine() + nine()
    return acc


print(nested_try())
print(nested_try_finally())
print(comp_list())
print(comp_set())
print(comp_dict())
print(comp_gen())
print(nested_def())
print(nested_gen())
print(nested_lambda())
