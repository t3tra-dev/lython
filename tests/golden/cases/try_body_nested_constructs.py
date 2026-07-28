# Constructs nested inside a `try` BODY, each with a local the loop or
# comprehension writes and the `except` handler reads back.
#
# Why execution is needed: every value below is produced by the refcount
# insertion and unwind-cleanup machinery deciding where a loop-carried local
# dies. That decision is invisible to the parser, to the emitter and to the
# verifiers -- all four spellings compile and verify clean on a tree where the
# `for` spelling of the same shape SIGSEGVs. Only running it tells them apart.
#
# Why these cells: `tests/probe/tools/nestgrid.py` reports that of 292 golden
# cases, exactly ONE couples a nested construct to a sibling region by a shared
# local -- and the shipped SIGSEGV in
# `tests/probe/wb_forloop_handler_local_unwind.py` needs precisely that edge.
# `try.body > listcomp`, `> setcomp`, `> dictcomp`, `> genexp`, `> def`,
# `> gen` and `> lambda` were all empty cells; all seven are correct today
# (5/5 against CPython), so they are locked in here rather than left to be
# rediscovered. The `for` and `while` spellings of the same cell are NOT here:
# they crash, and a red golden is not something to commit.
def comp_list() -> int:
    acc = 0
    try:
        acc += sum([v for v in [1, 2, 3]])
    except ZeroDivisionError:
        acc += 100
    return acc


def comp_set() -> int:
    acc = 0
    try:
        acc += len({v % 2 for v in [1, 2, 3, 4]})
    except ZeroDivisionError:
        acc += 100
    return acc


def comp_dict() -> int:
    acc = 0
    try:
        acc += len({v: v * 2 for v in [1, 2, 3]})
    except ZeroDivisionError:
        acc += 100
    return acc


def comp_gen() -> int:
    acc = 0
    try:
        acc += sum(v for v in [4, 5])
    except ZeroDivisionError:
        acc += 100
    return acc


def nested_def() -> int:
    acc = 0
    try:
        def inner(x: int) -> int:
            return x * 3

        acc += inner(4)
    except ZeroDivisionError:
        acc += 100
    return acc


def nested_gen() -> int:
    acc = 0
    try:
        def counter() -> "object":
            yield 7
            yield 8

        acc += sum(counter())
    except ZeroDivisionError:
        acc += 100
    return acc


def nested_lambda() -> int:
    acc = 0
    try:
        # No parameter, because that is the spelling the sweep measured: a
        # parameterised lambda needs a Callable annotation here and would be
        # testing the annotation rule rather than the nesting.
        nine = lambda: 9
        acc += nine() + nine()
    except ZeroDivisionError:
        acc += 100
    return acc


print(comp_list())
print(comp_set())
print(comp_dict())
print(comp_gen())
print(nested_def())
print(nested_gen())
print(nested_lambda())
