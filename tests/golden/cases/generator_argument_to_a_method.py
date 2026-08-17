# What this pins: a generator OBJECT passed to a manifest method that takes an
# iterable.
#
#     "-".join(g())
#     # cannot adapt types.GeneratorType to runtime input 2 of
#     # builtins.str.join
#
# `"-".join(list(g()))` worked, and so did `"-".join(x for x in xs)` -- a
# generator EXPRESSION in argument position fuses into the callee and never
# becomes an object. The gap was the generator object itself, which has no
# physical shape any manifest parameter declares.
#
# Every manifest method that takes an iterable consumes the whole of it, so
# materializing the generator is exact rather than a change of semantics, and
# `list(...)` is surface that already compiles. Manifest receivers only: a
# source class's method may hold a generator and consume it lazily, which is
# its own business.
#
# Why this needs to run rather than assert on a diagnostic: the rewrite decides
# WHEN the generator runs, and a generator's body has side effects. The counter
# below is incremented per element, so the printed count says the generator ran
# exactly once and to the end -- a materialization that ran it twice, or that
# left it half-consumed, prints a different number.
#
# Every expected line is python3.14's.


def words():
    yield "a"
    yield "bb"
    yield "c"


def numbers():
    yield 3
    yield 1
    yield 2


# --- str.join, which is where this shows up -------------------------------
print("-".join(words()))
print(", ".join(words()))
print("".join(words()))


# --- list.extend, and the same through a variable -------------------------
collected: list[str] = []
collected.extend(words())
print(collected)

more: list[int] = [9]
more.extend(numbers())
print(more)


# --- the generator runs ONCE and to the end -------------------------------
calls = 0


def counted():
    global calls
    i = 0
    while i < 4:
        calls += 1
        yield str(i)
        i += 1


print("|".join(counted()), calls)


# --- THE CONTROL: the two spellings that always worked --------------------
print("-".join(list(words())))
print("-".join(x for x in ["a", "b"]))
print("-".join(["a", "b"]))
