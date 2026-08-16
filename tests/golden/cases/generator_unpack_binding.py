# What this pins: a generator whose locals are bound by an UNPACK.
#
#     def fib():
#         a, b = 0, 1
#         while True:
#             yield a
#             a, b = b, a + b
#     # static type builtins.object does not provide manifest method '__gt__'
#
# The walk that decides a generator's yield type binds the names a `yield` will
# read, and it only ever looked at a bare `Name` target. So `a = 0` followed by
# `b = 1` typed the generator `Iterator[int]` and `a, b = 0, 1` typed it
# `Iterator[object]` -- the most common generator there is, refused at the
# first thing its consumer does with the value.
#
# The right-hand side is read POSITIONALLY when it is a literal of the same
# arity, which is exact and does not depend on how a heterogeneous tuple is
# spelled; otherwise the value's type is distributed, a positional
# `tuple[A, B]` by position and a one-argument container to every name.
#
# Why this needs to run rather than assert on a diagnostic: the yield type is
# what the CONSUMER compiles against, and it is a type the generator body never
# mentions. `Iterator[int]` and `Iterator[object]` both build; only the value
# the loop prints says which one the frame carried, and a wrong positional
# distribution (`a, b = "x", 1` binding a to int) prints the other member.
#
# Every expected line is python3.14's.

# --- the fibonacci generator, in both spellings ----------------------------
def fib():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b


out = []
for v in fib():
    if v > 30:
        break
    out.append(v)
print(out)


def fib_enumerated():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b


pairs = []
for i, v in enumerate(fib_enumerated()):
    if i >= 6:
        break
    pairs.append((i, v))
print(pairs)


# --- a HETEROGENEOUS unpack, which is where position matters ---------------
def three():
    a, b, c = "x", 1, 2.5
    yield a
    yield str(b)
    yield str(c)


print(list(three()))


# --- the swap alone, and a swap through a temporary ------------------------
def swapped():
    a, b = 1, 2
    a, b = b, a
    yield a
    yield b


def through_temp():
    a, b = 0, 1
    while True:
        yield a
        t = a + b
        a = b
        b = t


print(list(swapped()))
steps = []
for v in through_temp():
    if v > 10:
        break
    steps.append(v)
print(steps)


# --- a `for` target that is a tuple, which is the same binding -------------
def enumerated_text():
    for i, ch in enumerate("abc"):
        yield f"{i}{ch}"


print(list(enumerated_text()))


# --- THE CONTROL: the separate-binding spelling, which always worked -------
def separate():
    a = 0
    b = 1
    while True:
        yield a
        a, b = b, a + b


kept = []
for v in separate():
    if v > 10:
        break
    kept.append(v)
print(kept)
