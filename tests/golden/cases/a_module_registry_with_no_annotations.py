# A module-level container filled by the functions that use it takes its
# element from them, and two things stood between that answer and the program.
#
# The scan asked each callable with its parameters read off the ANNOTATIONS, so
# a top-level parameter that has a type only because the module fixpoint gave
# it one seeded the registry with `object`:
#
#     TASKS = []
#     def add(name, at):
#         TASKS.append((name, at))
#
# And every top-level signature was memoized before the answer existed, by
# sweeps that saw the global at its erased element -- so a body that MEASURES
# it recorded a failure there, and the memo reported it at emission for a
# program that is now fine:
#
#     def mean():
#         if len(SAMPLES) == 0:      # len(builtins.object)
#             return 0.0
#
# Those signatures are forgotten when the global is seeded, and recomputed
# where they are declared, which is what the generator arm of
# `emitTopLevelDeclarations` already does for the same reason. The result
# VARIABLE goes with the memo: it holds whatever the first walk answered, and
# unifying a second answer against it is "cannot unify builtins.object with
# builtins.str" rather than a correction.
#
# Why execution: the element type decides what may be read back out, and the
# averages below are what the bodies compute -- a float only because the
# samples are ints.
#
# ⛔ A parameter still holding an inference variable is left unbound: that is
# the fixpoint saying it does not know, and binding it would put a variable
# where the scan expects a type.

TASKS = []
DONE = {}
SAMPLES = []


def add(name, at):
    TASKS.append((name, at))


def fire(now):
    fired = []
    for t in TASKS:
        if t[1] <= now:
            fired.append(t[0])
            DONE[t[0]] = now
    return fired


def record(v):
    SAMPLES.append(v)


def mean():
    if len(SAMPLES) == 0:
        return 0.0
    total = 0
    for s in SAMPLES:
        total += s
    return total / len(SAMPLES)


def spread():
    lo = SAMPLES[0]
    hi = SAMPLES[0]
    for s in SAMPLES:
        if s < lo:
            lo = s
        if s > hi:
            hi = s
    return hi - lo


add("save", 1)
add("load", 3)
print(fire(2))
print(sorted(DONE.items()))
print(TASKS[1][0] + "!")

for v in [4, 8, 15]:
    record(v)
print(mean())
print(spread())
print(SAMPLES[0] + 1)
