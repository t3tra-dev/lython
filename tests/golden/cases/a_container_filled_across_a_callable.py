# An empty container literal takes its element type from the operations that
# fill it, and the scan that finds them stopped at the callable boundary --
# because the same name in an enclosing function is usually a different
# binding. It is not one when the callable never BINDS the name, which is the
# whole of a module registry and of an accumulator built by a closure:
#
#     XS = []
#     def put(n: int) -> None:
#         XS.append(n)
#     put(1)
#     print(XS[0] + 1)
#     # builtins.object does not provide manifest method '__add__'
#
# For a GLOBAL the answer also has to arrive earlier than the assignment: the
# cell's type is decided before any body is emitted, so seeding it where a
# local is seeded would be after every read had been typed against the erased
# element.
#
# Why execution: the element type decides what may be read back out, so the
# program has to DECODE what it stored. The counters and concatenations below
# are the assertions.
#
# ⭐ Each callable is asked with its OWN parameters in scope -- the seed is
# usually one of them -- and two that disagree leave the container erased,
# which is the rule the scan already applies to two seeds in one suite.
#
# ⛔ A callable that binds the name is skipped. In Python an assignment
# anywhere in a body makes the name local for the whole body, so a `LOG = []`
# inside one says nothing about the global -- unless it declares `global LOG`,
# which is the one spelling that puts the store back on it.

EVENTS = []
COUNTS = {}
SEEN = set()


def record(kind: str, weight: int) -> None:
    EVENTS.append(kind)
    SEEN.add(kind)
    if kind not in COUNTS:
        COUNTS[kind] = 0
    COUNTS[kind] = COUNTS[kind] + weight


def shadowed() -> int:
    EVENTS = ["not the global"]
    return len(EVENTS)


def collect(words: "list[str]") -> "list[str]":
    out = []

    def keep(word: str) -> None:
        out.append(word.upper())

    for w in words:
        if w != "":
            keep(w)
    return out


record("tick", 2)
record("tock", 3)
record("tick", 4)
print(EVENTS)
print(EVENTS[0] + "!")
print(COUNTS["tick"] + COUNTS["tock"])
print(sorted(SEEN))
print(shadowed())
picked = collect(["a", "", "bb"])
print(picked)
print(picked[1] + "?")
