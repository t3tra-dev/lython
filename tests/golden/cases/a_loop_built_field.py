# What this pins: a local built by a loop and then stored into a field.
#
#     class B:
#         def __init__(self, n: int) -> None:
#             raw = ""
#             for i in range(n):
#                 raw = raw + "x"
#             self.raw: str = raw
#
#     # owned resource from @LyUnicode_FromBytes result 0 is released or
#     # transferred more than once on one CFG path
#
# The field store may take the value's reference when nothing else needs it,
# and it decides that by asking whether any USE of the source is dominated by
# the store. A loop-carried value has no such use -- its other uses are the
# loop's own, which the store does not dominate -- so the store read "nobody
# else needs this", took the token, and the loop released the same reference
# again. A block argument is the signal: the store did not produce that value,
# and it may only move a token whose whole life it can see.
#
# Why this must run: the same reasoning also has to keep the store from
# LEAKING when the source really is a temporary, and net zero over a few
# thousand constructions is the only way to see both halves at once
# (tests/leak_gate.py reads 0 for this file).
#
# ⛔ The if/else merge is the same shape and is here for it: two arms each hand
# over a reference the frame still owns, and the empty loop trip is the case
# where the seed itself reaches the field.
class Joined:
    def __init__(self, n: int) -> None:
        raw = ""
        for i in range(n):
            raw = raw + "x"
        self.raw: str = raw


class Branched:
    def __init__(self, flag: bool, a: str) -> None:
        t = a
        if flag:
            t = a + "!"
        self.raw: str = t


class Collected:
    def __init__(self, xs: list[str]) -> None:
        out: list[str] = []
        for x in xs:
            out.append(x + "")
        self.items: list[str] = out


print(Joined(0).raw, repr(Joined(0).raw), Joined(3).raw)
print(Branched(True, "a").raw, Branched(False, "a").raw)
print(Collected(["a", "b"]).items, Collected([]).items)

total = 0
i = 0
while i < 400:
    total += len(Joined(i % 4).raw)
    total += len(Branched(i % 2 == 0, "seed").raw)
    total += len(Collected(["a", "b"]).items)
    i += 1
print("loop", total)
