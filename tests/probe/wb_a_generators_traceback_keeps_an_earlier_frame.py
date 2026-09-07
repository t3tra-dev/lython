# WHAT: the StopIteration that ends a generator carries a frame from an EARLIER
# next() that returned normally. CPython prints one frame (the call that
# raised); this prints two, the second naming a line that succeeded.
#
# MEASURED 2026-09-07 (RelWithDebInfo). The stale frame is always the FIRST
# next() on that generator, whatever comes between:
#
#   next, next (2 yields, 3rd call raises) ... extra frame = the 1st next
#   next, next, next .......................... extra frame = the 1st next
#   next then send then send .................. extra frame = the 1st next
#   for v in g(): ... then an unrelated raise .. correct (no extra frame)
#   list(g()) then an unrelated raise ......... correct
#   two DIFFERENT generators, one next each ... correct
#   two ordinary calls then a raise ........... correct
#
# So it takes a generator resumed by hand more than once. Frames are pushed at
# RAISE sites and the runtime has no per-call pop (`LyTraceback_Pop` is only
# reachable from `LyTraceback_Clear`'s drain loop), so a frame pushed by a
# raise that was caught inside the resume protocol has nothing to remove it.
#
# ⛔ NOT the inlined-method frame shape, which is the opposite defect (a frame
# that is MISSING): see wb_inlined_method_traceback_frame.
#
# The traceback is brought to STDOUT so the difference is a comparison and not
# two non-zero exits: an uncaught StopIteration makes both sides fail, which
# the differential reads as BOTH-FAIL and never compares.
import traceback
from typing import Iterator


def g() -> Iterator[int]:
    yield 1
    yield 2


def frame_lines(text: str) -> list[str]:
    out: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("File "):
            out.append(stripped[stripped.rfind(", line"):])
    return out


gen = g()
print(next(gen))
print(next(gen))
try:
    print(next(gen))
except StopIteration:
    print(frame_lines(traceback.format_exc()))
