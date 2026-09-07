# WHAT: the StopIteration that ends a generator carries a frame from an EARLIER
# next() that returned normally. CPython prints one frame (the call that
# raised); this prints two, the second naming a line that succeeded.
#
# MEASURED 2026-09-07 (RelWithDebInfo). ⭐ THE STALE FRAME IS THE FIRST next()
# IN THE PROGRAM, not the first on the generator that raised:
#
#   next, next, next (3rd raises) ............. extra frame = the 1st next
#   next x3 then a 4th ........................ extra frame = the 1st next
#   next then send then send .................. extra frame = the 1st next
#   next(a) once, THEN next(b) to exhaustion .. extra frame = next(A), which
#                                               is a different generator
#   a caught ValueError between the nexts ..... does NOT clear it
#   for v in g(): ... then an unrelated raise .. correct (no extra frame)
#   list(g()) then an unrelated raise ......... correct
#   two ordinary calls then a raise ........... correct
#   raise/except/pass then an uncaught raise ... correct (so the ordinary
#                                                clear works)
#
# So one frame is pushed by the first hand-written generator resume in the
# program and never removed -- not by the clear an ordinary caught exception
# performs, and not by the raise that reports it. Frames are pushed at RAISE
# sites and the runtime has no per-call pop (`LyTraceback_Pop` is reachable
# only from `LyTraceback_Clear`'s drain loop).
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
