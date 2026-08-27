# WHAT: `traceback._current_tb()` hands the exception being handled back as a
# chain of `types.TracebackType`, and `traceback.format_exception` lays it out
# the way CPython does.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: every assertion here is
# about a VALUE the runtime recorded -- which frames, in which order, with which
# lines -- and none of it exists before the program raises. The chain is also
# the first structure the compiler builds out of itself: each link holds the
# next one in an `Optional["TracebackType"]` field, so a wrong answer here is a
# wrong answer about boxed optional fields as much as about tracebacks.
#
# ⛔ Paths are printed as BASENAMES. The golden runner compares stdout exactly
# and the recorded filename is the absolute path the compiler saw.
import os
import sys
import traceback


def innermost() -> int:
    raise ValueError("boom")


def middle() -> int:
    return innermost()


def outermost() -> int:
    return middle()


try:
    outermost()
except ValueError as e:
    tb = traceback._current_tb()
    depth = 0
    cur = tb
    while cur is not None:
        code = cur.tb_frame.f_code
        print(os.path.basename(code.co_filename), code.co_name, cur.tb_lineno,
              cur.tb_lasti)
        depth += 1
        cur = cur.tb_next
    print("depth", depth)

    summary = traceback.extract_tb(tb)
    print("frames", len(summary.frames))
    print("first", summary.frames[0].name, summary.frames[0].line)
    print("last", summary.frames[3].name, summary.frames[3].line)

    limited = traceback.extract_tb(tb, 2)
    print("limited", len(limited.frames), limited.frames[1].name)

    for line in traceback.format_exception_only(e):
        sys.stdout.write(line)

# Outside a handler there is nothing in flight.
print("after", traceback._current_tb() is None, repr(traceback.format_exc()))
