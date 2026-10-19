# WHAT: An exception raised in a `with catch():` body and handled in the
#   @contextmanager's `except`: the traceback there (sys.exc_info()[2])
#   holds only the generator's own frame. CPython's also holds the frames of
#   the body down to the raise -- ['catch', '<module>', 'outer', 'inner']. A
#   plain try/except keeps them. unittest reports every failure from such a
#   handler, so its tracebacks show no test frames.
# KIND: バグ (bug) -- silent: frames are missing
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# FIXED: 2026-10-20 -- the exception thrown into the generator keeps the frames
#   the caller's handler had parked (cases/a_context_managers_handler_sees_the_with_bodys_frames).
# CLASSIFICATION @ lang-gaps 90ef0204 + the fix: 1 正しい
# CPython 3.14 expects: ['catch', '<module>', 'outer', 'inner']
import contextlib
import sys
import traceback
from collections.abc import Iterator


@contextlib.contextmanager
def catch() -> Iterator[None]:
    try:
        yield
    except ValueError:
        tb = sys.exc_info()[2]
        names = [entry.split(", in ")[1].split("\n")[0] for entry in traceback.format_tb(tb)]
        print(names)


def inner() -> None:
    raise ValueError("x")


def outer() -> None:
    inner()


with catch():
    outer()
