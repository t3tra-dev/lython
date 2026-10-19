# WHAT: An exception raised in a `with catch():` body and handled in the
#   @contextmanager's `except`: the traceback there (sys.exc_info()[2])
#   holds only the generator's own frame. CPython's also holds the frames of
#   the body down to the raise -- ['catch', '<module>', 'outer', 'inner']. A
#   plain try/except keeps them. unittest reports every failure from such a
#   handler, so its tracebacks show no test frames.
# KIND: バグ (bug) -- silent: frames are missing
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# differential: expect-wrong the frames of the with-body are missing from the handler's traceback
# CLASSIFICATION @ lang-gaps fa5c535a: 2 silent 誤実行
#   cpython="['catch', '<module>', 'outer', 'inner']\n" lyc="['catch']\n"
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
