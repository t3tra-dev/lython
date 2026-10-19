# WHAT: `sys.stdout = buf` (and `sys.stderr`) makes print write to `buf` until
# the stream is put back -- every argument, sep, end and flush of it, a bare
# print(), a `file=sys.stderr` print and `sys.stdout.write` alike -- whether
# the assignment is in this module, under an alias of `sys`, in a method, or
# in an imported module; `sys.__stdout__` is the stream the program started
# with.
#
# WHY THIS IS RUN: what print writes, and where, is the run's output.
import io
import sys
import sys as system

import a_replaced_sys_stdout_captures_print_lib as lib


class Result:
    """unittest.TestResult's buffering, in its shape."""

    def __init__(self) -> None:
        self._stdout_buffer = io.StringIO()
        self._stderr_buffer = io.StringIO()
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr

    def setup(self) -> None:
        sys.stdout = self._stdout_buffer
        sys.stderr = self._stderr_buffer

    def restore(self) -> str:
        sys.stdout = self._original_stdout
        system.stderr = self._original_stderr
        out = self._stdout_buffer.getvalue()
        err = self._stderr_buffer.getvalue()
        self._stdout_buffer.seek(0)
        self._stdout_buffer.truncate()
        return out + "|" + err


print("before")
r = Result()
r.setup()
print("one", 2, 3.5, None, sep=", ", end=";\n")
print()
print("to err", file=sys.stderr)
sys.stdout.write("direct\n")
xs: list[int] = [1, 2]
print(xs, {"k": 1}, (1,), flush=True)
captured = r.restore()
print("captured:", repr(captured))
r.setup()
print("again")
print("second:", repr(r.restore()))

text = lib.run_captured(3)
print("from the library:", repr(text))

sys.stdout = io.StringIO()
print("lost")
sys.stdout = sys.__stdout__
print("back on the stream the program started with")
