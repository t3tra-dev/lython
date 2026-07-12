"""The nt (Windows) process-identity natives, Lython style.

CPython implements the `nt` module in C (Modules/posixmodule.c compiled for
Windows) and `os.py` re-exports it. Here the natives are compiler-verified
ctypes calls against msvcrt. On non-Windows targets the branches fold away
and the function returns the -1 sentinel, keeping the module importable but
inert (CPython has no nt module there at all).
"""

import ctypes
import sys

__all__ = ["getpid"]


def getpid() -> int:
    """Return the current process id."""
    if sys.platform != "win32":
        return -1
    libc = ctypes.CDLL(None)
    f = libc["_getpid"]
    f.restype = ctypes.c_int
    f.argtypes = []
    return f()
