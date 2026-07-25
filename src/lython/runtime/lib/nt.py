"""The nt (Windows) process-identity natives, Lython style.

CPython implements the `nt` module in C (Modules/posixmodule.c compiled for
Windows) and `os.py` re-exports it. Here the natives are compiler-verified
ctypes calls against msvcrt. On non-Windows targets the branches fold away
and the function returns the -1 sentinel, keeping the module importable but
inert (CPython has no nt module there at all).

This is now much SMALLER than its posix counterpart, which moved to a native
manifest (runtime/modules/posix.mlir) and grew the filesystem and environment
surface. The asymmetry is deliberate for the moment: a Windows target gets
`os.getpid` and the target-folded `os.name`/`os.sep` constants and nothing
else, and `os.path` would need an ntpath port. Growing this file the same way
means either a `nt` manifest beside posix's or Windows entry points in that
one -- the LyHost_* OS cluster it would call is already target-aware, and
HostTargetLayout marks a Windows triple as non-POSIX for exactly this reason.
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
