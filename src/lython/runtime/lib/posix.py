"""The posix process-identity natives, Lython style.

CPython implements the `posix` module in C (Modules/posixmodule.c) and
`os.py` re-exports it. Here each native is a compiler-verified ctypes call;
`os.py` pulls them in with the same `from posix import *`. On Windows targets
(where CPython has no posix module) the libc branches fold away and every
function returns the -1 sentinel, keeping the module importable but inert.
"""

import ctypes
import sys

__all__ = ["getpid", "getppid", "getuid", "geteuid", "getgid", "getegid"]


def getpid() -> int:
    """Return the current process id."""
    if sys.platform == "win32":
        return -1
    libc = ctypes.CDLL(None)
    f = libc["getpid"]
    f.restype = ctypes.c_int
    f.argtypes = []
    return f()


def getppid() -> int:
    """Return the parent's process id."""
    if sys.platform == "win32":
        return -1
    libc = ctypes.CDLL(None)
    f = libc["getppid"]
    f.restype = ctypes.c_int
    f.argtypes = []
    return f()


def getuid() -> int:
    """Return the current process's user id."""
    if sys.platform == "win32":
        return -1
    libc = ctypes.CDLL(None)
    f = libc["getuid"]
    f.restype = ctypes.c_int
    f.argtypes = []
    return f()


def geteuid() -> int:
    """Return the current process's effective user id."""
    if sys.platform == "win32":
        return -1
    libc = ctypes.CDLL(None)
    f = libc["geteuid"]
    f.restype = ctypes.c_int
    f.argtypes = []
    return f()


def getgid() -> int:
    """Return the current process's group id."""
    if sys.platform == "win32":
        return -1
    libc = ctypes.CDLL(None)
    f = libc["getgid"]
    f.restype = ctypes.c_int
    f.argtypes = []
    return f()


def getegid() -> int:
    """Return the current process's effective group id."""
    if sys.platform == "win32":
        return -1
    libc = ctypes.CDLL(None)
    f = libc["getegid"]
    f.restype = ctypes.c_int
    f.argtypes = []
    return f()
