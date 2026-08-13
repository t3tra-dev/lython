# A signal handler written entirely in Python, proving the stack-guard's core
# mechanism. The handler is async-signal-safe (compiler-verified): it holds no
# GC objects, reads a pre-resolved libc `write` pointer from a module global,
# builds its message on the stack (a c_long cell, no heap), and calls write
# through the address. Install resolves the pointer; a raised signal fires it.
#
# The global is a `c_void_p` and not an `int`: an `int` module global is a
# Python integer, so it is boxed and reading one allocates, which a signal
# handler may not do. A ctypes pointer type is the machine word.
import ctypes

g_write: ctypes.c_void_p = ctypes.cast(0, ctypes.c_void_p)


def handler(sig: int) -> int:
    msg = ctypes.c_long(2851465066542439)
    a = ctypes.addressof(msg)
    WPROTO = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_int, ctypes.c_void_p, ctypes.c_long)
    w = ctypes.cast(g_write, WPROTO)
    w(1, a, 7)
    return 0


def install() -> None:
    global g_write
    libc = ctypes.CDLL(None)
    write_fn = libc["write"]
    write_fn.restype = ctypes.c_long
    write_fn.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_long]
    g_write = ctypes.cast(write_fn, ctypes.c_void_p)
    HANDLER = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int)
    cb = HANDLER(handler)
    signal_fn = libc["signal"]
    signal_fn.restype = ctypes.c_void_p
    signal_fn.argtypes = [ctypes.c_int, ctypes.c_void_p]
    signal_fn(30, cb)


install()
libc = ctypes.CDLL(None)
raise_fn = libc["raise"]
raise_fn.restype = ctypes.c_int
raise_fn.argtypes = [ctypes.c_int]
raise_fn(30)
