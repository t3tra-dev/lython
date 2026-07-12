"""Native stack-overflow guard, written in Python.

Pre-lowered per target platform at build time (tools/CMakeLists.txt); the
`sys.platform` branches fold against the --target triple, so each artifact
carries only its own platform's layouts and libc symbols.

`LyRt_InstallStackGuard` is called once from runtime startup
(LyRunPythonMain, built by RuntimeSupportBuilder).
It captures the main thread's stack bounds, installs an alternate signal
stack, and registers `LyStackGuard_Handler` (unix) or `LyStackGuard_Filter`
(windows). The handler bodies are compiler-verified async-signal-safe: they
hold no GC objects and only perform raw loads plus calls through function
pointers pre-resolved at install time into module globals.

Layout notes (mirrors the retired hand-written native modules):
  - darwin  stack_t {ss_sp, ss_size, ss_flags}; sigaction {handler, mask:u32,
    flags}; si_addr at siginfo+24; SIGBUS=10; SA_ONSTACK|SA_SIGINFO = 0x41.
  - linux   glibc stack_t {ss_sp, ss_flags, pad, ss_size}; sigaction
    {handler, mask:128B, flags, restorer}; si_addr at siginfo+16; SIGBUS=7;
    SA_ONSTACK|SA_SIGINFO = 0x08000004.
  - windows SetUnhandledExceptionFilter; EXCEPTION_STACK_OVERFLOW =
    0xC00000FD (-1073741571 signed).

The RecursionError message is materialized into a malloc'd buffer at install
time as packed 8-byte little-endian words (no str objects may reach the
handler).
"""

import ctypes
import sys

g_installed: int = 0
g_limit: int = 0
g_msg: int = 0
g_write: int = 0
g_exit: int = 0
g_signal: int = 0
g_raise: int = 0
g_stderr: int = 0
g_write_file: int = 0
g_exit_process: int = 0


class DarwinStackT(ctypes.Structure):
    _fields_ = [
        ("ss_sp", ctypes.c_void_p),
        ("ss_size", ctypes.c_long),
        ("ss_flags", ctypes.c_int),
    ]


class DarwinSigAction(ctypes.Structure):
    _fields_ = [
        ("sa_handler", ctypes.c_void_p),
        ("sa_mask", ctypes.c_uint),
        ("sa_flags", ctypes.c_int),
    ]


class LinuxStackT(ctypes.Structure):
    _fields_ = [
        ("ss_sp", ctypes.c_void_p),
        ("ss_flags", ctypes.c_int),
        ("ss_pad", ctypes.c_int),
        ("ss_size", ctypes.c_long),
    ]


class LinuxSigAction(ctypes.Structure):
    _fields_ = [
        ("sa_handler", ctypes.c_void_p),
        ("m0", ctypes.c_long),
        ("m1", ctypes.c_long),
        ("m2", ctypes.c_long),
        ("m3", ctypes.c_long),
        ("m4", ctypes.c_long),
        ("m5", ctypes.c_long),
        ("m6", ctypes.c_long),
        ("m7", ctypes.c_long),
        ("m8", ctypes.c_long),
        ("m9", ctypes.c_long),
        ("m10", ctypes.c_long),
        ("m11", ctypes.c_long),
        ("m12", ctypes.c_long),
        ("m13", ctypes.c_long),
        ("m14", ctypes.c_long),
        ("m15", ctypes.c_long),
        ("sa_flags", ctypes.c_int),
        ("sa_pad", ctypes.c_int),
        ("sa_restorer", ctypes.c_void_p),
    ]


class LinuxPthreadAttr(ctypes.Structure):
    _fields_ = [
        ("a0", ctypes.c_long),
        ("a1", ctypes.c_long),
        ("a2", ctypes.c_long),
        ("a3", ctypes.c_long),
        ("a4", ctypes.c_long),
        ("a5", ctypes.c_long),
        ("a6", ctypes.c_long),
        ("a7", ctypes.c_long),
        ("a8", ctypes.c_long),
        ("a9", ctypes.c_long),
        ("a10", ctypes.c_long),
        ("a11", ctypes.c_long),
        ("a12", ctypes.c_long),
        ("a13", ctypes.c_long),
        ("a14", ctypes.c_long),
        ("a15", ctypes.c_long),
    ]


def LyStackGuard_HandlerDarwin(sig: int, info: int, ctx: int) -> int:
    if sys.platform != "darwin":
        return 0
    if info == 0:
        return 0
    lo = g_limit
    if lo == 0:
        return 0
    fault = ctypes.c_long.from_address(info + 24).value
    if fault >= lo - 262144:
        if fault < lo + 4096:
            WPROTO = ctypes.CFUNCTYPE(
                ctypes.c_long, ctypes.c_int, ctypes.c_void_p, ctypes.c_long
            )
            WPROTO(g_write)(2, g_msg, 73)
            EPROTO = ctypes.CFUNCTYPE(None, ctypes.c_int)
            EPROTO(g_exit)(1)
            return 0
    # Not a guard hit: restore the default disposition and re-raise so the
    # process still crashes the normal way.
    SPROTO = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p)
    RPROTO = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int)
    if sig == 10:
        SPROTO(g_signal)(10, 0)
        RPROTO(g_raise)(10)
        return 0
    SPROTO(g_signal)(11, 0)
    RPROTO(g_raise)(11)
    return 0


def LyStackGuard_HandlerLinux(sig: int, info: int, ctx: int) -> int:
    if sys.platform != "linux":
        return 0
    if info == 0:
        return 0
    lo = g_limit
    if lo == 0:
        return 0
    fault = ctypes.c_long.from_address(info + 16).value
    if fault >= lo - 262144:
        if fault < lo + 4096:
            WPROTO = ctypes.CFUNCTYPE(
                ctypes.c_long, ctypes.c_int, ctypes.c_void_p, ctypes.c_long
            )
            WPROTO(g_write)(2, g_msg, 73)
            EPROTO = ctypes.CFUNCTYPE(None, ctypes.c_int)
            EPROTO(g_exit)(1)
            return 0
    SPROTO = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p)
    RPROTO = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int)
    if sig == 7:
        SPROTO(g_signal)(7, 0)
        RPROTO(g_raise)(7)
        return 0
    SPROTO(g_signal)(11, 0)
    RPROTO(g_raise)(11)
    return 0


def LyStackGuard_Filter(pointers: int) -> int:
    if sys.platform != "win32":
        return 0
    if pointers == 0:
        return 0
    record = ctypes.c_long.from_address(pointers).value
    if record == 0:
        return 0
    code = ctypes.c_int.from_address(record).value
    if code != -1073741571:
        return 0
    written = ctypes.c_int(0)
    WPROTO = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
    )
    WPROTO(g_write_file)(g_stderr, g_msg, 74, ctypes.addressof(written), 0)
    EPROTO = ctypes.CFUNCTYPE(None, ctypes.c_uint)
    EPROTO(g_exit_process)(1)
    return 0


def _LyStackGuard_StoreQword(addr: int, value: int) -> None:
    cell = ctypes.c_long.from_address(addr)
    cell.value = value


def _LyStackGuard_WriteMessage(buf: int, windows_newline: int) -> None:
    # "RecursionError: maximum recursion depth exceeded (native stack
    # overflow)\n" packed as little-endian 8-byte words (73 bytes + NUL pad;
    # the windows variant ends "\r\n", 74 bytes).
    _LyStackGuard_StoreQword(buf, 8028074746197534034)
    _LyStackGuard_StoreQword(buf + 8, 2322294380849939822)
    _LyStackGuard_StoreQword(buf + 16, 2336652894456537453)
    _LyStackGuard_StoreQword(buf + 24, 8028074746197534066)
    _LyStackGuard_StoreQword(buf + 32, 2335244432877822062)
    _LyStackGuard_StoreQword(buf + 40, 7234298763096062053)
    _LyStackGuard_StoreQword(buf + 48, 7311146993654310944)
    _LyStackGuard_StoreQword(buf + 56, 8007518212045697824)
    _LyStackGuard_StoreQword(buf + 64, 2987979389149537654)
    if windows_newline != 0:
        _LyStackGuard_StoreQword(buf + 72, 2573)
        return
    _LyStackGuard_StoreQword(buf + 72, 10)


def LyRt_InstallStackGuard() -> None:
    global g_installed, g_limit, g_msg, g_write, g_exit, g_signal, g_raise
    global g_stderr, g_write_file, g_exit_process
    if g_installed != 0:
        return
    g_installed = 1
    libc = ctypes.CDLL(None)

    malloc_fn = libc["malloc"]
    malloc_fn.restype = ctypes.c_void_p
    malloc_fn.argtypes = [ctypes.c_long]
    maddr: int = ctypes.cast(malloc_fn, ctypes.c_void_p).value
    MPROTO = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_long)
    msgbuf: int = ctypes.cast(MPROTO(maddr)(80), ctypes.c_void_p).value
    if msgbuf == 0:
        return

    if sys.platform == "win32":
        _LyStackGuard_WriteMessage(msgbuf, 1)
        g_msg = msgbuf
        get_std_handle = libc["GetStdHandle"]
        get_std_handle.restype = ctypes.c_void_p
        get_std_handle.argtypes = [ctypes.c_int]
        stderr_handle: int = ctypes.cast(
            get_std_handle(-12), ctypes.c_void_p
        ).value
        g_stderr = stderr_handle
        write_file = libc["WriteFile"]
        write_file.restype = ctypes.c_int
        write_file.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        g_write_file = ctypes.cast(write_file, ctypes.c_void_p).value
        exit_process = libc["ExitProcess"]
        exit_process.restype = None
        exit_process.argtypes = [ctypes.c_uint]
        g_exit_process = ctypes.cast(exit_process, ctypes.c_void_p).value

        FPROTO = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
        cb = FPROTO(LyStackGuard_Filter)
        set_filter = libc["SetUnhandledExceptionFilter"]
        set_filter.restype = ctypes.c_void_p
        set_filter.argtypes = [ctypes.c_void_p]
        set_filter(ctypes.cast(cb, ctypes.c_void_p).value)
        return

    _LyStackGuard_WriteMessage(msgbuf, 0)
    g_msg = msgbuf

    write_fn = libc["write"]
    write_fn.restype = ctypes.c_long
    write_fn.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_long]
    g_write = ctypes.cast(write_fn, ctypes.c_void_p).value
    exit_fn = libc["_exit"]
    exit_fn.restype = None
    exit_fn.argtypes = [ctypes.c_int]
    g_exit = ctypes.cast(exit_fn, ctypes.c_void_p).value
    signal_fn = libc["signal"]
    signal_fn.restype = ctypes.c_void_p
    signal_fn.argtypes = [ctypes.c_int, ctypes.c_void_p]
    g_signal = ctypes.cast(signal_fn, ctypes.c_void_p).value
    raise_fn = libc["raise"]
    raise_fn.restype = ctypes.c_int
    raise_fn.argtypes = [ctypes.c_int]
    g_raise = ctypes.cast(raise_fn, ctypes.c_void_p).value

    if sys.platform == "darwin":
        self_fn = libc["pthread_self"]
        self_fn.restype = ctypes.c_void_p
        self_fn.argtypes = []
        stackaddr_fn = libc["pthread_get_stackaddr_np"]
        stackaddr_fn.restype = ctypes.c_void_p
        stackaddr_fn.argtypes = [ctypes.c_void_p]
        stacksize_fn = libc["pthread_get_stacksize_np"]
        stacksize_fn.restype = ctypes.c_long
        stacksize_fn.argtypes = [ctypes.c_void_p]
        me = self_fn()
        top: int = ctypes.cast(stackaddr_fn(me), ctypes.c_void_p).value
        limit = top - stacksize_fn(me)
        if limit == 0:
            return
        g_limit = limit

        altbuf: int = ctypes.cast(MPROTO(maddr)(524288), ctypes.c_void_p).value
        if altbuf == 0:
            return
        alt = DarwinStackT()
        alt.ss_sp = altbuf
        alt.ss_size = 524288
        alt.ss_flags = 0
        sigaltstack_fn = libc["sigaltstack"]
        sigaltstack_fn.restype = ctypes.c_int
        sigaltstack_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        if sigaltstack_fn(ctypes.addressof(alt), 0) != 0:
            return

        HPROTO = ctypes.CFUNCTYPE(
            ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p
        )
        cb = HPROTO(LyStackGuard_HandlerDarwin)
        act = DarwinSigAction()
        act.sa_handler = ctypes.cast(cb, ctypes.c_void_p).value
        act.sa_mask = 0
        act.sa_flags = 65
        sigaction_fn = libc["sigaction"]
        sigaction_fn.restype = ctypes.c_int
        sigaction_fn.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        sigaction_fn(11, ctypes.addressof(act), 0)
        sigaction_fn(10, ctypes.addressof(act), 0)
        return

    self_fn = libc["pthread_self"]
    self_fn.restype = ctypes.c_void_p
    self_fn.argtypes = []
    getattr_fn = libc["pthread_getattr_np"]
    getattr_fn.restype = ctypes.c_int
    getattr_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    getstack_fn = libc["pthread_attr_getstack"]
    getstack_fn.restype = ctypes.c_int
    getstack_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    destroy_fn = libc["pthread_attr_destroy"]
    destroy_fn.restype = ctypes.c_int
    destroy_fn.argtypes = [ctypes.c_void_p]

    attr = LinuxPthreadAttr()
    me = self_fn()
    if getattr_fn(me, ctypes.addressof(attr)) != 0:
        return
    stackaddr = ctypes.c_long(0)
    stacksize = ctypes.c_long(0)
    rc = getstack_fn(
        ctypes.addressof(attr),
        ctypes.addressof(stackaddr),
        ctypes.addressof(stacksize),
    )
    destroy_fn(ctypes.addressof(attr))
    if rc != 0:
        return
    limit = stackaddr.value
    if limit == 0:
        return
    g_limit = limit

    altbuf: int = ctypes.cast(MPROTO(maddr)(524288), ctypes.c_void_p).value
    if altbuf == 0:
        return
    alt = LinuxStackT()
    alt.ss_sp = altbuf
    alt.ss_flags = 0
    alt.ss_pad = 0
    alt.ss_size = 524288
    sigaltstack_fn = libc["sigaltstack"]
    sigaltstack_fn.restype = ctypes.c_int
    sigaltstack_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    if sigaltstack_fn(ctypes.addressof(alt), 0) != 0:
        return

    HPROTO = ctypes.CFUNCTYPE(
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p
    )
    cb = HPROTO(LyStackGuard_HandlerLinux)
    act = LinuxSigAction()
    act.sa_handler = ctypes.cast(cb, ctypes.c_void_p).value
    act.sa_flags = 134217732
    act.sa_restorer = 0
    sigaction_fn = libc["sigaction"]
    sigaction_fn.restype = ctypes.c_int
    sigaction_fn.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    sigaction_fn(11, ctypes.addressof(act), 0)
    sigaction_fn(7, ctypes.addressof(act), 0)
