r"""OS routines for NT or Posix depending on what system we're on.

This is Lython's port of CPython's Lib/os.py, restricted to the well-typed
statically compilable surface. It ships as SOURCE inside the compiler:
`import os` resolves this file through the same path as user source modules
and compiles it with the program, so typing derives from the annotations
below, `sys.platform` switches fold against the target triple, and the
platform module re-export below binds the target's flavor exactly like
CPython's import-time dispatch.

This exports:
  - all functions from posix or nt, e.g. getpid, getuid, etc.
  - os.path is either posixpath or ntpath
  - os.name is either 'posix' or 'nt'
  - os.curdir is a string representing the current directory (always '.')
  - os.pardir is a string representing the parent directory (always '..')
  - os.sep is the (or a most common) pathname separator ('/' or '\\')
  - os.extsep is the extension separator (always '.')
  - os.pathsep is the component separator used in $PATH etc
  - os.linesep is the line separator in text files ('\n' or '\r\n')
  - os.defpath is the default search path for executables
  - os.devnull is the file path of the null device ('/dev/null', etc.)
  - os.altsep stays unsupported (None on posix; not a static string)

Deviations from CPython:
  - `environ` (and `environb`, `getenvb`) is absent. It is a computed dict,
    and a container-typed module global is not yet visible across an import
    boundary (reported to the Wave 3 foundation track). `getenv` / `putenv` /
    `unsetenv` read and write the process environment directly, so they stay
    consistent with each other; `environ_entries()` exposes the raw
    "KEY=VALUE" vector for callers that need to enumerate.
  - `getenv(key, default=None)` is CPython's, including the None. This said
    until 2026-08-17 that an `Optional[str]` return had no physical layout
    across the native boundary and defaulted to `''` -- which made an UNSET
    variable indistinguishable from one set to the empty string, so
    `os.getenv(k) is None` answered False where CPython answers True. The
    native call still returns a str; the None is the Python-level default and
    never crosses the boundary. `has_env(key)` remains the direct question.
  - `stat()` returns the `stat_result` class below, a plain class with the
    ten `st_*` attributes rather than a structseq (no tuple indexing, no
    `st_atime_ns` family, no `st_birthtime`). Each attribute read costs one
    stat(2) call because `posix._stat_field` fetches one field; see
    modules/posix.mlir for why. `os.path`'s predicates read st_mode only.
  - `walk()` is not a generator: generators defined in an imported module do
    not compile yet, so it returns the fully materialized list of
    `(dirpath, dirnames, filenames)` triples, top-down only. There is no
    `topdown=`, `onerror=` or `followlinks=` keyword, unreadable directories
    are skipped silently (CPython's default `onerror=None` does the same),
    and callers cannot prune the walk by mutating `dirnames`.
  - `path` is reachable as `os.path` but is NOT in `__all__`, so
    `from os import *` does not bind a bare `path` (CPython's does). Module
    members live in one flat symbol table that a local variable of the same
    name does not shadow, so exporting `path` would make every `path.foo(...)`
    on a local named `path` resolve to `posixpath.foo` -- including inside
    posixpath itself, whose own parameters are called `path`.
  - `open`, `fdopen`, `read`, `write`, `close`, the *at() family, `fork` and
    the process/signal surface are not ported. Use the `io` module's `open`.
"""

import sys
import posixpath as path

__all__ = [
    "name", "curdir", "pardir", "extsep", "sep", "pathsep", "defpath",
    "linesep", "devnull", "getpid", "getppid", "getuid", "geteuid", "getgid",
    "getegid", "getcwd", "chdir", "listdir", "mkdir", "makedirs", "rmdir",
    "remove", "unlink", "rename", "replace", "access", "stat", "lstat",
    "stat_result", "strerror", "getenv", "putenv", "unsetenv", "has_env",
    "environ_entries", "walk", "F_OK", "R_OK", "W_OK", "X_OK",
]

name: str = "nt" if sys.platform == "win32" else "posix"
curdir: str = "."
pardir: str = ".."
extsep: str = "."
sep: str = "\\" if sys.platform == "win32" else "/"
pathsep: str = ";" if sys.platform == "win32" else ":"
defpath: str = ".;C:\\bin" if sys.platform == "win32" else "/bin:/usr/bin"
linesep: str = "\r\n" if sys.platform == "win32" else "\n"
devnull: str = "nul" if sys.platform == "win32" else "/dev/null"

if sys.platform == "win32":
    from nt import *
else:
    from posix import *

import posix


class stat_result:
    """Result of os.stat / os.lstat: the ten portable st_* fields.

    CPython's is a structseq that also indexes like a 10-tuple; this is a
    plain class, so only attribute access works.
    """

    def __init__(self, path: str, follow: bool) -> None:
        flag = 1 if follow else 0
        self.st_mode: int = _stat_or_raise(path, flag, 0)
        self.st_ino: int = _stat_or_raise(path, flag, 1)
        self.st_dev: int = _stat_or_raise(path, flag, 2)
        self.st_nlink: int = _stat_or_raise(path, flag, 3)
        self.st_uid: int = _stat_or_raise(path, flag, 4)
        self.st_gid: int = _stat_or_raise(path, flag, 5)
        self.st_size: int = _stat_or_raise(path, flag, 6)
        self.st_atime: int = _stat_or_raise(path, flag, 7)
        self.st_mtime: int = _stat_or_raise(path, flag, 8)
        self.st_ctime: int = _stat_or_raise(path, flag, 9)


def _stat_or_raise(path: str, follow: int, index: int) -> int:
    """One stat_result field, raising the mapped OSError on failure.

    posix._stat_field reports `-errno` so os.path's predicates can stay
    silent; the raising callers turn that back into an exception through
    posix._raise_errno, which maps errno with the compiler's own table.
    """
    value = posix._stat_field(path, follow, index)
    if value < 0:
        posix._raise_errno(-value, path)
    return value


def stat(p: str) -> stat_result:
    """Perform a stat system call on the given path."""
    return stat_result(p, True)


def lstat(p: str) -> stat_result:
    """Like stat(), but do not follow symbolic links."""
    return stat_result(p, False)


def remove(p: str) -> None:
    """Remove a file (same as unlink())."""
    unlink(p)


def replace(src: str, dst: str) -> None:
    """Rename a file or directory, overwriting the destination.

    On posix rename(2) already replaces an existing destination, so this is
    rename; CPython's replace differs from rename on Windows only.
    """
    rename(src, dst)


def getenv(key: str, default: str | None = None) -> str | None:
    """Get an environment variable, returning `default` if it doesn't exist.

    ⛔ The last two lines are `if default is None: return None` / `return
    default` rather than one `return default`, and that is not a style. A
    borrowed union PARAMETER returned as an owned union RESULT leaks its str
    member -- 43 B, measured -- while the same function returning the narrowed
    member does not. Reproducer and the shape of the underlying defect are in
    tests/probe/wb_grid_leftovers_2026_08_16.py.
    """
    if posix._has_env(key):
        return posix._getenv(key)
    if default is None:
        return None
    return default


def has_env(key: str) -> bool:
    """True when the environment variable is set, even to the empty string."""
    return posix._has_env(key)


def environ_entries() -> list[str]:
    """The process environment as raw "KEY=VALUE" strings.

    Stands in for `os.environ` until a container-typed module global survives
    an import boundary.
    """
    return posix._environ_entries()


def makedirs(name: str, mode: int = 0o777, exist_ok: bool = False) -> None:
    """makedirs(name [, mode=0o777][, exist_ok=False])

    Super-mkdir; create a leaf directory and all intermediate ones.  Works
    like mkdir, except that any intermediate path segment (not just the
    rightmost) will be created if it does not exist.
    """
    head, tail = path.split(name)
    if tail == "":
        head, tail = path.split(head)
    if head != "" and tail != "" and not path.exists(head):
        makedirs(head, mode, True)
    if exist_ok and path.isdir(name):
        return
    mkdir(name, mode)


def _quiet_listdir(p: str) -> list[str]:
    """listdir that reports an unreadable directory as empty.

    CPython's walk() with the default `onerror=None` swallows the OSError and
    skips the directory. Asked in advance rather than caught: `return` inside
    a `try` does not compile yet, and the two questions access(2) answers --
    is it a directory, is it readable and searchable -- are the two reasons
    listdir would raise here.
    """
    if not path.isdir(p):
        return []
    if not access(p, R_OK | X_OK):
        return []
    return listdir(p)


def walk(top: str) -> list[tuple[str, list[str], list[str]]]:
    """Directory tree generator, top-down, fully materialized.

    Each element is a `(dirpath, dirnames, filenames)` triple, as CPython's
    generator yields. See the module docstring for why this is a list.
    """
    result: list[tuple[str, list[str], list[str]]] = []
    pending: list[str] = [top]
    # A read cursor rather than pop(0): the runtime has no list.pop, and the
    # queue keeps growing as subdirectories are discovered.
    cursor = 0
    while cursor < len(pending):
        current = pending[cursor]
        cursor = cursor + 1
        dirnames: list[str] = []
        filenames: list[str] = []
        entries: list[str] = _quiet_listdir(current)
        entries.sort()
        for entry in entries:
            if path.isdir(path.join(current, entry)):
                dirnames.append(entry)
            else:
                filenames.append(entry)
        result.append((current, dirnames, filenames))
        for name_ in dirnames:
            pending.append(path.join(current, name_))
    return result
