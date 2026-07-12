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
"""

import sys

__all__ = [
    "name", "curdir", "pardir", "extsep", "sep", "pathsep", "defpath",
    "linesep", "devnull", "getpid", "getppid", "getuid", "geteuid", "getgid",
    "getegid",
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
