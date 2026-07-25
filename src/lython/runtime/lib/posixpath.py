"""Common operations on Posix pathnames.

This is Lython's port of CPython's Lib/posixpath.py, restricted to the
well-typed statically compilable surface. `os.py` publishes it as `os.path`
with `import posixpath as path`, exactly as CPython's does, so both
`posixpath.join(...)` and `os.path.join(...)` resolve to the same functions.

Instead of importing os.path use os.sep etc. as os.path is a
platform-dependent alias for this module.

Deviations from CPython:
  - `join(a, *p)` becomes `join(a, b, c, d)` with three optional components: a
    `*args` parameter read inside an imported module currently mis-executes
    (reported to the Wave 3 foundation track), and four components cover every
    real call. Each component behaves exactly as CPython's loop treats it,
    including `join('a', '')` == 'a/'; the unfilled parameters default to a NUL
    sentinel, which no path component can contain, rather than to '', so an
    explicit '' stays distinguishable from an omitted argument.
  - paths are `str` only. CPython's posixpath is generic over str and bytes
    through `os.fspath` and a `_get_sep` that switches on the argument's
    type; that is a runtime type test, so this module is the str
    instantiation.
  - `exists`/`isfile`/`isdir`/`islink`/`lexists` read st_mode through
    `posix._stat_field`, which reports `-errno` instead of raising, so they
    return False for every error exactly like CPython's OSError-swallowing
    versions.
  - `expanduser` expands a bare `~` and `~/rest` from $HOME only. CPython
    falls back to the password database (`pwd.getpwuid`) when $HOME is unset
    and expands `~user` for other users; there is no `pwd` module yet, so
    both cases return the path unchanged (CPython also returns it unchanged
    when it cannot resolve the user).
  - `realpath`, `samefile`, `relpath`, `commonprefix`, `commonpath`,
    `expandvars`, `getsize`, `getmtime`, `ismount` and the
    `splitdrive`/`splitroot` drive surface are not ported yet. commonprefix in
    particular does not compile: reassigning a list element alias inside a
    loop and returning it trips the ownership verifier
    ("owned resource ... is returned with 1 additional retained ownership
    token(s)"). `altsep` is None on posix and is omitted rather than
    given a non-str value.
"""

import posix

__all__ = [
    "curdir", "pardir", "sep", "pathsep", "defpath", "extsep", "devnull",
    "normcase", "isabs", "join", "split", "splitext", "basename", "dirname",
    "exists", "lexists", "isfile", "isdir", "islink",
    "abspath", "normpath", "expanduser",
]

curdir: str = "."
pardir: str = ".."
extsep: str = "."
sep: str = "/"
pathsep: str = ":"
defpath: str = "/bin:/usr/bin"
devnull: str = "/dev/null"

# S_IFMT and the two file kinds the predicates below need. CPython reads them
# from the stat module; they are POSIX-fixed values, not per-target ones.
_S_IFMT: int = 0o170000
_S_IFDIR: int = 0o040000
_S_IFREG: int = 0o100000
_S_IFLNK: int = 0o120000


def normcase(s: str) -> str:
    """Normalize case of pathname.  Has no effect under Posix"""
    return s


def isabs(s: str) -> bool:
    """Test whether a path is absolute"""
    return s.startswith("/")


# No path component can contain a NUL byte, so it stands in for "argument not
# passed" -- `join(a, b='')` would be indistinguishable from `join(a)`, and
# CPython's join DOES treat a trailing '' as a request for a trailing
# separator.
_UNSET: str = "\x00"


def _join_one(path: str, piece: str) -> str:
    if piece.startswith("/"):
        # `piece + ""`, not `piece`: returning a borrowed parameter as owned
        # from more than one branch of a function needs a dominating retain the
        # ownership verifier does not see.
        return piece + ""
    if path == "" or path.endswith("/"):
        return path + piece
    return path + "/" + piece


def join(a: str, b: str = _UNSET, c: str = _UNSET, d: str = _UNSET) -> str:
    """Join two or more pathname components, inserting '/' as needed.

    If any component is an absolute path, all previous path components
    will be discarded.  An empty last part will result in a path that
    ends with a separator.
    """
    # `a + ""` so the accumulator is owned on every path, including the
    # one-argument call that never reassigns it.
    path = a + ""
    if b != _UNSET:
        path = _join_one(path, b)
    if c != _UNSET:
        path = _join_one(path, c)
    if d != _UNSET:
        path = _join_one(path, d)
    return path


def split(p: str) -> tuple[str, str]:
    """Split a pathname.

    Returns tuple "(head, tail)" where "tail" is everything after the final
    slash.  Either part may be empty.
    """
    i = p.rfind("/") + 1
    head = p[:i]
    tail = p[i:]
    if head != "" and head != "/" * len(head):
        head = head.rstrip("/")
    return head, tail


def splitext(p: str) -> tuple[str, str]:
    """Split the extension from a pathname.

    Extension is everything from the last dot to the end, ignoring
    leading dots.  Returns "(root, ext)"; ext may be empty.
    """
    sep_index = p.rfind("/")
    dot_index = p.rfind(".")
    if dot_index <= sep_index:
        return p, ""
    # Skip all leading dots of the final component ('.cshrc' has no
    # extension, and neither does '..').
    start = sep_index + 1
    while start < dot_index:
        if p[start] != ".":
            return p[:dot_index], p[dot_index:]
        start = start + 1
    return p, ""


def basename(p: str) -> str:
    """Returns the final component of a pathname"""
    return p[p.rfind("/") + 1:]


def dirname(p: str) -> str:
    """Returns the directory component of a pathname"""
    i = p.rfind("/") + 1
    head = p[:i]
    if head != "" and head != "/" * len(head):
        head = head.rstrip("/")
    return head


def _mode(path: str, follow: bool) -> int:
    if follow:
        return posix._stat_field(path, 1, 0)
    return posix._stat_field(path, 0, 0)


def exists(path: str) -> bool:
    """Test whether a path exists.  Returns False for broken symbolic links"""
    return _mode(path, True) >= 0


def lexists(path: str) -> bool:
    """Test whether a path exists.  Returns True for broken symbolic links"""
    return _mode(path, False) >= 0


def isfile(path: str) -> bool:
    """Test whether a path is a regular file"""
    mode = _mode(path, True)
    if mode < 0:
        return False
    return (mode & _S_IFMT) == _S_IFREG


def isdir(s: str) -> bool:
    """Return true if the pathname refers to an existing directory."""
    mode = _mode(s, True)
    if mode < 0:
        return False
    return (mode & _S_IFMT) == _S_IFDIR


def islink(path: str) -> bool:
    """Test whether a path is a symbolic link"""
    mode = _mode(path, False)
    if mode < 0:
        return False
    return (mode & _S_IFMT) == _S_IFLNK


def normpath(path: str) -> str:
    """Normalize path, eliminating double slashes, etc."""
    if path == "":
        return "."
    # The leading-slash count is carried as the prefix STRING, and the
    # component stack as the accumulated result string. CPython keeps a list
    # and pops it for each '..'; none of that shape compiles yet -- `del` /
    # `pop()` inside a loop, and an int accumulator mutated across branches of
    # the loop body, each make the ownership verifier reject the function.
    # POSIX gives one or two leading slashes meaning and collapses three or
    # more to one.
    prefix = ""
    if path.startswith("///"):
        prefix = "/"
    elif path.startswith("//"):
        prefix = "//"
    elif path.startswith("/"):
        prefix = "/"
    out = ""
    comps = path.split("/")
    for comp in comps:
        if comp == "" or comp == ".":
            continue
        if comp == "..":
            if out == "":
                # Nothing to cancel: a relative path keeps the '..', an
                # absolute one lets the root absorb it ('/..' is '/').
                if prefix == "":
                    out = ".."
                continue
            if out[out.rfind("/") + 1:] == "..":
                out = out + "/.."
            elif out.rfind("/") < 0:
                out = ""
            else:
                out = out[:out.rfind("/")]
            continue
        if out == "":
            out = comp
        else:
            out = out + "/" + comp
    out = prefix + out
    if out == "":
        # One exit after the split: an early return would leave the component
        # list unreleased.
        out = "."
    return out


def abspath(path: str) -> str:
    """Return an absolute path."""
    if path.startswith("/"):
        return normpath(path)
    return normpath(join(posix.getcwd(), path))


def expanduser(path: str) -> str:
    """Expand ~ and ~user constructions.

    If user or $HOME is unknown, do nothing.
    """
    # One exit through `out`: returning the borrowed `path` parameter from
    # several branches needs a dominating retain the verifier does not see, so
    # the unexpanded answer is an owned copy.
    out = path + ""
    if path == "~" or path.startswith("~/"):
        if posix._has_env("HOME"):
            userhome = posix._getenv("HOME").rstrip("/")
            if userhome == "":
                userhome = "/"
            out = userhome + path[1:]
    return out
