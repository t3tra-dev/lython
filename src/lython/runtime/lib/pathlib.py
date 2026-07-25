"""Object-oriented filesystem paths.

This is Lython's port of CPython's Lib/pathlib, restricted to the well-typed
statically compilable surface.

CPython's PurePath dispatches every operation through a `_flavour` object
(posixpath or ntpath) chosen at class-creation time, and PurePath.__new__
returns a PurePosixPath or PureWindowsPath depending on os.name. That is a
runtime type switch, and Lython resolves the platform at compile time instead:
this module has ONE `Path` class over posixpath, and `PurePath`,
`PurePosixPath`, `PureWindowsPath`, `PosixPath` and `WindowsPath` are absent.
A Windows target would get an ntpath-backed Path from an ntpath port, selected
by the same `sys.platform` fold os.py uses -- not by a runtime flavour.

Deviations from CPython:
  - the PurePath hierarchy and the flavour mechanism are gone (above). `Path`
    is final and concrete.
  - `Path(...)` takes up to four str segments rather than `*args` of
    str-or-PathLike: a `*args` parameter read inside an imported module
    currently mis-executes (reported to the Wave 3 foundation track). Pass a
    Path with `str(other)`, or use `/` and `joinpath`, both of which accept a
    str segment.
  - `parts` is a `list[str]`, not a tuple: the segment count is not static, so
    a tuple has no layout. It follows CPython's content -- an absolute path's
    first element is '/'.
  - `parents` is absent (it is a lazy immutable sequence view); walk `parent`
    instead. `glob`/`rglob` accept only a pattern of the form '*', '*.ext',
    'prefix*', or a literal name -- no '**', no character classes, no
    multi-component patterns -- and are non-recursive except for `rglob`,
    which walks the whole subtree. `match`, `relative_to`, `resolve`,
    `absolute`, `owner`, `group`, `symlink_to`, `hardlink_to`, `chmod`,
    `touch`, `samefile`, `open`, `walk` and the `with_*` family beyond
    `with_name`/`with_suffix` are not ported.
  - `mkdir(parents=True)` uses os.makedirs; `unlink(missing_ok=True)` and
    `rmdir()` behave as CPython's.
  - `suffixes` is absent (it needs the repeated-splitext loop over a list),
    and `stem`/`suffix`/`name` follow CPython exactly otherwise.
  - `read_text`/`write_text` are UTF-8 only (no `encoding=`/`errors=`
    keywords), which is the only encoding the io layer implements.
"""

import os
import posixpath

__all__ = ["Path"]


def _clean(raw: str) -> str:
    """Drop redundant separators and single dots, as PurePath's parser does.

    Not normpath: '..' is a lexical component PurePath deliberately keeps,
    because collapsing it is wrong in the presence of symlinks. POSIX gives one
    or two leading slashes meaning and collapses three or more to one, so the
    root is kept verbatim.

    The accumulator is assigned from exactly ONE expression per iteration and
    is never overwritten with a literal afterwards: `out = "."` after a loop
    that concatenated into `out` drops the owned string without releasing it,
    and the ownership verifier rejects the function.
    """
    out = ""
    if raw.startswith("///"):
        out = "/"
    elif raw.startswith("//"):
        out = "//"
    elif raw.startswith("/"):
        out = "/"
    pieces = raw.split("/")
    for piece in pieces:
        if piece == "" or piece == ".":
            continue
        sep = "/"
        if out == "" or out.endswith("/"):
            sep = ""
        out = out + sep + piece
    if out == "":
        return "."
    return out


def _fnmatch(name: str, pattern: str) -> bool:
    """The single-component pattern match glob()/rglob() accept.

    Only the three shapes the module docstring lists, matched directly rather
    than through a compiled regex: fnmatch's translate() plus `re` would pull
    in far more than the patterns pathlib callers actually write.
    """
    if pattern == "*":
        return True
    if pattern.startswith("*"):
        return name.endswith(pattern[1:])
    if pattern.endswith("*"):
        return name.startswith(pattern[:-1])
    return name == pattern


class Path:
    """A concrete path on the local filesystem.

    Immutable in use: every operation that would change the path returns a new
    Path. The str form is normalized only as far as CPython's PurePath does --
    redundant separators and single dots are dropped, '..' is kept.
    """

    def __init__(self, a: str = ".", b: str = "", c: str = "",
                 d: str = "") -> None:
        self._raw: str = _clean(posixpath.join(a, b, c, d))

    def __str__(self) -> str:
        return self._raw + ""

    def __repr__(self) -> str:
        return "PosixPath(" + repr(self._raw) + ")"

    def __eq__(self, other: "Path") -> bool:
        return self._raw == other._raw

    def __ne__(self, other: "Path") -> bool:
        return self._raw != other._raw

    def __hash__(self) -> int:
        return hash(self._raw)

    def __truediv__(self, segment: str) -> "Path":
        return Path(posixpath.join(self._raw, segment))

    def joinpath(self, a: str, b: str = "", c: str = "") -> "Path":
        """This path joined with the given segments."""
        return Path(posixpath.join(self._raw, a, b, c))

    # --- pure lexical surface ----------------------------------------------

    @property
    def name(self) -> str:
        """The final path component, if any."""
        return posixpath.basename(self._raw)

    @property
    def suffix(self) -> str:
        """The final component's last suffix, including the leading dot."""
        return posixpath.splitext(posixpath.basename(self._raw))[1]

    @property
    def stem(self) -> str:
        """The final component, minus its last suffix."""
        return posixpath.splitext(posixpath.basename(self._raw))[0]

    @property
    def parent(self) -> "Path":
        """The logical parent of the path.

        A path with no parent (the root, or a bare name) is its own parent, as
        CPython's is.
        """
        # ONE construction site: a property (or method) returning a user class
        # instance from two different `return` statements segfaults at runtime
        # (reported to the Wave 3 foundation track).
        head = posixpath.dirname(self._raw)
        if head == "":
            head = "."
        return Path(head)

    @property
    def parts(self) -> list[str]:
        """The path's components.

        A list, not CPython's tuple: the count is not static. An absolute
        path's first element is '/', as CPython's is.
        """
        out: list[str] = []
        rest = self._raw
        if rest.startswith("/"):
            out.append("/")
        pieces = rest.split("/")
        for piece in pieces:
            if piece != "" and piece != ".":
                out.append(piece)
        return out

    def is_absolute(self) -> bool:
        """True if the path is absolute."""
        return self._raw.startswith("/")

    def with_name(self, name: str) -> "Path":
        """This path with the final component replaced."""
        if posixpath.basename(self._raw) == "":
            raise ValueError("Path has an empty name")
        head = posixpath.dirname(self._raw)
        target = name
        if head != "":
            target = posixpath.join(head, name)
        return Path(target)

    def with_suffix(self, suffix: str) -> "Path":
        """This path with the final component's suffix replaced or removed."""
        if suffix != "" and not suffix.startswith("."):
            raise ValueError("Invalid suffix " + repr(suffix))
        root = posixpath.splitext(self._raw)[0]
        return Path(root + suffix)

    # --- filesystem queries -------------------------------------------------

    def exists(self) -> bool:
        """Whether this path exists (following symlinks)."""
        return posixpath.exists(self._raw)

    def is_file(self) -> bool:
        """Whether this path is a regular file."""
        return posixpath.isfile(self._raw)

    def is_dir(self) -> bool:
        """Whether this path is a directory."""
        return posixpath.isdir(self._raw)

    def is_symlink(self) -> bool:
        """Whether this path is a symbolic link."""
        return posixpath.islink(self._raw)

    def stat(self) -> os.stat_result:
        """os.stat() on this path."""
        return os.stat(self._raw)

    # --- reading and writing ------------------------------------------------

    def read_text(self) -> str:
        """The file's whole contents, decoded as UTF-8."""
        handle = open(self._raw, "r")
        data = handle.read()
        handle.close()
        return data

    def write_text(self, data: str) -> int:
        """Write a str to the file, encoded as UTF-8; returns the count."""
        handle = open(self._raw, "w")
        written = handle.write(data)
        handle.close()
        return written

    def read_bytes(self) -> bytes:
        """The file's whole contents as bytes."""
        handle = open(self._raw, "r")
        data = handle.read()
        handle.close()
        return data.encode()

    def write_bytes(self, data: bytes) -> int:
        """Write bytes to the file; returns the count."""
        handle = open(self._raw, "w")
        written = handle.write(data.decode())
        handle.close()
        return written

    # --- directories --------------------------------------------------------

    def iterdir(self) -> list["Path"]:
        """The directory's entries, as Paths.

        A list rather than CPython's generator: a generator defined in an
        imported module does not compile yet. Sorted, unlike CPython's
        readdir order, so callers are deterministic without asking.
        """
        out: list[Path] = []
        names = os.listdir(self._raw)
        names.sort()
        for entry in names:
            out.append(Path(posixpath.join(self._raw, entry)))
        return out

    def glob(self, pattern: str) -> list["Path"]:
        """Entries of this directory matching a single-component pattern.

        The pattern grammar is '*', '*.ext', 'prefix*', or a literal name --
        see the module docstring.
        """
        out: list[Path] = []
        names = os.listdir(self._raw)
        names.sort()
        for entry in names:
            if _fnmatch(entry, pattern):
                out.append(Path(posixpath.join(self._raw, entry)))
        return out

    def rglob(self, pattern: str) -> list["Path"]:
        """glob() over this whole subtree, this directory first."""
        out: list[Path] = []
        triples = os.walk(self._raw)
        for triple in triples:
            here = triple[0]
            entries = triple[1] + triple[2]
            entries.sort()
            for entry in entries:
                if _fnmatch(entry, pattern):
                    out.append(Path(posixpath.join(here, entry)))
        return out

    def mkdir(self, mode: int = 0o777, parents: bool = False,
              exist_ok: bool = False) -> None:
        """Create this directory."""
        if parents:
            os.makedirs(self._raw, mode, exist_ok)
            return
        if exist_ok and posixpath.isdir(self._raw):
            return
        os.mkdir(self._raw, mode)

    def rmdir(self) -> None:
        """Remove this directory; it must be empty."""
        os.rmdir(self._raw)

    def unlink(self, missing_ok: bool = False) -> None:
        """Remove this file or symbolic link."""
        if missing_ok and not posixpath.lexists(self._raw):
            return
        os.unlink(self._raw)

    def rename(self, target: str) -> "Path":
        """Rename this path to `target`; returns the new Path."""
        os.rename(self._raw, target)
        return Path(target)
