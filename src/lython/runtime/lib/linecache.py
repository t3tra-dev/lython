"""Cache lines from Python source files.

Port of CPython 3.14's Lib/linecache.py, restricted to files on disk.

Deviations from CPython:
- an entry is (size, mtime, lines, fullname) with no lazy form: there are no
  module loaders to ask for source (`lazycache` and the `module_globals`
  argument are not provided), and no `sys.path` to search for a relative
  name that is not found.
- the file is read by `open` with its default encoding, UTF-8, rather than
  through `tokenize.open`, which would honour a coding cookie.
"""

import os

__all__ = ["getline", "clearcache", "checkcache"]


# The cache. Maps filenames to a tuple (size, mtime, lines, fullname).
cache: dict[str, tuple[int, float, list[str], str]] = {}


def clearcache() -> None:
    """Clear the cache entirely."""
    cache.clear()


def getline(filename: str, lineno: int) -> str:
    """Get a line for a Python source file from the cache.
    Update the cache if it doesn't contain an entry for this file already."""

    lines = getlines(filename)
    if 1 <= lineno <= len(lines):
        return lines[lineno - 1]
    return ''


def getlines(filename: str) -> list[str]:
    """Get the lines for a Python source file from the cache.
    Update the cache if it doesn't contain an entry for this file already."""

    if filename in cache:
        return cache[filename][2]
    return updatecache(filename)


def _source_unavailable(filename: str) -> bool:
    """Return True if the source code is unavailable for such file name."""
    return (
        not filename
        or (filename.startswith('<')
            and filename.endswith('>')
            and not filename.startswith('<frozen '))
    )


def checkcache(filename: str | None = None) -> None:
    """Discard cache entries that are out of date.
    (This is not checked upon each call!)"""

    filenames: list[str] = []
    if filename is None:
        filenames = list(cache.keys())
    else:
        filenames = [filename]

    for name in filenames:
        if name not in cache:
            continue
        size, mtime, lines, fullname = cache[name]
        try:
            stat = os.stat(fullname)
        except OSError:
            cache.pop(name, None)
            continue
        if size != stat.st_size or mtime != stat.st_mtime:
            cache.pop(name, None)


def updatecache(filename: str) -> list[str]:
    """Update a cache entry and return its list of lines.
    If something's wrong, print a message, discard the cache entry,
    and return an empty list."""

    cache.pop(filename, None)
    if _source_unavailable(filename):
        return []
    fullname = filename
    try:
        stat = os.stat(fullname)
    except OSError:
        return []
    try:
        with open(fullname) as fp:
            lines = fp.read().splitlines(keepends=True)
    except (OSError, UnicodeDecodeError):
        return []
    if not lines:
        lines = ['\n']
    elif not lines[-1].endswith('\n'):
        lines[-1] += '\n'
    cache[filename] = (stat.st_size, float(stat.st_mtime), lines, fullname)
    return lines
