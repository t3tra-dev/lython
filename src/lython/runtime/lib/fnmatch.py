"""Filename matching with shell patterns.

fnmatch(FILENAME, PATTERN) matches according to the local convention.
fnmatchcase(FILENAME, PATTERN) always takes case in account.

The functions operate by translating the pattern into a regular
expression.  They cache the compiled regular expressions for speed.

The function translate(PATTERN) returns a regular expression
corresponding to PATTERN.  (It does not compile it.)

This is Lython's port of CPython 3.14's Lib/fnmatch.py, restricted to the
well-typed statically compilable surface.

Deviations from CPython:
- str names and patterns only (no bytes).
- the compiled patterns are cached in a dict of Pattern objects bounded
  like CPython's lru_cache (32768), not an lru_cache of bound `match`
  methods; re.escape is not cached.
- os.path is posixpath here, so normcase is the identity.
"""

import os
import re
from collections.abc import Iterable

__all__ = ["filter", "filterfalse", "fnmatch", "fnmatchcase", "translate"]


def fnmatch(name: str, pat: str) -> bool:
    """Test whether FILENAME matches PATTERN.

    Patterns are Unix shell style:

    *       matches everything
    ?       matches any single character
    [seq]   matches any character in seq
    [!seq]  matches any char not in seq

    An initial period in FILENAME is not special.
    Both FILENAME and PATTERN are first case-normalized
    if the operating system requires it.
    If you don't want this, use fnmatchcase(FILENAME, PATTERN).
    """
    name = os.path.normcase(name)
    pat = os.path.normcase(pat)
    return fnmatchcase(name, pat)


_cache: dict[str, re.Pattern] = {}
_MAXCACHE = 32768


def _compile_pattern(pat: str) -> re.Pattern:
    found = _cache.get(pat)
    if found is not None:
        return found
    compiled = re.compile(translate(pat))
    if len(_cache) >= _MAXCACHE:
        _cache.clear()
    _cache[pat] = compiled
    return compiled


def filter(names: Iterable[str], pat: str) -> list[str]:
    """Construct a list from those elements of the iterable NAMES that match PAT."""
    result: list[str] = []
    pat = os.path.normcase(pat)
    pattern = _compile_pattern(pat)
    for name in names:
        if pattern.match(os.path.normcase(name)) is not None:
            result.append(name)
    return result


def filterfalse(names: Iterable[str], pat: str) -> list[str]:
    """Construct a list from those elements of the iterable NAMES that do not match PAT."""
    pat = os.path.normcase(pat)
    pattern = _compile_pattern(pat)
    result: list[str] = []
    for name in names:
        if pattern.match(os.path.normcase(name)) is None:
            result.append(name)
    return result


def fnmatchcase(name: str, pat: str) -> bool:
    """Test whether FILENAME matches PATTERN, including case.

    This is a version of fnmatch() which doesn't case-normalize
    its arguments.
    """
    return _compile_pattern(pat).match(name) is not None


def translate(pat: str) -> str:
    """Translate a shell PATTERN to a regular expression.

    There is no way to quote meta-characters.
    """

    parts, star_indices = _translate(pat, '*', '.')
    return _join_translated_parts(parts, star_indices)


_re_setops = re.compile(r'([&~|])')


def _translate(pat: str, star: str,
               question_mark: str) -> tuple[list[str], list[int]]:
    res: list[str] = []
    star_indices: list[int] = []

    i = 0
    n = len(pat)
    while i < n:
        c = pat[i]
        i = i+1
        if c == '*':
            # store the position of the wildcard
            star_indices.append(len(res))
            res.append(star)
            # compress consecutive `*` into one
            while i < n and pat[i] == '*':
                i += 1
        elif c == '?':
            res.append(question_mark)
        elif c == '[':
            j = i
            if j < n and pat[j] == '!':
                j = j+1
            if j < n and pat[j] == ']':
                j = j+1
            while j < n and pat[j] != ']':
                j = j+1
            if j >= n:
                res.append('\\[')
            else:
                stuff = pat[i:j]
                if '-' not in stuff:
                    stuff = stuff.replace('\\', r'\\')
                else:
                    chunks: list[str] = []
                    k = i+2 if pat[i] == '!' else i+1
                    while True:
                        k = pat.find('-', k, j)
                        if k < 0:
                            break
                        chunks.append(pat[i:k])
                        i = k+1
                        k = k+3
                    chunk = pat[i:j]
                    if chunk:
                        chunks.append(chunk)
                    else:
                        chunks[-1] += '-'
                    # Remove empty ranges -- invalid in RE.
                    for k in range(len(chunks)-1, 0, -1):
                        if chunks[k-1][-1] > chunks[k][0]:
                            chunks[k-1] = chunks[k-1][:-1] + chunks[k][1:]
                            del chunks[k]
                    # Escape backslashes and hyphens for set difference (--).
                    # Hyphens that create ranges shouldn't be escaped.
                    stuff = '-'.join([s.replace('\\', r'\\').replace('-', r'\-')
                                      for s in chunks])
                i = j+1
                if not stuff:
                    # Empty range: never match.
                    res.append('(?!)')
                elif stuff == '!':
                    # Negated empty range: match any character.
                    res.append('.')
                else:
                    # Escape set operations (&&, ~~ and ||).
                    stuff = _re_setops.sub(r'\\\1', stuff)
                    if stuff[0] == '!':
                        stuff = '^' + stuff[1:]
                    elif stuff[0] in ('^', '['):
                        stuff = '\\' + stuff
                    res.append(f'[{stuff}]')
        else:
            res.append(re.escape(c))
    return res, star_indices


def _join_translated_parts(parts: list[str], star_indices: list[int]) -> str:
    if not star_indices:
        return fr'(?s:{"".join(parts)})\z'
    j = star_indices[0]
    buffer = parts[:j]  # fixed pieces at the start
    i = j + 1
    for j in star_indices[1:]:
        # Now deal with STAR fixed STAR fixed ...
        # For an interior `STAR fixed` pairing, we want to do a minimal
        # .*? match followed by `fixed`, with no possibility of backtracking.
        # Atomic groups ("(?>...)") allow us to spell that directly.
        # Note: people rely on the undocumented ability to join multiple
        # translate() results together via "|" to build large regexps matching
        # "one of many" shell patterns.
        buffer.append('(?>.*?')
        buffer.extend(parts[i:j])
        buffer.append(')')
        i = j + 1
    buffer.append('.*')
    buffer.extend(parts[i:])
    res = ''.join(buffer)
    return fr'(?s:{res})\z'
