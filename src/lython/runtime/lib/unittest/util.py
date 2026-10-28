"""Various utility functions.

Deviations from CPython 3.14 (Lib/unittest/util.py): `_count_diff_all_purpose`
returns plain (actual, expected, value) tuples rather than `_Mismatch` named
tuples, and `_count_diff_hashable` is not provided (case.py counts with the
all-purpose walk alone).
"""

from os.path import commonprefix

__unittest = True

_MAX_LENGTH = 80
_PLACEHOLDER_LEN = 12
_MIN_BEGIN_LEN = 5
_MIN_END_LEN = 5
_MIN_COMMON_LEN = 5
_MIN_DIFF_LEN = _MAX_LENGTH - \
               (_MIN_BEGIN_LEN + _PLACEHOLDER_LEN + _MIN_COMMON_LEN +
                _PLACEHOLDER_LEN + _MIN_END_LEN)
assert _MIN_DIFF_LEN >= 0

def _shorten(s: str, prefixlen: int, suffixlen: int) -> str:
    skip = len(s) - prefixlen - suffixlen
    if skip > _PLACEHOLDER_LEN:
        s = '%s[%d chars]%s' % (s[:prefixlen], skip, s[len(s) - suffixlen:])
    return s

def _common_shorten_repr(*args: object) -> tuple[str, ...]:
    reprs = tuple(map(safe_repr, args))
    maxlen = max(map(len, reprs))
    if maxlen <= _MAX_LENGTH:
        return reprs

    prefix = commonprefix(reprs)
    prefixlen = len(prefix)

    common_len = _MAX_LENGTH - \
                 (maxlen - prefixlen + _MIN_BEGIN_LEN + _PLACEHOLDER_LEN)
    if common_len > _MIN_COMMON_LEN:
        assert _MIN_BEGIN_LEN + _PLACEHOLDER_LEN + _MIN_COMMON_LEN + \
               (maxlen - prefixlen) < _MAX_LENGTH
        prefix = _shorten(prefix, _MIN_BEGIN_LEN, common_len)
        return tuple(prefix + s[prefixlen:] for s in reprs)

    prefix = _shorten(prefix, _MIN_BEGIN_LEN, _MIN_COMMON_LEN)
    return tuple(prefix + _shorten(s[prefixlen:], _MIN_DIFF_LEN, _MIN_END_LEN)
                 for s in reprs)

def safe_repr(obj: object, short: bool = False) -> str:
    try:
        result = repr(obj)
    except Exception:
        result = object.__repr__(obj)
    if not short or len(result) < _MAX_LENGTH:
        return result
    return result[:_MAX_LENGTH] + ' [truncated]...'

def strclass(cls: type) -> str:
    return "%s.%s" % (cls.__module__, cls.__qualname__)

def _count_diff_all_purpose[T, U](actual: list[T], expected: list[U]) -> list[tuple[int, int, T | U]]:
    'Returns list of (cnt_act, cnt_exp, elem) triples where the counts differ'
    # elements need not be hashable
    s, t = list(actual), list(expected)
    m, n = len(s), len(t)
    # CPython overwrites a counted element with a NULL sentinel; a list here
    # holds one element type, so the counted positions are marked beside it.
    s_counted = [False] * m
    t_counted = [False] * n
    result: list[tuple[int, int, T | U]] = []
    for i, elem in enumerate(s):
        if s_counted[i]:
            continue
        cnt_s = cnt_t = 0
        for j in range(i, m):
            if not s_counted[j] and s[j] == elem:
                cnt_s += 1
                s_counted[j] = True
        for j, other_elem in enumerate(t):
            if not t_counted[j] and other_elem == elem:
                cnt_t += 1
                t_counted[j] = True
        if cnt_s != cnt_t:
            result.append((cnt_s, cnt_t, elem))

    for i, elem in enumerate(t):
        if t_counted[i]:
            continue
        cnt_t = 0
        for j in range(i, n):
            if not t_counted[j] and t[j] == elem:
                cnt_t += 1
                t_counted[j] = True
        result.append((0, cnt_t, elem))
    return result
