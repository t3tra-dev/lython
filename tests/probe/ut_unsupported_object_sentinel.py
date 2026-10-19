# WHAT: `_sentinel = object()` compared with `is`: a bare object has no run-
#   time allocation of its own. unittest's subTest uses one for "no
#   message".
# KIND: 未対応 (unsupported)
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# FIXED: 2026-10-20 -- `object()` allocates an object of class `object` with an
#   empty slot, released by the box's own deallocator
#   (cases/an_object_made_by_object).
# CLASSIFICATION @ lang-gaps a1f774b0 + the fix: 1 正しい
# CPython 3.14 expects: 1 missing
_sentinel = object()


def get(d: dict[str, int], k: str, default: object = _sentinel) -> object:
    if k in d:
        return d[k]
    if default is _sentinel:
        return "missing"
    return default


print(get({"a": 1}, "a"), get({}, "b"))
