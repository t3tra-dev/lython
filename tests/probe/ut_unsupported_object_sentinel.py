# WHAT: `_sentinel = object()` compared with `is`: a bare object has no run-
#   time allocation of its own. unittest's subTest uses one for "no
#   message".
# KIND: 未対応 (unsupported)
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   `object()` cannot be constructed yet: a bare object has no runtime allocation of its own. Declare a class (`class Sentinel: pass`) and instantiate that instead
# CPython 3.14 expects: 1 missing
_sentinel = object()


def get(d: dict[str, int], k: str, default: object = _sentinel) -> object:
    if k in d:
        return d[k]
    if default is _sentinel:
        return "missing"
    return default


print(get({"a": 1}, "a"), get({}, "b"))
