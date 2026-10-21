# WHAT: `fs.append(lambda: value())` where `value() -> object` and `fs:
#   list[Callable[[], object]]`: "lambda requires a Callable annotation
#   because its type contains unresolved Unknown". The parameter type of
#   list.append is not the lambda's expected type; `lambda: 3` goes through,
#   as does a lambda bound to an annotated name first. `d[k] = lambda: ...`
#   and setdefault fail the same way.
# KIND: バグ (bug) -- loud: well-typed code refused
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# FIXED: 2026-10-21 -- a lambda argument of a builtin container's method is
#   typed by the method's parameter on that receiver. (`dict.get` /
#   `setdefault` with a lambda default take another path and are not.)
# CLASSIFICATION @ lang-gaps a30d9f09 + the fix: 1 正しい
# CPython 3.14 expects: [3]
from collections.abc import Callable


def value() -> object:
    return 3


fs: list[Callable[[], object]] = []
fs.append(lambda: value())
print([f() for f in fs])
