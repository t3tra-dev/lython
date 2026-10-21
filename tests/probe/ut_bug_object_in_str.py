# WHAT: `x in s` with `x: object` and `s: str`: the emitter accepts it, the
#   lowering stops with "cannot adapt builtins.object to runtime input 2 of
#   builtins.str.__contains__". CPython raises TypeError for a non-str x.
# KIND: バグ (bug) -- loud, but only in the lowering
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# FIXED: 2026-10-21 -- decided at run time by a frameless helper: a str
#   needle is searched for, anything else raises CPython's TypeError.
# CLASSIFICATION @ lang-gaps 5bdaeeb4 + the fix: 1 正しい
# CPython 3.14 expects: True
def has(x: object, s: str) -> bool:
    return x in s


print(has("a", "abc"))
