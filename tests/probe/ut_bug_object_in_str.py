# WHAT: `x in s` with `x: object` and `s: str`: the emitter accepts it, the
#   lowering stops with "cannot adapt builtins.object to runtime input 2 of
#   builtins.str.__contains__". CPython raises TypeError for a non-str x.
# KIND: バグ (bug) -- loud, but only in the lowering
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   cannot adapt builtins.object to runtime input 2 of builtins.str.__contains__ [values: 'memref<5xi64>', expected 'memref<2xi64>']
# CPython 3.14 expects: True
def has(x: object, s: str) -> bool:
    return x in s


print(has("a", "abc"))
