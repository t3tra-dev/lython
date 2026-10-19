# WHAT: `(i,) + pair` concatenates a tuple[int] and a tuple[str, str]; the
#   result is typed with `object` members instead of tuple[int, str, str],
#   so `t[0] + 1` is refused (and `fmt % t` says "static type does not
#   provide '__format__'", naming no type).
# KIND: 未対応 (unsupported)
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   static type !py.contract<"builtins.object"> does not provide manifest method '__add__'
# CPython 3.14 expects: 4 X 3
def pair(a: str, b: str) -> tuple[str, str]:
    return a, b


i = 3
t = (i,) + pair("x", "y")
print(t[0] + 1, t[1].upper(), len(t))
