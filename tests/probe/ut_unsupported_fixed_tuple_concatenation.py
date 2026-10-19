# WHAT: `(i,) + pair` concatenates a tuple[int] and a tuple[str, str]; the
#   result is typed with `object` members instead of tuple[int, str, str],
#   so `t[0] + 1` is refused (and `fmt % t` says "static type does not
#   provide '__format__'", naming no type).
# KIND: 未対応 (unsupported)
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# FIXED: 2026-10-20 -- two tuples of known arity concatenate member by member.
# CLASSIFICATION @ lang-gaps e3a74bc1 + the fix: 1 正しい
# CPython 3.14 expects: 4 X 3
def pair(a: str, b: str) -> tuple[str, str]:
    return a, b


i = 3
t = (i,) + pair("x", "y")
print(t[0] + 1, t[1].upper(), len(t))
