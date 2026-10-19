# WHAT: `"\n".join(("case", doc))` where `doc: str | None` has been narrowed
#   by `is not None` (or by truthiness): the ownership verifier stops it,
#   "owned resource ... reaches function exit without release".
#   Concatenating (`"case\n" + doc`) or returning the tuple compiles.
#   unittest's getDescription joins a test's str and its docstring line this
#   way.
# KIND: バグ (bug) -- loud, from the ownership verifier
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   owned resource from @_ZN8__main__5shortE result 1 reaches function exit without release, transfer, or owned return
# CPython 3.14 expects: case case / described
def short(flag: bool) -> str | None:
    if flag:
        return "described"
    return None


def describe(flag: bool) -> str:
    doc = short(flag)
    if doc is not None:
        return "\n".join(("case", doc))
    return "case"


print(describe(False), describe(True))
