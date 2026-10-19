# WHAT: `"\n".join(("case", doc))` where `doc: str | None` has been narrowed
#   by `is not None` (or by truthiness): the ownership verifier stops it,
#   "owned resource ... reaches function exit without release".
#   Concatenating (`"case\n" + doc`) or returning the tuple compiles.
#   unittest's getDescription joins a test's str and its docstring line this
#   way.
# KIND: バグ (bug) -- loud, from the ownership verifier
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# FIXED: 2026-10-20 -- a union's unwrapped member is a view of the union, not a
#   temporary a tuple display may take the reference of
#   (cases/a_narrowed_optional_in_a_tuple_display).
# CLASSIFICATION @ lang-gaps 326fa275 + the fix: 1 正しい
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
