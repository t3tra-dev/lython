# What: `MISSING = None` at module scope, read from inside a function. The
# module-constant channel re-emits a name bound ONCE to a literal at each use,
# which is what makes `N = 5` readable from a body -- and None was excluded
# from it with no stated reason, so `V = None` was "unresolved name 'V'" one
# line from a `V = 3` that worked. CPython does not distinguish the two.
#
# Running it is the evidence: the identity test and the comparison have to see
# the same None the module bound, and the sentinel has to survive being passed
# and returned rather than resolving to nothing.
MISSING = None
LIMIT = 5


def lookup(key: str) -> str:
    table = {"a": "x"}
    if key in table:
        return table[key]
    return "missing" if MISSING is None else "?"


def pass_it() -> str:
    held = MISSING
    return "still none" if held is None else "changed"


def beside_a_number() -> int:
    return LIMIT if MISSING is None else 0


print(lookup("a"), lookup("b"))
print(pass_it())
print(beside_a_number())
print(MISSING is None, MISSING == None)
