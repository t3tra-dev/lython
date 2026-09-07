# OPEN, and PRE-EXISTING: storing into a union-typed field inside an `if`, a
# `for` or a `while` and reading the field after the merge produces INVALID IR
# -- "operand #0 does not dominate this use", then "Failed to run lowering
# pipeline". An MLIR verifier message is the worst answer this compiler gives:
# it names no line of the program and no thing the author did.
#
# MEASURED 2026-09-08, on this tree and on the pre-session binary alike, so the
# union-field guard repair did not cause it -- it only lets more programs reach
# it (`if isinstance(b.v, str): ...; b.v = 1` inside a method called twice).
#
#   if <cond>: c.v = 1        then read c.v ....... invalid IR
#   if <cond>: c.v = 1 else: c.v = 2  then read ... invalid IR
#   for i in range(2): c.v = i  then read ......... invalid IR
#   while ...: c.v = n  then read ................. invalid IR
#   c.v = 1 in a straight line, then read ......... correct
#   the same four shapes with an `int` field ...... correct
#   the same four shapes with an `int | None` field correct
#
# ⭐ WHY THE UNION AND NOT THE OTHERS: an `int` field is a WORD in the instance
# body and an `int | None` field is a BOX slot, so both are memory and a read
# after a merge loads what the merge left there. A union of two real members is
# stored INLINE -- the tag and every member's lanes are part of the object's
# own physical value group (`classFieldStoredBoxed` returns false for it, and
# `runtimeValueTypesFor` splices the lanes into the class) -- so the store
# rebinds an SSA VALUE, and a store inside a region rebinds it in a block that
# does not dominate the read.
#
# ⛔ SO THIS IS NOT A PLACEMENT BUG and no amount of care in the store's
# lowering fixes it: the field has no memory to be written. It is the same
# missing mechanism as "a Box value expands to 5 physical values and nothing
# can rebuild them from one address", which is what refuses `list[Box]` when
# Box has such a field -- one storage decision, three symptoms.
class C:
    def __init__(self) -> None:
        self.v: "int | str" = "a"


c = C()
if len("x") == 1:
    c.v = 1
x = c.v
print(isinstance(x, int))
