# WHAT: a store proves what it wrote, and that proof is spent at the READ with
# a check -- so a program that CHANGES the field in between raises where CPython
# just answers. Same deviation the guard form has had
# (wb_a_field_read_after_a_call_that_changed_it), reached from an assignment
# instead of a test, and the message says "a guard or an assignment" for that
# reason.
#
# MEASURED 2026-09-07. The two shapes that reach it, and the ones that do not:
#
#   self.f = 5; self.clear(); self.f is None ......... raises (this file)
#   a.f = 5; b.f = None; a.f is None   (b is a) ...... raises (this file)
#   c.f = 5; for ...: c.f = None; c.f is None ....... CORRECT -- the walk that
#                                                     erases a fact now looks
#                                                     INSIDE a compound
#                                                     statement, which fixed
#                                                     the guard form too
#   self.f = 5; self.note("x"); self.f + 1 .......... correct (the call does
#                                                     not change the field)
#   guard, then a call, then a read ................. correct
#
# ⛔ Erasing the fact at any CALL naming the root was measured and rejected: it
# would refuse `if self.v is not None: self.note("a"); return self.v.upper()`,
# which compiles and runs today. The deviation is the project's chosen answer
# for a proof about the past -- loud, never a wrong value -- and this is the
# same answer one statement earlier.
class Box:
    def __init__(self) -> None:
        self.f: "int | None" = None

    def clear(self) -> None:
        self.f = None

    def set_then_clear(self) -> str:
        self.f = 5
        self.clear()
        return "none" if self.f is None else "value"


print(Box().set_then_clear())
