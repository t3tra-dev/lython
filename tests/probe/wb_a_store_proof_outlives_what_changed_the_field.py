# WHAT IS LEFT: a store proves what it wrote, and that proof is spent at the
# READ with a check -- so a program that changes the field through ANOTHER NAME
# for the same object raises where CPython just answers.
#
#   a = Box(); b = a
#   a.f = 5
#   b.f = None
#   print(a.f is None)      # CPython True; here AttributeError
#
# MEASURED 2026-09-08. The shapes that reach it and the ones that no longer do:
#
#   a.f = 5; b.f = None; a.f is None   (b is a) ...... raises (this file)
#   self.f = 5; self.clear(); self.f is None ......... CORRECT since the walk
#                                                      that erases a fact asks
#                                                      what the CALLEE assigns
#   c.f = 5; for ...: c.f = None; c.f is None ....... correct
#   self.f = 5; self.note("x"); self.f + 1 .......... correct -- the callee
#                                                      assigns nothing
#   guard, then a mutating call, then a read ........ correct
#
# ⭐ WHY THE ALIAS IS THE HARD ONE. Every fact is keyed on a dotted PATH, and
# `b.f` is a different path from `a.f` however the two names came to share an
# object. Closing it needs an alias relation between local names, which nothing
# in the emitter has -- and the deviation is loud (a raise, never a wrong
# value), which is the project's chosen answer for a proof about the past.
class Box:
    def __init__(self) -> None:
        self.f: "int | None" = None


a = Box()
b = a
a.f = 5
b.f = None
print(a.f is None)
