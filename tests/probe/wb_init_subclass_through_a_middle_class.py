# A class that DECLARES `__init_subclass__` does not get its own parent's hook
# run when it is defined:
#
#     class Base:      __init_subclass__ prints "hook"
#     class Mid(Base): __init_subclass__ prints "mid hook"
#     class Leaf(Mid): pass
#
#     CPython: hook | mid hook          lyc: mid hook
#
# The hook fires for every class that does NOT declare one (fixed 2026-08-31,
# golden cases/a_base_hook_runs_when_the_subclass_is_defined.py); this is the
# remaining half.
#
# ⭐ WHY THE SAME SPELLING DOES NOT WORK. The call is emitted as
# `Sub.__init_subclass__()`, and looking an INHERITED classmethod up through
# the subclass is what binds `cls` to the subclass -- exactly the argument
# CPython passes. When the class declares its own, that spelling finds the
# class's own method instead, which CPython never runs on itself.
#
# ⛔ AND THE OBVIOUS ALTERNATIVE IS A WRONG ANSWER, not a missing one:
# `Base.__init_subclass__()` binds `cls` to Base, so a hook that registers
# `cls` would register the wrong class silently. Reaching the parent's body
# with `cls` bound to the new class is what `super()` does inside a method, and
# there is no expression for it at the class statement's position.
#
# THE SHAPE OF THE REPAIR: an unbound spelling for a classmethod -- the
# `Base.m(receiver)` form that already exists for INSTANCE methods, with the
# type object as the explicit first argument.
class Base:
    @classmethod
    def __init_subclass__(cls) -> None:
        print("hook")


class Mid(Base):
    @classmethod
    def __init_subclass__(cls) -> None:
        print("mid hook")


class Leaf(Mid):
    pass


print("done")
