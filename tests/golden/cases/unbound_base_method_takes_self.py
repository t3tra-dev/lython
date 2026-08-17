# What this pins: calling a base class's method unbound, with `self`.
#
#     class Child(Base):
#         def __init__(self, n: int) -> None:
#             Base.__init__(self, "c")
#     # argument 'self' of '__init__' is declared Base and this call gives it
#     # Child
#
# `Child` IS a `Base`, so the refusal was simply wrong. Two neighbours show what
# it really was: `Base.__init__(c, "z")` at module scope works, and so does
# `Base.greet(c, "x")` -- by then `py.class @Child` is in the module carrying its
# `mro_names`, and the assignability walk can find `Base` among them. Inside
# Child's OWN method the class op does not exist yet, the walk sees no bases at
# all, and a subtype was reported as an unrelated type.
#
# The emitter has known the hierarchy all along: `classMros` is populated before
# any body is emitted, and it is what `resolveMroMethod` already reads. The check
# consults it when the module cannot answer.
#
# Why this needs to run rather than assert on a diagnostic: what the call does is
# initialise the base's half of the instance, so the failure mode of a wrong fix
# is a field that never got written. Every section below reads back a field the
# base's method set, and one of them sets it twice to show which write wins.
#
# ⛔ `super().__init__(...)` always worked and is the spelling to prefer; the
# unbound form exists in real code (cooperative `__init__` chains, mixins that
# name their base explicitly) and refusing it refused the program.
#
# ⛔ Still refused, and correctly: a base method that assigns a field the BASE
# does not declare (`class Base: def start(self, n): self.name = n` with `name`
# declared only on the child) is "class Base has no field 'name'". Instance
# attributes are fixed slots in the static layout, so the field has to be
# declared where the method that writes it lives.
#
# Every expected line is python3.14's.


class Base:
    def __init__(self, name: str) -> None:
        self.name = name

    def tag(self) -> str:
        return "<" + self.name + ">"

    def rename(self, name: str) -> str:
        self.name = name
        return self.name


class Child(Base):
    def __init__(self, n: int) -> None:
        Base.__init__(self, "c")
        self.n = n

    def reset(self) -> None:
        Base.__init__(self, "reset")


class GrandChild(Child):
    def __init__(self) -> None:
        Child.__init__(self, 9)


# --- the unbound base __init__ from a subclass __init__ --------------------
c = Child(3)
print(c.name, c.n, c.tag())

# --- and from another method of the subclass -------------------------------
c.reset()
print(c.name, c.n, c.tag())

# --- two levels deep -------------------------------------------------------
g = GrandChild()
print(g.name, g.n, g.tag())

# --- a non-dunder base method, called unbound with self --------------------
print(Base.rename(c, "again"), c.name, c.tag())

# --- the spellings that always worked, as the control ---------------------
print(Base.__init__(c, "direct"), c.name)
print(Base.tag(c), Base.rename(c, "z"), c.name)


class Super(Base):
    def __init__(self) -> None:
        super().__init__("super")


print(Super().name, Super().tag())
