# OPEN, and NEW (2026-08-15): a `type[X]` whose class is SUBCLASSED.
#
#     callable ABI type is '!py.type<!py.contract<"Base">>', whose class is
#     subclassed in this program, so which class it names is not decided by
#     its type
#
# This is the stated boundary of the type-object representation
# (tests/probe/wb_type_object_field.py, fixed the same day): a `type[X]` has an
# EMPTY physical shape because its type decides the class, and a subclass makes
# that false. Refused rather than carried, because carrying it means either an
# i64 class id -- which would then need a dynamic constructor dispatch this
# compiler does not have -- or silently constructing the base.
#
# MEASURED (./build/bin/lyc):
#
#   def make(t: type[Base]) -> Base: return t(3)
#   make(Base), make(Derived) ........ refused    CPython 3, 30   <- this file
#   the same with no subclass of Base . runs      (wb_type_object_field)
#   t: type[Base] = Base; t(3) ....... runs, and rebinding to Derived runs too
#
# The LOCAL spelling works and is the clue: each binding re-types the local, so
# `t(3)` after `t = Derived` is statically `!py.type<Derived>`. A parameter has
# one type for all callers, and that is the whole difference.
#
# ⭐ THE MECHANISM IS AN EXACTNESS BIT. The argument specializer built the same
# day handles the analogous numeric case (`f(3)` against `def f(x: float)`
# gets its own body), and it would handle this one too -- `make(Base)` and
# `make(Derived)` are two ground signatures -- except that the specialized
# parameter for the first is spelled `!py.type<Base>`, which is still the
# subclassed type. `py::TypeType` cannot say "exactly Base"; with that bit (or
# a marker on a specialized parameter recording that only exact classes reach
# it) the specializer covers both calls and this refusal goes away.
#
# ⛔ Why NOT relax the ABI check to trust the declared type: `make(Derived)`
# would construct a Base and print 3 where CPython prints 30. A refusal is the
# right outcome for a representation that cannot tell the two apart.
#
# differential: skip refused; the point is the refusal


class Base:
    def __init__(self, n: int) -> None:
        self.n: int = n


class Derived(Base):
    def __init__(self, n: int) -> None:
        self.n = n * 10


def make(t: type[Base]) -> Base:
    return t(3)


print(make(Base).n, make(Derived).n)
