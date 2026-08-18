# What this pins: a method's default argument is evaluated ONCE.
#
#     class Bag:
#         def add(self, into: list[int] = []) -> int:
#             into.append(1)
#             return len(into)
#     b = Bag()
#     print(b.add(), b.add())      # printed 1 1; CPython prints 1 2
#
#     def make() -> int:
#         print("eval"); return 1
#     class Bag:
#         def get(self, n: int = make()) -> int: ...
#     Bag().get(); Bag().get()     # printed eval twice; CPython prints it once
#
# The FREE-function spelling of both was already right. CPython evaluates a def's
# defaults when the def statement executes -- for a method, when the class body
# runs -- and parks them on the function; Lython has the same mechanism (a
# module-lifetime R6 cell) and it was gated on the def being a direct child of the
# module body, so a method fell to the per-call PROVIDER instead: a fresh list on
# every call, and a side-effecting default firing again each time.
#
# Two halves. The cell is registered under the CLASS statement, because the module
# walk flushes pending cells at the statement it skipped and for a method that is
# the ClassDef -- the note at that call site already said method defaults would
# flow through it, and nothing had ever registered one. And the INLINED call reads
# the cell instead of re-emitting the expression, which is what an inlined method
# does with everything else.
#
# Why this needs to run rather than assert on a diagnostic: the whole defect is
# WHICH object the second call sees. A per-call default prints a plausible number
# and never raises, so the sections below call twice and print both answers, and
# the side-effect counter says how many times the expression ran.
#
# ⛔ The control function here is called `plain`, not `free`: a module-level
# `def free(...)` collides with the C library symbol and the whole program stops
# with "redefinition of reserved function 'free' of different type is prohibited",
# 34 times over. That is its own defect, recorded in the probe.
#
# ⛔ A @classmethod's default is still per-call (`Reg.make()` twice prints 1 1
# where CPython prints 1 2): its body is emitted from a node that is not the one
# in the class body, so the cell is never registered for it. @staticmethod is
# fixed, and is a section below.
#
# Every expected line is python3.14's.


def make_list() -> list[int]:
    return [7]


def counted() -> int:
    print("eval")
    return 1


class Bag:
    def add(self, into: list[int] = []) -> int:
        into.append(1)
        return len(into)

    def get(self, n: int = counted()) -> int:
        return n

    def grown(self, xs: list[int] = make_list()) -> int:
        xs.append(1)
        return len(xs)

    def tag(self, name: str = "n", extra: list[int] = []) -> str:
        extra.append(len(extra))
        return name + str(len(extra))

    def kw(self, *, into: list[int] = []) -> int:
        into.append(1)
        return len(into)


# --- the mutable default is ONE object across calls ------------------------
b = Bag()
print(b.add(), b.add(), b.add())

# --- the side-effecting default runs once ----------------------------------
print(b.get(), b.get())

# --- a default built by a call, mutated through ---------------------------
print(b.grown(), b.grown())

# --- an explicit argument does not touch the shared default ---------------
print(b.tag(), b.tag("z"), b.tag(extra=[9]), b.tag())

# --- keyword-only defaults take the same cell -----------------------------
print(b.kw(), b.kw(), b.kw(into=[]), b.kw())

# --- two instances share the default, as CPython does ---------------------
class Seeded:
    def __init__(self, seed: list[int] = []) -> None:
        self.items = seed


s1 = Seeded()
s2 = Seeded()
s1.items.append(1)
print(s1.items, s2.items, len(s2.items))


# --- @staticmethod ---------------------------------------------------------
class Reg:
    @staticmethod
    def note(into: list[str] = []) -> int:
        into.append("x")
        return len(into)


print(Reg.note(), Reg.note())


# --- THE CONTROL: the free function, which was always right ---------------
def plain(into: list[int] = []) -> int:
    into.append(1)
    return len(into)


print(plain(), plain())
