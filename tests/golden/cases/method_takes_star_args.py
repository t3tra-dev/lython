# What this pins: `*args` on a method.
#
#     class Registry:
#         def many(self, *items: str) -> int:
#             return len(items)
#     Registry().many("p", "q")
#     # too many positional arguments for inlined class method
#
# The free-function spelling of the same body always worked: a real function
# binds its vararg parameter to the tuple the call packed, and the inlined method
# path had no such step -- it walked the declared positionals and refused what was
# left. It packs the remainder now and binds it to the vararg name, which is what
# the callee would have received.
#
# The empty case needed a second piece. `R().tag("p")` binds an empty tuple, and
# ITERATING that reported "list iteration evidence match/value count mismatch" --
# an internal sentence for a loop that simply does not run. An evidence sequence
# with no elements now iterates zero times; `valid` was already false there, so the
# element the op still has to produce is a dead placeholder nothing reads.
#
# Why this needs to run rather than assert on a diagnostic: what the fix decides is
# WHICH arguments land in the tuple and in what order, and an off-by-one in the
# split compiles. Every section below prints the contents, and the zero-, one- and
# three-argument calls are all present because the boundary between "declared
# positional" and "packed" is where a mistake would sit.
#
# ⛔ `**kwargs` on a method is still "unexpected keyword argument 'b' for inlined
# class method": collecting the unmatched keywords needs a dict built from values
# that are already emitted, which is a different mechanism than packing a tuple.
#
# ⛔ `self.xs = list(xs)` with NO field annotation leaves the field typed
# `builtins.object` for readers outside the class, so `len(r.xs)` is refused there
# while the body's own `len(xs)` is fine. The class-field pre-pass types the field
# without the call site, and a vararg has no type until one exists. Annotating the
# field (`self.xs: list[int] = list(xs)`) is the working spelling and is below.
#
# Every expected line is python3.14's.


class Registry:
    def many(self, *items: str) -> int:
        return len(items)

    def joined(self, *items: str) -> str:
        out = ""
        for i in items:
            out += i
        return out

    def tag(self, prefix: str, *rest: int, sep: str = "-") -> str:
        out = prefix
        for n in rest:
            out += sep + str(n)
        return out

    @classmethod
    def counted(cls, *xs: int) -> int:
        return len(xs)

    @staticmethod
    def summed(*xs: int) -> int:
        total = 0
        for x in xs:
            total += x
        return total


r = Registry()

# --- zero, one and many ----------------------------------------------------
print(r.many(), r.many("a"), r.many("a", "b", "c"))
print(r.joined(), r.joined("x"), r.joined("x", "y", "z"))

# --- a declared positional in front, and a keyword-only behind -------------
print(r.tag("p"))
print(r.tag("p", 1, 2))
print(r.tag("p", 1, sep="+"))
print(r.tag("p", sep="+"))

# --- classmethod and staticmethod -----------------------------------------
print(Registry.counted(), Registry.counted(1, 2))
print(Registry.summed(), Registry.summed(1, 2, 3))


# --- a constructor's vararg, and the annotation the field needs -----------
class Bag:
    def __init__(self, *xs: int) -> None:
        self.xs: list[int] = list(xs)
        self.n = len(xs)


b = Bag(1, 2, 3)
print(b.xs, b.n, len(b.xs))
empty = Bag()
print(empty.xs, empty.n)


# --- THE CONTROL: the free function, which was always right ---------------
def free_many(*items: str) -> int:
    return len(items)


def free_sum(*xs: int) -> int:
    total = 0
    for x in xs:
        total += x
    return total


print(free_many(), free_many("a", "b"), free_sum(), free_sum(4, 5))
