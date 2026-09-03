# A generator's int frame lane is a BOX plus an unboxed word, and the resume
# entry forwards only the WORD into the clone body (its block arguments are the
# unboxed prim-i64 shape). `try_unbox.i64` now rebuilds the word from the box
# when the word is missing, which is what an int read out of a container needs
# -- but a value WIDER than the window has no word to rebuild, and the suspend
# guard raises:
#
#     ValueError: int too large to convert to a native 64-bit integer
#
# MEASURED (2026-09-03, RelWithDebInfo, today's tree). Everything that fits the
# window is correct now, including the four shapes that raised before it:
#
#   total = total + x over a list ......... [1, 3, 6]     correct
#   last = x over a tuple ................. correct
#   seen = xs[i] in a while ............... correct
#   total = other where other = x ......... correct
#   the same loop with 10**30 in the list .. raises (this file)
#   10**30 as the generator's ARGUMENT ..... raises the same way, from the
#                                            creation site's unbox
#
# ⛔ AND ONE SHAPE WHERE THE MESSAGE IS STILL ABOUT A SMALL INT (measured
# 2026-09-03). An accumulator whose addend comes through a DISPATCHER whose
# arms return different representations raises it for the int 1:
#
#     class Node:        def value(self) -> int: return 0
#     class Leaf(Node):  def value(self) -> int: return self.v   # a FIELD
#     def gen(ns):
#         total = 0
#         for n in ns:
#             total = total + n.value()
#             yield total
#     [v for v in gen([Leaf(1), Leaf(2)])]
#
# Each ingredient alone is correct: the same accumulation in a plain function,
# the same generator with a Leaf that returns a literal, `yield n.value()` with
# no accumulator, and the same field read with no subclass. What the pair adds
# is a value whose word is invalid on one arm -- a boxed int FIELD has none --
# merging into a lane that is all the clone has. The box is dropped at that
# merge, so there is nothing left to rebuild the word from, which is why the
# repair for the container case does not reach it.
#
# and the same sum in a PLAIN function is correct for 10**30 too, so the limit
# is the generator frame's and not the arithmetic's.
#
# ⭐ THE WORD IS THE FRAME'S ONLY INT CHANNEL ACROSS A SUSPENSION. The box
# travels beside it and is dropped at the resume entry, so the fix that made
# the fitting values right cannot make this one right: it rebuilds a word, and
# there is no word. What the shape needs is for the resume entry to forward the
# BOX into the body -- which is the clone's representation, not one lane's, and
# every generator shares it.
#
# ⛔ The MESSAGE is now honest, which it was not before: this int really is
# wider than a native 64-bit integer. The same sentence used to come out for
# the int 1.
def gen(xs: "list[int]"):
    total = 0
    for x in xs:
        total = total + x
        yield total


print([v for v in gen([10**30, 1])])


def echo(k: int):
    yield k


def from_first(xs: "list[int]") -> "list[int]":
    return [v for v in echo(xs[0])]


print(from_first([10**30]))
