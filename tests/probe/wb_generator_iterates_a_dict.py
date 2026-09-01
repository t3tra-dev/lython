# A generator that iterates a DICT or a SET is refused. A list, tuple, str,
# range, bytes and enumerate all work inside one, and every one of these loops
# works in a plain function.
#
# TWO messages, and they are the same cause seen from two places:
#
#   for k in d: yield k          -> the tier-below refusal, whose decline
#                                   reason names "a value of type
#                                   !py.protocol<"Iterator", [str]> is live
#                                   across a yield and has no frame lane"
#
#   for k in d: ks.append(k)     -> "protocol-typed receiver ... has no
#   for k2 in ks: yield k2          concrete runtime method evidence for
#                                   __next__" -- and note the dict iteration
#                                   here finishes BEFORE the first yield, so
#                                   liveness is not what refuses it.
#
# ⭐ THE ITERATOR IS COMPILE-TIME EVIDENCE PLUS AN ALLOCA CELL. `lowerIter`
# (Ops/SpecialMethodOps.cpp) iterates a runtime dict or set by POSITION through
# a hoisted cell -- there is no runtime iterator object for either -- so the
# value that crosses into the generator clone is a token whose state lives in
# memory the clone re-creates, and whose `__next__` evidence rides a bundle the
# clone does not carry. The list and tuple forms work because the EMITTER
# rewrites them into an index loop first (`emitGeneratorIndexedFor`), so no
# cell and no protocol receiver is ever made.
#
# ⛔ THAT REWRITE CANNOT REACH A DICT: it needs `__getitem__` by POSITION and a
# dict's takes a key. `.keys()`, `.values()` and `.items()` all reduce to this
# same plain iteration (EmitterIterators.cpp), so all three are refused too.
#
# ⛔ MATERIALIZING WAS NOT A WAY AROUND IT EITHER, until the second message was
# repaired: `list(d)` and `sorted(d)` inside the generator were refused as well,
# because building them iterates the dict in the same place.
#
# FIXED (the second message, 2026-09-01): the evidence iterator is a
# compile-time token whose position lives in a function-level cell, and the
# resume clone threads the loop's values through BLOCK ARGUMENTS -- where the
# bundle was rebuilt from the type, which is the bare protocol. `lowerNext`
# follows the forwarding edges back to the value that owns the token, and the
# cell is valid in every block of the function. A dict or set walked before the
# first yield now compiles (golden:
# a_generator_walks_a_dict_before_it_yields), and so does `list(d)` inside a
# generator.
#
# FIXED (the first message, 2026-09-01): with `list(d)` compiling inside a
# generator, the yield INSIDE the loop takes the same rewrite the list source
# does -- the keys through a list, whose int index rides a frame lane where the
# token's cell cannot. The per-step size guard the cell carried comes with it,
# checked at the top of every trip AND on exhaustion, which is where CPython
# raises for a one-key dict grown in its own loop (goldens:
# a_generator_yields_a_dicts_keys, errors/a_generator_that_grows_the_dict_it_
# walks).
#
# ⛔ Only inside a generator. Outside one the cell walks the live table, which
# is closer to CPython than a copy and costs no list.
#
# THIS PROBE IS KEPT for the shape, not for a live defect: both spellings above
# compile now. It is the record of what the two messages meant.
def keys_of(d: "dict[str, int]"):
    for k in d:
        yield k


print(list(keys_of({"a": 1, "b": 2})))
