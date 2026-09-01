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
# ⛔ WHAT REMAINS is the yield INSIDE the loop, which is a different refusal:
# the state machine declines because the protocol-typed token is live across a
# yield and a frame lane is keyed on a runtime contract, which a compile-time
# token does not have. The position is an i64 and the frame already carries i64
# lanes, so the cell can be saved at the suspend and written back at the
# resume, keyed as `builtins.int`. A rewrite that materialized the keys instead
# would have to carry CPython's per-step size guard ("dictionary changed size
# during iteration"), which a copy does not have -- trading a refusal for a
# silent divergence is the trade this compiler does not make.
def keys_of(d: "dict[str, int]"):
    for k in d:
        yield k


print(list(keys_of({"a": 1, "b": 2})))
