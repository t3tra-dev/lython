# `raise exc` where `exc`'s static type is a UNION of exception classes and
# WHICH member is a runtime fact. Refused, with a message that says what to do
# instead; it used to reach the manifest lookup with a contract that names no
# class -- "runtime manifest has no .raise primitive", with an EMPTY contract
# in it.
#
# The spelling whose active member is STATIC works now
# (`annotated: "ValueError | KeyError" = KeyError(m); raise annotated`): the
# wrap still names the object it wrapped, so it is the plain named raise.
#
# ⛔ THREE REPAIRS, ALL MEASURED, ALL WRONG.
#
# 1. A BRANCH PER MEMBER. Compiles, runs, and the exception then ESCAPES the
#    `try` that covers it. The raise has to stay in the block the try's anchor
#    guards -- `anchorTrueEdgeGuardedCall` pairs an anchor with the marker at
#    the head of its false successor -- and an arm is not that block.
#
# 2. A PER-LANE SELECT of the active member, which is what the anchor forces.
#    Its lanes ALIAS the union's, and a raise TRANSFERS what it is handed, so
#    the ownership walk can no longer tell whether the frame still owes a
#    release. Every guess at that is wrong somewhere:
#
#      retain the select result ... leaks when the raise ESCAPES the function
#                                   (341 B over two trips of
#                                   `def fail(k): err = ...; raise err`)
#      never retain ............... crashes when a handler in the SAME frame
#                                   resumes (`Ly_IncRef observed non-positive
#                                   refcount`)
#      retain iff a local `try` ... leaks as soon as one exception is raised
#                                   TWICE inside one (355 B over a nested loop)
#
# 3. TEACH THE OWNERSHIP WALK, which is where the answer lives. Two attempts:
#    crediting the merge borrow through the select, and pushing the guarded
#    raise into `unfoldRetainBefore`. The second gets the retain placed and the
#    release then lands on the far side of a call that never returns --
#    unreachable, and still counted by a verifier that reads the CFG. Skipping
#    an after-use release behind a raise-like call removes THAT one and another
#    takes its place from a different insertion path.
#
# ⛔ AND NOT IN THE EMITTER EITHER, which this note proposed until it was
# tried. Narrowing with an isinstance chain and raising inside each branch --
# written OUT BY HAND, so no synthesis could be blamed --
#
#     for exc in [ValueError("v"), KeyError("k")]:
#         try:
#             if isinstance(exc, ValueError):
#                 raise exc
#             else:
#                 raise exc
#         except Exception as e: ...
#
# crashes exactly as the select does. A narrowing is a VIEW: the arm's raise
# still transfers lanes the union owns, and the loop's release of the element
# still runs. The same loop over a HOMOGENEOUS list is clean, which is what
# says the union wrapper is the whole difference.
#
# ⭐ SO THE ANSWER IS IN THE OWNERSHIP WALK after all: a raise consuming a
# value some OTHER group (here the union, elsewhere the frame's binding) also
# holds. That is the same question wb_raise_a_named_exception.py's open half
# asks, and the two want one answer -- which is the opposite of what the
# paragraph this replaced concluded, and the reason it is written down: the
# emitter route looks right from the IR and is wrong on the machine.
for exc in [ValueError("v"), KeyError("k")]:
    try:
        raise exc
    except Exception as e:
        print(type(e).__name__, e)
