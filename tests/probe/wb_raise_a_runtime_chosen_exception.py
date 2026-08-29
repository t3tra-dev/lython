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
# ⭐ THE REPAIR IS IN THE EMITTER. Narrow the union with the `isinstance` chain
# the emitter already builds and emit one `py.raise` per member INSIDE the
# try, so each arm is a plain named raise and the EH wiring comes out right by
# construction -- which is also why 1 fails: the lowering is too late to build
# that structure. The same emitter narrowing would answer the open half of
# tests/probe/wb_raise_a_named_exception.py.
for exc in [ValueError("v"), KeyError("k")]:
    try:
        raise exc
    except Exception as e:
        print(type(e).__name__, e)
