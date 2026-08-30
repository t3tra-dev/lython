# FIXED 2026-08-30 for the read-after-the-handler shape; what is left is
# below.
#
# WAS: `raise <a name>` failed in every spelling, in three different sentences.
# The first two were closed 2026-08-29 (the anchor's true edge is the unwind
# path out of the raise, so a release there frees what the callee owns; and a
# borrowed value CONSUMED needs the retain a borrowed value RETURNED always
# had). The third was reading the name AFTER the handler:
#
#     held = ValueError("m")
#     try:
#         raise held
#     except ValueError as e: ...
#     print(str(held))       # <- the raise gave the reference away
#
# ⭐ THREE REPAIRS, and each one only exposed the next.
#
#  1. A release stood BEHIND the raise, on the edge out of the raise block.
#     `collectEdgeDeaths` reached that block because it is live-OUT -- the
#     handler edge does read the name later -- and concluded the dead edge was
#     a death. A call in the block had already given the token away, so no edge
#     out of it is one.
#
#  2. The frame needs its reference BACK, which is what CPython's raise does by
#     incrementing. A retain before the raise, and only when the group is
#     LIVE-IN at the handler the anchor branches to: without that test it fires
#     for every raise whose group has a release scheduled anywhere, and
#     `an_exception_named_by_a_union` leaked 128 B.
#
#  3. The verifier then had to CREDIT it, which the earlier note predicted.
#     Both models of the unwind take the token state from BEFORE the retain:
#     the exceptional edge built at `LyEH_TryCallSiteMarker` applies the
#     consumes between the marker and the guarded call but not the retains, and
#     the anchor's true edge applies the guarded call's transfer and nothing
#     else. Both now apply the plain retains standing in front of the guarded
#     call -- the same argument the consumes were already applied by.
#
# ⭐ THE TOOL: `LYTHON_TRACE_RELEASE_SITE_LINES=1` stamps every release with
# the line of the placement that wrote it (`ly.debug.release_site`). Three
# placements write releases that read identically in the IR, and telling them
# apart by reading the code cost two rounds.
#
# NEWLY VISIBLE, and a different group: a local assigned INSIDE the handler and
# read after it, in a function that ALSO reads the raised name after the
# handler, is refused for the SLOT's own double release
# ("builtin.unrealized_conversion_cast ... released or transferred more than
# once", path ^bb2>^bb3>^bb5>^bb11, exceptional=1). Measured to be independent
# of the retain: with the retain suppressed the same program is refused for the
# exception group instead, so the slot defect was masked rather than caused.
# Either half alone compiles -- the handler-assigned local without the later
# read of the raised name, and the later read without the handler-assigned
# local.
#
#     def f() -> str:
#         problem = KeyError("k")
#         try:
#             raise problem
#         except KeyError as caught:
#             first = "x"
#         return first + str(problem)
#
# The dynamic union-typed raise has a probe of its own
# (wb_raise_a_runtime_chosen_exception.py) and is deliberately NOT in this
# file: the emitter refuses it before the verifier runs, so it would mask the
# shape above.
#
# The fixed shapes are goldens:
# cases/an_exception_raised_through_a_name.py and
# cases/an_exception_outlives_the_handler_that_caught_it.py, both registered in
# the leak gate.
def slot_and_read(message: str) -> str:
    problem = KeyError(message)
    try:
        raise problem
    except KeyError as caught:
        first = "x"
    return first + str(problem)


print(slot_and_read("k"))
