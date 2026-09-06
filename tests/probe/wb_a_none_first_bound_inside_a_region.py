# A name whose FIRST binding is `None` and whose binding sits inside a region
# cannot be read after it:
#
#     def use() -> int:
#         for _ in range(1):
#             v = None
#         return 0 if v is None else 1
#     # unresolved name 'v'
#
# MEASURED (2026-09-06, RelWithDebInfo, today's tree) by sweeping every
# annotation kind through one shape -- bind it inside a region, read it after.
# The region matters and the TYPE matters, and each isolates the other half:
#
#   v = 3 inside for / while / try / with .......... correct
#   v = None inside a plain `if` (both arms) ....... correct
#   v = None with a `v = None` BEFORE the region ... correct
#   v = None first bound inside for/while/try/with . this file
#   v = SomeClass (a type[X]) in the same place .... the same, and recorded
#                                                    with its own measured
#                                                    reason in
#                                                    inferConditionalLocalType
#
# ⭐ WHY THE `if` SPELLING IS THE DISCRIMINATOR. A plain `if` joins its arms
# through BLOCK ARGUMENTS, and a None passes one as zero physical values on
# both edges. `for`, `while`, `try` and `with` take a SLOT instead -- a
# synthesized class's box-fronted field, which is also what records whether the
# name was written so a zero-trip region can still raise.
#
# ⛔ AND LETTING None THROUGH THE SLOT RULE IS NOT THE REPAIR. Measured: it
# turns the refusal into "field 'v' of '!py.literal<None>' has no instance body
# word" out of the LOWERING, for all four regions. A None expands to no
# physical values, so the field has no word -- what the slot needs is the
# written FLAG alone, and the field layout has no shape for a zero-width one.
# That is the mechanism, and it is the same one `type[X]` waits on
# ([[lython-zero-lane-values]]).
def use() -> int:
    for _ in range(1):
        v = None
    return 0 if v is None else 1


print(use())
