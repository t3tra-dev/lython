# A nested generator that CAPTURES an enclosing local is refused with a message
# about the compiler's own bookkeeping:
#
#     generator resume clone entry was not seeded as a primitive-i64 callable
#
# MEASURED (2026-09-02, RelWithDebInfo, today's tree):
#
#   a nested generator with NO capture ............... correct
#   the same value passed as a PARAMETER ............. correct
#   a module-level generator reading a global ........ correct
#   a nested generator capturing an int local ........ the message above
#   a nested generator capturing a str local ......... "a generator cannot
#                                                       carry a value of
#                                                       contract 'builtins.str'
#                                                       across a suspension"
#
# ⭐ A CLOSURE CAPTURE IS AN ENTRY ARGUMENT THE CALLABLE TYPE DOES NOT LIST.
# `callableLogicalInputTypes` appends the closure types after the positionals,
# but the generator state machine builds its argument lanes from
# `callable.getPositionalTypes()` alone -- so the resume clone is given a
# function type shorter than its own entry block, and the seeding check reports
# that mismatch about itself. `isPrimitiveI64CallableEligible` already declines
# a function with closure types for the same reason; this path had not been
# told.
#
# ⛔ THREE SITES, MEASURED ONE AT A TIME by building the lanes from the entry
# block instead:
#   1. the lane/`argumentCount` construction   -> fixed by using entry args
#   2. `callableLogicalInputTypes` on the CLONE re-appending the closures
#      (the clone inherits `closure_types`)    -> fixed by removing the attr
#   3. `appendGeneratorArgumentOperands` at the creation site: the capture's
#      bundle carries no `primitiveI64` evidence, so an int capture cannot be
#      handed to the int lane ("state-machine generator frame sources must
#      carry primitive int evidence")
# The third is where it stands. A capture arrives as a closure VALUE and the
# resume ABI wants the creation-site evidence a parameter has, so the repair is
# at the creation site, not in the lane layout.
def outer(n: int):
    def gen():
        for i in range(3):
            yield i + n

    return list(gen())


print(outer(10))
