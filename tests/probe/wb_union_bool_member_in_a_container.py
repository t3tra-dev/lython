# A union whose active member is a bool cannot be stored in a container.
# Returning one works (golden: a_return_that_is_a_number_or_a_flag); putting
# the same value in a list does not:
#
#   builtins.bool runtime object header has invalid type 'i1'
#
# MEASURED (2026-09-02, RelWithDebInfo, today's tree):
#
#   print(classify(-1)) ............................ correct (False)
#   got = classify(-1); print(got) ................. correct
#   [classify(v) for v in (-1, 1)] ................. the message above
#   [classify(-1)] ................................. the message above
#   (classify(-1),) ................................ the message above
#   {"k": classify(-1)} ............................ the message above
#   a union of int | str in the same list .......... correct
#   [True, False] and list[bool] ................... correct
#
# ⭐ A BOOL OWNS NO LANE, which is what makes the return work: a bool member
# lowers to a bare i1 and the union result's owned-lane list has to leave it
# out (Runtime/ABI/CallableABI.cpp). A container payload asks the opposite
# question -- every element needs an object handle to store -- and a bool
# member has none to give.
#
# ⛔ This is the same wall `optionalPayloadRebuildableFromBox`
# (Runtime/Ops/AttributeOps.cpp) already names: `builtins.bool` is the one
# contract with no entity, its manifest shape is `i1`, and boxing it needs a
# heap bool to exist first. A plain `[True, False]` gets past it only because
# a bare bool element widens to `object` on the way in; a bool that arrives as
# a union MEMBER never reaches that widening, because the union's per-member
# storage is spliced inline.
def classify(n: int):
    if n < 0:
        return False
    return n * 2


print([classify(v) for v in (-1, 1)])
