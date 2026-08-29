# Two closures over the SAME captured object, returned together. Refused:
#
#   ownership: this block-argument merge needs a retain on the edge and the
#   header prefix cannot be spelled at the point the retain must go (header
#   type 'memref<9xi64>', op result)
#
# ⛔ THE REDUCTION. Every neighbour compiles:
#   - ONE closure over the list, returned .................... works
#   - two closures, only ONE returned ........................ works
#   - two closures returned, the second reading no capture ... works
#   - two closures over an int through `nonlocal` ............ works
# What is left is: two closures naming one captured OBJECT, both returned.
# The captured type does not matter -- list, dict and str all refuse.
#
# ⛔ AND THE PREDICATE THE MESSAGE NAMES LOOKS RIGHT. The entity is
# `%0 = call @LyList_FromLength(%c0)`, and `prefixIsInitializedAtDefinition`
# (ABI/EntityHeaderPrefix.h) accepts a `func.call` root explicitly -- "an
# entity the callee finished before returning". So the header the pass is
# failing to spell is NOT that value: it is something else of the same type
# whose provenance walk does not reach the call. Find what, before touching
# the predicate; widening it is how an over-release shipped once already.
def make():
    values: list[int] = []

    def get() -> list[int]:
        return values

    def get2() -> list[int]:
        return values

    return get, get2


a, b = make()
print(a(), b())
