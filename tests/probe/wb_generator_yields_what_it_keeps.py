# A generator that YIELDS an object and keeps binding it aborts at runtime:
# `Ly_DecRef observed non-positive refcount` on a later trip. The value goes to
# the consumer AND stays in the suspended frame, which is TWO references, and
# the suspend lanes carry one.
#
# ⭐ THE LANE MACHINERY ALREADY STATES THE RULE. `appendGeneratorLaneReturnOperands`
# (Ops/GeneratorStateMachine.cpp) retains when `forceRetain` says "the SAME
# value carried by two lanes, and two lanes are two references however the
# first one was obtained". Its caller (ABI/Returns.cpp) computes that with
#
#     bool duplicate = !laneCarriedValues.insert(operand).second;
#
# on the LOGICAL operand -- and one entity reaches the yield lane and the frame
# lane through two py-dialect values, so the set never sees a duplicate and
# neither lane retains. Traced by printing each lane's index, contract and
# resolved value at that line.
#
# ⛔ COMPARING THE PHYSICAL ENTITY IS NOT ENOUGH EITHER, measured. Keying on
# `underlyingObjectValue(bundle->physicalValues().front())` fixes the plain
# `yield x; x = []` loop; the CONDITIONAL yield then still fails, because the
# two lanes arrive as two ARGUMENTS (index 1 and index 9) of the same suspend
# block. Walking each block argument back to the value every predecessor
# forwards fixes that one too -- and the shape below, whose only difference is
# that the condition names a PARAMETER, still aborts. That last step is where
# the reduction stopped being trustworthy: three bisects in a row put the
# boundary somewhere the next program contradicted.
#
# ⛔ SO NOTHING IS SHIPPED. Both keys above are correct as far as they go and
# neither characterises the boundary, and a refcount change whose boundary is
# not characterised is the one kind this compiler must not carry. What is
# recorded is the mechanism and the two keys, so the next attempt starts from
# the lane comparison rather than from the crash.
#
# THE NEIGHBOURS, all measured:
#   yield without a rebind ............................ clean
#   rebind without the append (nothing mutates) ....... clean
#   the same accumulate/rebind loop NOT in a generator  clean
#   `for` over a list parameter instead of range ...... aborts
#   `list(chunks(...))` or a comprehension ............ aborts
#   two generators of this shape in one module ........ aborts
def chunks(values: list[int], size: int):
    current: list[int] = []
    for v in values:
        current.append(v)
        if len(current) == size:
            yield current
            current = []


for chunk in chunks([1, 2, 3, 4], 2):
    print(chunk)
