# ⛔ REFUSED, BEFORE AND AFTER the 2026-07-28 dict repair, on the FIRST block --
# and that is a separate pre-existing defect, not this repair's. Kept because it
# holds the DENOMINATOR, which is the part of this investigation that was easiest
# to get wrong and hardest to recover once wrong.
#
# The runtime-key block below refuses with `ownership CFG exploration exceeded
# 20000 states (last: retained=909 parked=0 ...)`. Byte-identical message AND
# counters on a genuine pre-fix binary, so the dict source-move repair neither
# caused nor touched it -- it cannot, because the shape never reaches
# initializeDictPayload at all (that is the whole point of this file).
#
# `parked=0` is the interesting number: the `setitem_box` probe path emits its
# per-entry retains with NO aggregate parent, so the retain count climbs every
# iteration and the visited-state key never repeats. That is exactly the shape
# `chargeSlotRetainsToParent` was added to fix on the payload path, applied there
# and not here. NOT fixed as part of the dict source-move work because it is a
# different pass's accounting and lives behind a file another track holds; it is
# recorded here so the next person does not read this file's refusal as a
# regression of the source-move rule.
#
# ⚠️ So "the key side is unreachable" does NOT mean "the key side is safe". It
# means the key side fails somewhere else, earlier, and safely.
#
# The brief for the dict-side measurement said "dict keys are hashed, so the key
# side and the value side may behave differently -- MEASURE BOTH". Reading
# PackAndBindingOps.cpp first showed that half of that instruction is
# unsatisfiable, and reading it AFTER a sweep would have produced a sweep whose
# key-side rows were all clean for a reason unrelated to ownership.
#
# THE GATE. `initializeDictPayload` -- and therefore the source-move decision at
# all -- is reached only when EVERY key of the literal is a `py.str_constant`.
# `keywordNameFromValue` answers only for StrConstantOp; a single non-static key
# clears `allStaticStringKeys`, and the whole literal is then built by the
# `setitem_box` probe path (LyDict_New plus one insert per entry), which never
# asks whether a source is a temporary.
#
# So `{i: v}` with a loop-variable key CANNOT reach the defect, and the key side
# can only ever vary the constant. The 25-shape grid was cut on the value side
# for that reason, not because the key side was skipped.
#
# ⛔ IF THIS FILE EVER STOPS BEING THE GATE -- if non-constant keys start reaching
# the evidence payload path -- the dict source-move rule acquires a whole family
# of shapes nobody has measured, and the key side of that grid has to be cut
# before the change lands.
#
# Below: the runtime-key spelling (probe path, no move question -- REFUSED, see
# above) and the constant-key spelling (payload path, move question asked --
# repaired, runs), side by side in the loop nesting that makes the frequency
# mismatch reachable. Run the second block alone to see it pass.
probe = 0
for i in range(3, 6):
    for j in range(2):
        d = {i: 1}
        for k in d:
            probe += k
print(probe)  # CPython 3.14: 24

payload = 0
for i in range(3, 6):
    for j in range(2):
        e = {"k": i}
        payload += e["k"]
print(payload)  # CPython 3.14: 24
