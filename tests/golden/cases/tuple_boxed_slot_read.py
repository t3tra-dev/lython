# Element reads that go through the ERASED lane, i.e. through the
# `builtins.object` `from_slot` primitive rather than through a statically
# resolved slot.
#
# Why this needs to execute: the defect it pins is a LEAK, and a leak is
# invisible to exit code and to stdout -- the shipped compiler got both right
# while never freeing the boxed element. Only `leaks --atExit` on an AOT binary
# can see it, and that gate (tests/leak_gate.py, the `leak` label) takes a golden
# case as its input. The stdout below is therefore a bonus, not the assertion.
#
# What separates these three reads from a literal tuple's: their payload comes
# from a runtime primitive that allocates a fresh items array (concat, repeat,
# `BaseException.args`), so the element cannot be resolved at compile time and
# `LyObject_FromSlot` runs. A literal tuple's `t[0]` never calls it at all --
# measured 0 FromSlot calls in the whole module -- which is why a literal-tuple
# case would not cover this and does not.
#
# The leak was one object per READ, unbounded: 10 reads leaked 10 roots and 50
# leaked 50. `LyObject_FromSlot` initialises the box's refcount to 1 and declares
# `ly.ownership.owned_results = [0]`; the lowering retained it as well, so the
# counter sat at 2 against the one release refcount-insertion emits.

c = (1, 2) + (3, 4, 5)
print(c[0], c[1], c[2], c[3], c[4])

r = (1, 2) * 3
print(r[0], r[3], r[5])

# The loop is the part that made it unbounded rather than a fixed cost, so it is
# what a regression would show up in first: ten reads, so a per-read leak comes
# back as ten roots rather than as one.
#
# The element is PRINTED rather than added to: an erased read has contract
# `builtins.object`, and `total + x[0]` is refused at the frontend ("operand 1
# type builtins.object does not match selected Callable evidence builtins.int").
# That refusal is the project working as intended, and it is why the erased lane
# is exercised by reads and not by arithmetic.
i = 0
while i < 5:
    x = (i, i + 1) + (i + 2,)
    print(x[0], x[2])
    i = i + 1

# `.args` is a tuple the exception machinery builds, read back after the frame
# that raised is gone.
try:
    raise ValueError("boom", 7)
except ValueError as e:
    print(len(e.args))
    print(e.args[0])
    print(e.args[1])
