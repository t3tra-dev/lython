# What this pins: reading a list back after `insert` shifted it.
#
#     xs = [1, 3]
#     xs.insert(1, 2)
#     print(xs[1])          # printed 3; CPython prints 2
#     print(xs[0], xs[1], xs[2])
#     # error: owned resource from @LyLong_FromI64 result 0 is released or
#     # transferred more than once on one CFG path
#
# A silent wrong answer for one read and an ownership abort for three. The list
# carries compile-time ELEMENT EVIDENCE -- the emitter knows what each slot
# holds, so `xs[1]` can answer without touching the runtime -- and an insert
# SHIFTS every slot at or past the index. `list` is handle-fronted now, so its
# mutators are void (the new items address is written through the handle), which
# means the rebind after the call hands the receiver straight back, evidence and
# all: the evidence then named the pre-insert slots. Reading two of them handed
# the same box to two releases.
#
# `__setslice__` already demoted for exactly this reason. This is the sibling
# that also changes a length, and it did not.
#
# Why this needs to run rather than assert on a diagnostic: half of the failure
# printed a plausible list and a wrong element, and no diagnostic was involved
# at all. What is being pinned is which VALUE each read produces after the
# shift, so the reads below are spelled out one index at a time as well as in
# bulk -- `repr` always answered from the payload and would have passed
# throughout.
#
# ⛔ The evidence is DROPPED, not shifted: an insert index is a runtime value,
# so which slots moved is not known where the demotion happens. Reads after an
# insert therefore cost a runtime load. `append` keeps its evidence, because it
# only ever adds past the end.
#
# Every expected line is python3.14's.

# --- one index at a time, which is where the wrong answer showed -----------
xs = [1, 3]
xs.insert(1, 2)
print(xs[0])
print(xs[1])
print(xs[2])
print(len(xs), xs)

# --- and all of them live at once, which is where the abort showed ---------
ys = [1, 3]
ys.insert(1, 2)
print(ys[0], ys[1], ys[2])

# --- strings, so the shifted element is a heap object ----------------------
ws = ["a", "b"]
ws.insert(1, "c")
print(ws[0], ws[1], ws[2], ws)

# --- inserting twice, and at the ends -------------------------------------
zs = [1, 3]
zs.insert(1, 2)
zs.insert(0, 0)
zs.insert(4, 4)
print(zs, zs[0], zs[2], zs[4], sum(zs))

# --- a runtime index, and one past the end --------------------------------
n = 1
ks = [10, 30]
ks.insert(n, 20)
print(ks[n], ks)
ks.insert(99, 40)
print(ks[3], len(ks))

# --- through a loop, where the evidence is already runtime-mode -----------
acc: list[int] = []
for i in range(4):
    acc.insert(0, i)
print(acc, acc[0], acc[3])

# --- THE CONTROL: append, which keeps its evidence ------------------------
# `append` only ever writes past the end, so no existing slot moves and the
# evidence stays valid. These reads must still answer.
bs = [1]
bs.append(2)
print(bs[0], bs[1], bs)
