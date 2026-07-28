# Landed 2026-07-28, red on bcfbbf9 (which had just repaired the SEQUENCE literal
# and left the dict literal deciding the same move on the same insufficient fact).
# Each block is one axis, and the axes are the ones that were MEASURED over the 25
# enumerated shapes, not the ones guessed from the sequence side.
#
# ---------------------------------------------------------------------------
# Why this needs execution: a dict literal hands the element SOURCE's token to the
# dict when the literal is "the only user" of that source. That is a use-SET fact
# standing in for an execution-FREQUENCY one. With the literal in a loop the
# source is defined OUTSIDE of, the single use runs once per inner iteration and
# the one token is handed over that many times. Every group balances, so the
# affine verifier reports nothing and the program compiles clean under --release;
# the damage only exists once the releases RUN. On bcfbbf9 the shapes below
# aborted with `Ly_DecRef observed non-positive refcount` (exit 134) or SIGSEGV,
# OR printed a wrong value with exit 0, varying between runs of ONE binary
# (measured: `.....` / `XXXXX` / `WXXWX` for the same program).
#
# Why the values matter and not just the exit code: the rc=0 face is a SILENT
# wrong answer (the accumulator came out 0), so an exit-code assertion misses
# half of it. Two blocks below were 5/5 SILENT on bcfbbf9 and never aborted at
# all.
#
# Why the immortal small-int cache is load-bearing. LyLong_FromI64 returns an
# immortal global for exactly {0, 1, 2} and a heap allocation otherwise, so an
# over-release is absorbed while the loop variable stays in that set. Measured:
# `range(0, 3)` clean and `range(3, 6)` aborting at the SAME trip count, and
# negatives failing despite small magnitude. The blocks therefore cross the
# boundary in both directions instead of only raising n.
#
# ⚠️ ONLY THE VALUE SIDE APPEARS BELOW, and that is a reachability fact rather
# than an omission: a dict literal reaches the payload path at all only when
# EVERY key is a string constant, so `{i: v}` cannot get there -- one non-static
# key sends the whole literal down the setitem_box probe path, which never asks
# the move question. `k_*` blocks vary the key CONSTANT to pin that the entry
# count and key reads are unaffected.
#
# ⛔ MEASURED WITH tests/leak_gate.py, AND IT LEAKS -- so this case may NOT be
# added to the `leak` stage (which holds only cases measured at net zero).
#
#     leak_gate.py: subject 18 roots / 541760 B, baseline 1 / 540672 B
#                   NET 17 roots / 1088 B   -> rc=1
#
# Recorded here rather than left to the next reader to discover, because a stdout
# golden cannot see a leak and this family had an unbounded one running under
# `ctest` 491/491 green the whole time it was being repaired.
#
# That leak is a SEPARATE, PRE-EXISTING defect and not a cost of the repair,
# established three ways:
#   - the leaking blocks are the ones where the move still HAPPENS (the controls
#     at the end), never the declined-move blocks the repair added;
#   - the same blocks leak byte-identically on a pre-fix binary;
#   - it scales one 64 B object per LITERAL EXECUTION and is therefore unbounded
#     (`for i in range(3, 13): c = {"a": i}` -> 10 roots / 640 B, pre-fix).
# The already-landed sequence twin leaks the same way (9 roots / 576 B, identical
# before and after), so it is container-literal-wide rather than dict-specific:
# the literal's slot retains are not matched when the container dies.
#
# The one block whose leak the repair MADE VISIBLE is the dedup block, and that
# is the cover coming off rather than a new hole -- it previously over-released
# the same object, which is why it read as leak-free while printing 0.
#
# Guard-rail, not a feature test: every value below is CPython 3.14's.

# Crossing the cache boundary by raising the trip count: i reaches 3.
total = 0
for i in range(4):
    for j in range(4):
        d = {"a": i, "b": j}
        total += d["a"] + d["b"]
print(total)

# The minimal shape: one entry, heap values only.
seen = 0
for i in range(3, 5):
    for j in range(2):
        e = {"a": i}
        seen += e["a"]
print(seen)

# Negative outer values are heap ints too, so magnitude is not the axis. This is
# the block that alternated between a silent 0 and an abort within one 5-rep run.
neg = 0
for i in range(-3, -1):
    for j in range(2):
        n = {"a": i}
        neg += n["a"]
print(neg)

# THE EVIDENCE-DEMOTION AXIS, and the only block in this file that the frequency
# query alone does not fix. The read is bound to a name that OUTLIVES the dict, so
# the container's compile-time contents evidence -- which resolves `r["a"]` to the
# stored SSA value directly, with no reference of its own -- becomes a lie the
# moment the source keeps its claim. Measured with the demotion ablated: `WWXWX`,
# i.e. a refusal on bcfbbf9 turned into a SILENT WRONG ANSWER, the one direction
# this family may never move in.
v = 0
for i in range(3, 4):
    for j in range(2):
        r = {"a": i}
        v = r["a"]
print(v)

# Three levels: the literal is two backedges away from the source, so the
# frequency mismatch compounds rather than being off by one.
deep = 0
for a in range(3, 5):
    for b in range(2):
        for c in range(2):
            g = {"a": a}
            deep += g["a"]
print(deep)

# The dict itself outlives the loop, so the last iteration's entry is read after
# every earlier one has been torn down.
keep = {"a": 0}
for i in range(3, 6):
    for j in range(2):
        keep = {"a": i}
print(keep["a"])

# THE DEDUP AXIS, which is dict-specific and NOT a copy of the sequence defect.
# One source filling two ENTRIES held one token and had it taken twice, because
# the dict path had no equivalent of the sequence path's per-source dedup set.
# `(j, j)` repeats within one slot list; `{"a": x, "b": x}` repeats across
# entries, so the sequence repair did not cover it. No loop is needed -- this
# block was 5/5 SILENT WRONG (printed 0) on bcfbbf9.
sx = 0
for i in range(1):
    x = "q" + "rs"
    dup = {"a": x, "b": x}
    sx += len(dup["a"])
print(sx)

# The same dedup shape with a heap int source, which took the abort face instead
# of the silent one (measured `XX...` -- intermittent, so reps=1 misses it).
ix = 0
for i in range(1):
    y = i + 345
    dupi = {"a": y, "b": y}
    ix += len(dupi)
print(ix)

# Dedup and frequency in one shape: the same loop variable in two entries of a
# literal that outruns it.
both = 0
for i in range(3, 6):
    for j in range(2):
        db = {"a": i, "b": i}
        both += db["a"] + db["b"]
print(both)

# Key side: only the key CONSTANT varies, to pin that declining the value move
# does not disturb the entry count or the key reads.
klen = 0
for i in range(3, 6):
    for j in range(2):
        kd = {"kk": i}
        for k in kd:
            klen += len(k)
print(klen)

# Control: no loop at all, and the value is a temporary this literal is the only
# user of. The move is correct here and must STAY -- declining it everywhere is
# what makes the container hold a reference nothing releases.
print({"a": "x" + "y"}["a"])

# Control: single loop, heap values, literal built in the SAME block as the
# source, so the literal runs exactly once per production.
one = 0
for i in range(3, 6):
    c = {"a": i}
    one += c["a"]
print(one)

# Control: only the INNER variable enters the dict, so there is no cross-loop
# borrow even though the values are heap ints.
inner = 0
for i in range(2):
    for j in range(3, 6):
        jd = {"a": j}
        inner += jd["a"]
print(inner)

# Control: a source that OUTLIVES the literal (read again afterwards) inside a
# loop. This is the no-move path, and the sequence side's slot retain in this
# shape is what stopped the affine walk from reaching a fixpoint.
s = "abc"
slen = 0
for k in range(3):
    t = {"a": s}
    slen = len(t)
print(len(s), slen)

# Control: the value source is produced INSIDE the inner loop from the outer
# variable, so it is a genuine temporary and the move must stay even though a
# loop variable is involved.
mul = 0
for i in range(3, 6):
    for j in range(2):
        m = {"a": i * 2}
        mul += m["a"]
print(mul)
