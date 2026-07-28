# Landed 2026-07-28 (was tests/probe/seqlit_pending_golden_full_coverage.py,
# red until the sequence-literal source-move repair). Each block below is one
# axis, cut on the axis that was MEASURED -- the immortal small-int cache --
# rather than the one first guessed, a threshold at n=4.
#
# ---------------------------------------------------------------------------
# Why this needs execution: a container literal hands the element SOURCE's token
# over to the container when the literal is "the only user" of that source. That
# is a use-SET fact used as a proxy for an execution-FREQUENCY one. When the
# literal sits in a loop the source is defined OUTSIDE of, the single use runs
# once per inner iteration and the single token is handed over that many times.
# Nothing is unbalanced per group, so the affine verifier reports nothing and the
# program compiles clean under --release; the damage is only observable once the
# releases actually RUN. On main before this case the shapes below aborted with
# `Ly_DecRef observed non-positive refcount` (exit 134) OR printed a wrong value
# with exit 0, varying between runs of the same binary.
#
# Why the values matter and not just the exit code: the failure is
# non-deterministic and its rc=0 face is a SILENT wrong answer (the accumulator
# came out 0), so an exit-code-only assertion misses half of it.
#
# Why the immortal small-int cache is load-bearing here. LyLong_FromI64 returns
# an immortal global for exactly {0, 1, 2} and a heap allocation for everything
# else, so an over-release of a loop variable is absorbed while the variable
# stays inside that set. `range(3)` therefore passes and `range(4)` does not, and
# negative values fail despite being small -- which is why the cases below cross
# the boundary in both directions rather than only raising n.
#
# Guard-rail, not a feature test: every value below is CPython 3.14's.

# Crossing the cache boundary by raising the trip count: i reaches 3.
total = 0
for i in range(4):
    for j in range(4):
        ys = [i, j]
        total += ys[0] + ys[1]
print(total)

# The minimal shape: one element, and the container is never read for its own
# sake. The accumulator is not what breaks; the element store is.
seen = 0
for i in range(3, 5):
    for j in range(2):
        zs = [i]
        seen += zs[0]
print(seen)

# Not list-specific: a tuple literal takes the same source-move decision.
tt = 0
for i in range(5, 7):
    for j in range(2):
        ts = (i, j)
        tt += ts[0] + ts[1]
print(tt)

# Negative outer values are heap ints too, so magnitude is not the axis.
neg = 0
for i in range(-3, -1):
    for j in range(2):
        ns = [i]
        neg += ns[0]
print(neg)

# Control: single loop, the same heap values, container built in the SAME block
# as the source. Here the literal runs exactly once per production, so the move
# is correct and must stay.
one = 0
for i in range(3, 6):
    cs = [i, i]
    one += cs[0] + cs[1]
print(one)

# Control: only the INNER variable enters the container, so there is no
# cross-loop borrow even though the values are heap ints.
inner = 0
for i in range(2):
    for j in range(3, 6):
        js = [j]
        inner += js[0]
print(inner)

# A source that OUTLIVES the literal (read again after it) INSIDE a loop. This is
# the no-move path, and it is the shape whose slot retain accumulates once per
# iteration in the affine walk's state key: counting it there never reaches a
# fixpoint, and main refused this program with `ownership CFG exploration
# exceeded 20000 states` rather than running it.
s = "abc"
tlen = 0
for k in range(3):
    t = (s,)
    tlen = len(t)
print(len(s), tlen)

# The same no-move shape with a heap int source.
q = 7
ulen = 0
for k in range(3):
    u = (q, q)
    ulen = len(u)
print(q, ulen)

# Three levels: the innermost literal is two backedges away from the source, so
# the frequency mismatch compounds rather than being off by one.
deep = 0
for a in range(3, 5):
    for b in range(2):
        for c in range(2):
            ds = [a]
            deep += ds[0]
print(deep)
