# ⛔ THE MOST IMPORTANT GUARD IN THIS GROUP. Read before touching either of
# CollectionPayload.cpp's source-move predicate or AffineOwnership.cpp's retain
# accounting.
#
# On main 4699488 this program is REFUSED (rc=1). A refusal is a safe failure:
# nothing runs, nothing is mis-executed.
#
# Combining the two candidate repairs -- (1) declining the sequence-literal source
# move when the literal can execute more often than the source is produced, and
# (2) skipping slot-absorption retains in the affine walk -- turns this refusal
# into exit 0 printing the WRONG VALUE. Measured 4/4 runs.
#
# That is the one direction a change here may never take. It is strictly worse
# than the double-free the repair was aimed at, because a double-free at least
# announces itself sometimes (the guard fires on roughly half the runs) whereas
# this is silent, deterministic, and survives --release.
#
# Why the pair does it when neither does alone: (1) stops the source's token from
# moving into the container, and (2) removes the retain that would have kept the
# walk honest about the read-back -- so the release placement is free to sink past
# the read with nothing left to object.
#
# Corollary for anyone measuring this family: fold rc != 0 AND value-mismatch into
# FAIL when DETECTING, but never when measuring which DIRECTION a change moved --
# `R` and `X` and `W` are three different outcomes and collapsing them hides
# exactly this transition. Use reps >= 5; single-rep grids miss the intermittent
# faces entirely.
v = 0
for i in range(3, 4):
    for j in range(2):
        ys = [i]
        v = ys[0]
print(v)  # CPython 3.14: 3
