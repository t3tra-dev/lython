# What this pins: `break` / `continue` inside a `try`, in a loop that carries a
# reassigned local.
#
#     total = 0
#     for s in ["1", "x", "3"]:
#         try:
#             total += int(s)
#         except ValueError:
#             continue
#     # break/continue through try/finally in a loop with carried (reassigned)
#     # locals is not implemented yet
#
# That is the canonical parse-and-skip loop, and the refusal covered every
# accumulator loop with a jump inside a try. Without the accumulator the same
# statement always compiled, which is what named the missing piece: the
# completion branches emitted after `py.try` forward the loop's carried operands,
# and in SSA they could only forward the pre-try values.
#
# They no longer read SSA. A local the try body rebinds is promoted to an R6 cell
# for the extent of the statement -- the mechanism was already there, and it was
# extended to loop-carried names earlier -- so the completion branches LOAD the
# cell and forward what the body last stored.
#
# Why this needs to run rather than assert on a diagnostic: what the branch
# forwards IS the answer. Forwarding the pre-try value compiles and prints a
# plausible smaller total, so each section below skips a middle element and the
# printed accumulator is what says which value crossed the edge. The owned cases
# (a str and a list accumulator) are the leak gate's half of the same question:
# the edge releases the replaced loop-header value and retains the forwarded one,
# and getting that wrong is invisible in the output.
#
# ⛔ `continue` inside a `finally` is not in this file: CPython 3.14 emits a
# SyntaxWarning for it that Lython does not, so the two outputs differ on a line
# that has nothing to do with the jump. The behaviour agrees; the warning is
# recorded in the probe.
#
# ⛔ Two nested shapes stay refused, both of which were refused before this fix
# too, and both with a message that names the shape rather than a resource limit:
# a `continue` out of an INNER loop whose accumulator the outer loop carries as
# well (the ownership walk explores the promoted cell across both back edges and
# gives up at 20000 states), and a nested try whose outer arm has a `finally`
# (the inner try promotes the name, so the outer one does not, and the outer
# completion branch has nothing to forward). The same nested program with `break`
# compiles, and so does one whose accumulator belongs to the inner loop alone --
# both are sections below.
#
# Every expected line is python3.14's.

# --- the parse-and-skip loop, which is the shape that was refused ----------
total = 0
for s in ["1", "x", "3"]:
    try:
        total += int(s)
    except ValueError:
        continue
print(total)


# --- continue from inside the try body itself ------------------------------
kept = 0
for v in [1, 2, 3]:
    try:
        if v == 2:
            continue
        kept += v
    except ValueError:
        pass
print(kept)


# --- break, which must carry the accumulator out of the loop ---------------
count = 0
for v in [1, 2, 3]:
    count += 1
    try:
        if v == 2:
            break
    except ValueError:
        pass
print(count)


# --- a while loop with its own counter -------------------------------------
i = 0
seen = 0
while i < 4:
    i += 1
    try:
        if i == 2:
            continue
        seen += i
    except ValueError:
        pass
print(i, seen)


# --- an OWNED accumulator: the edge has a reference to balance -------------
acc = ""
for s in ["a", "x", "c"]:
    try:
        if s == "x":
            continue
        acc = acc + s
    except ValueError:
        pass
print(acc, len(acc))

names = ["a"]
for s in ["b", "c"]:
    try:
        names = names + [s]
        if s == "b":
            continue
    except ValueError:
        pass
print(names, len(names))

best = "z"
for s in ["m", "a", "q"]:
    try:
        if s == "a":
            best = s
            break
    except ValueError:
        pass
print(best)


# --- two accumulators of different kinds, and a real exception ------------
count2 = 0
text = ""
for s in ["1", "x", "3"]:
    try:
        count2 += int(s)
        text = text + s
    except ValueError:
        continue
print(count2, text)


# --- a finally that runs on the way out -----------------------------------
log = ""
sum2 = 0
for v in [1, 2, 3]:
    try:
        if v == 2:
            continue
        sum2 += v
    finally:
        log = log + str(v)
print(sum2, log)


# --- the two nested shapes that DO compile, next to the refused ones -------
# `break` out of the inner loop with an accumulator both loops carry: the break
# edge leaves the inner loop instead of re-entering it.
grid = 0
for a in [1, 2]:
    for b in [10, 20]:
        try:
            if b == 20:
                break
            grid += a * b
        except ValueError:
            pass
print(grid)

# and a `continue` whose accumulator belongs to the inner loop alone.
outer = 0
for a in [1, 2]:
    inner = 0
    for b in [10, 20]:
        try:
            if b == 20:
                continue
            inner += b
        except ValueError:
            pass
    outer += inner
print(outer)


# --- THE CONTROL: no carried local, which always worked -------------------
for v in [1, 2, 3]:
    try:
        if v == 2:
            continue
        print("v", v)
    except ValueError:
        pass
print("done")
