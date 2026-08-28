# WHAT: `for line in f` -- the way a text file is read line by line -- over a
# file with a blank line, a line with no trailing newline, an empty file, and
# with `break`, `continue` and `else` in the loop.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: every part of the answer is
# a value. The lines come back with the newline they ended with and the last
# one may have none; the loop must stop at the empty string EOF returns and not
# one line early or late; and `else` runs on exhaustion but not after a break,
# which is the difference between ending the loop with a `break` and ending it
# with the condition.
#
# ⛔ The file is READ ONE LINE AHEAD of the body, because the advance has to
# precede the body for `continue` to terminate. CPython's own file iterator
# buffers ahead too, so mixing this loop with a direct `read()` on the same
# handle is unpredictable in both.
#
# ⛔ The temporary is named for this case: the golden runner compares stdout in
# a shared working directory, so two cases writing one name race under -j.
import os
import sys

path = "io_line_iteration_case.tmp"

with open(path, "w") as f:
    f.write("alpha\n")
    f.write("\n")
    f.write("beta\n")
    f.write("no newline")

def show(text: str) -> str:
    return "<" + text.replace("\n", "\\n") + ">"


with open(path) as f:
    for line in f:
        sys.stdout.write(show(line) + "\n")

seen = 0
with open(path) as f:
    for line in f:
        seen += 1
        if seen == 2:
            break
sys.stdout.write("broke after " + str(seen) + "\n")

kept = 0
with open(path) as f:
    for line in f:
        if line == "\n":
            continue
        kept += 1
sys.stdout.write("kept " + str(kept) + "\n")

with open(path) as f:
    for line in f:
        pass
    else:
        sys.stdout.write("exhausted\n")

with open(path) as f:
    for line in f:
        if line == "alpha\n":
            break
    else:
        sys.stdout.write("not reached\n")
sys.stdout.write("after break-else\n")

total = 0
with open(path) as f:
    for line in f:
        total += len(line)
sys.stdout.write("bytes " + str(total) + "\n")

empty = "io_line_iteration_empty.tmp"
with open(empty, "w") as f:
    pass
# ⛔ `str(count)` inside the `with` is the shape a rebound name is promoted to
# storage for: its static type there is the cell, not the int.
count = 0
with open(empty) as f:
    for line in f:
        count += 1
    else:
        sys.stdout.write("empty file, " + str(count) + " lines\n")

os.remove(path)
os.remove(empty)
