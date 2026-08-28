# WHAT: `print(..., end=...)` and `print(..., sep=...)`, including the two
# spellings with no arguments at all, and the loop that is the reason `end=""`
# exists.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: what is checked is the
# BYTES on stdout. `end` decides whether a newline is written, so every way of
# getting it wrong -- writing the default too, dropping it, writing it before
# the arguments -- produces output that still looks like output.
#
# ⛔ ONE write and not two: `end` is concatenated onto the joined arguments
# rather than written after them. CPython makes two calls; the difference is
# only observable through a `file` argument this fold does not take.
import sys

print("a", end="")
print("b", end="")
print()

print(1, 2, 3, end="!\n")
print("x", "y", sep="-", end="|")
print()

# No arguments at all, with and without a terminator.
print(end="")
print(end="Z\n")
print()

for i in [1, 2, 3]:
    print(i, end=" ")
print()

# A computed terminator, and every renderer the join goes through.
terminator = "//" + "\n"
xs: "list[int]" = [4, 5]
print(xs, {"a": 1}, (1, 2), 2.5, None, True, end=terminator)
print("unicode 日本語", end="…\n")

# The union argument the single-argument path cannot take still renders here.
d: "dict[str, int]" = {"k": 3}
print(d.get("k"), end=" ")
print(d.get("zz"), end="\n")

sys.stdout.write("after\n")
