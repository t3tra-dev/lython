# What this pins: slicing across the adaptive-width representations, which is
# where the slice now BRANCHES.
#
# A slice of a width-1 source cannot contain a code point above 255, so the
# pass that picks the narrowest output width is skipped for it and the copy is
# bytes rather than code points -- the two things CPython does for a latin-1
# source. A wider source still runs that pass, because a slice of it may be
# all-ASCII and has to narrow; and a step other than one is not a contiguous
# run, so it keeps walking ordinals.
#
# `str_widths` pins mixed-width CONCATENATION and indexing. Slicing had
# nothing on it, and it is the operation with the width branch.
#
# Why this needs to run: the output's width is not observable directly. What
# is observable is what the slice CONTAINS and what it compares equal to, so
# the assertions are values and the wrong width shows up as wrong characters.

one = "abcdefghij"
two = "caf\xe9" + "\xe9\xe8\xe7" * 3
four = "a\U0001F600b\U0001F601c\U0001F602"

# --- width 1: the scan is skipped and the copy is bytes ---------------------
print(one[2:7], one[:4], one[6:], one[-3:], one[:])
print(one[::2], one[::-1], one[1:8:3])

# --- width 2: the scan runs, and an all-ASCII window must NARROW ------------
print(two, len(two))
print(two[0:3], two[0:3] == "caf")
print(two[3:6], two[-4:], two[::2], two[::-1])
print(len(two[0:3]), len(two[3:]))

# --- width 4: a slice that drops every wide code point narrows too ----------
print(four, len(four))
print(four[0:1], four[0:1] == "a", four[1:2] == "\U0001F600")
print(four[::2], four[::-2], four[2:5])
print(four[0:1] + four[2:3], four[0:1] + four[2:3] == "ab")

# --- the narrowed slice must behave like the string it equals --------------
narrowed = two[0:3]
print(narrowed + "x", narrowed * 2, narrowed.upper(), narrowed.find("af"))
wide_kept = four[1:2]
print(wide_kept + "z", len(wide_kept + "z"), (wide_kept + "z")[1])

# --- empty and degenerate windows ------------------------------------------
print(repr(one[5:2]), repr(two[9:9]), repr(four[0:0]), repr(one[100:200]))
