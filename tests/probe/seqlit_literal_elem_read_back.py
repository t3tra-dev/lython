# ⛔ GUARD PROBE. CLEAN on main 4699488 and must stay clean.
#
# Companion to seqlit_single_loop_read_back.py. Same guard, but with NO loop
# variable in the container at all -- the element is a literal `7`. So neither the
# cross-loop borrow nor the loop variable is required to break this; only the
# element READ-BACK is.
#
# That matters for attribution: the nested-loop over-release
# (seqlit_outer_var_nested_overrelease.py) and the refusal this probe guards
# against have DIFFERENT necessary conditions, so a repair aimed at the first can
# regress the second without any shape in common. Refused with
#   `released owned resource ... is used after release (by call to 'LyLong_Add')`
# if the affine walk is made to skip `aggregate_retain`.
total = 0
for i in range(3, 4):
    for j in range(2):
        ys = [7]
        total += ys[0]
print(total)  # CPython 3.14: 14
