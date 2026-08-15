# FIXED 2026-08-15. A set of small ints used to iterate in INSERTION order;
# CPython iterates its hash table's slots, which for small ints is the value
# itself, so CPython printed them ascending and Lython printed them as written.
# It was the last member of the differential's WRONG bucket -- a stdout
# divergence with no diagnostic anywhere.
#
# ⭐ WHAT IT WAS. There was no hash table. `builtins.set` was a dense array of
# 16-word boxes with a LINEAR probe over the live prefix; the hash was computed
# and cached in box word 15 and then used only to skip a comparison. Everything
# about the set was correct except the one thing an array cannot have, which is
# a slot order.
#
# ⭐ WHAT REPLACED IT. CPython's table, transcribed into builtins.mlir:
# PySet_MINSIZE 8, LINEAR_PROBES 9, PERTURB_SHIFT 5, the `fill*5 >= mask*3`
# growth trigger and the `used > 50000 ? used*2 : used*4` target, with the
# dense array KEPT IN SLOT ORDER so that `items[0..used)` -- what __repr__, the
# for-loop lowering, list(s) and every algebra scan already read -- is CPython's
# order without any of them knowing a table exists.
#
# ⭐ THE MODEL CAME FIRST, and three of its details would not have been guessed:
#
#   1. The freeslot is the LAST dummy in a probe run, not the first. CPython
#      writes `freeslot = entry` with no null check. Taking the first
#      disagreed on 20 of 1632 measured insert/discard sequences.
#   2. Every special case in set_merge / set_intersection / set_difference is
#      load-bearing for the ORDER, not just for speed. Measured over 2000
#      random pairs, dropping each one costs:
#        set_merge's wholesale table copy .......... 274 copies wrong
#        set_merge's up-front resize ............... 1342 copies wrong
#        set_intersection's smaller-operand swap ... 44 wrong
#        set_difference's copy-and-discard path .... 195 wrong
#      There is no simplification of CPython's algebra that keeps its order.
#   3. symmetric_difference is a copy of the RIGHT operand toggled by the left,
#      not difference(l,r) followed by difference(r,l). The identity holds for
#      the contents and not for the build order.
#
#   Validated before a line of MLIR was written: 1632/1632 insert/discard
#   sequences and 1500/1500 algebra pairs against python3.14, then 150/150
#   generated whole programs against the built compiler.
#
# ⛔ WHAT IT COST, because the ordered dense array is what buys the ordering
# for free. An insert is O(n) -- it shifts the tail and renumbers the slots
# after it -- which is the class the linear probe was already in, so nothing
# regressed asymptotically, and lookup went the other way:
#
#   20k adds     2.7 s -> 5.4 s
#   200k `in`    4.6 s -> 2.6 s
#
# The route to O(1) inserts is written down at the layout comment in
# builtins.mlir; it was not taken here.
#
# ⭐ AND IT FOUND A SECOND, SEPARATE DIVERGENCE, which is NOT the set's:
# CPython's COMPILER folds an all-constant set literal of more than two
# elements into a frozenset constant and emits BUILD_SET 0 + SET_UPDATE, so the
# literal's table is built by a merge rather than by the adds. See
# tests/probe/wb_const_set_literal_fold.py. Everything below agrees with
# python3.14 today.
#
# golden: tests/golden/cases/set_table_order.py (red-checked against the
# pre-fix binary; also in LYTHON_LEAK_GATE_CASES, since the table is a second
# allocation per set and three of the new paths move references in bulk)

print({1, 0})
print({2, 1, 0})
