# OPEN, and found 2026-08-15 while closing the set table order
# (tests/probe/wb_small_int_set_iteration_order.py). It is NOT the set: the set
# now agrees with CPython on every construction that goes through add, discard
# and the algebra. It is CPython's COMPILER.
#
#     t = {0, 7, 14, 21, 28}
#     CPython 3.14   {0, 21, 7, 28, 14}
#     lyc            {0, 7, 14, 21, 28}
#
# ⭐ WHAT CPYTHON DOES. `codegen`'s starunpack_helper folds a displayed set
# whose elements are ALL CONSTANTS and whose length is > 2 into a frozenset
# constant, then emits an empty set and merges:
#
#     BUILD_SET 0
#     LOAD_CONST frozenset({0, 7, 14, 21, 28})
#     SET_UPDATE 1
#
# So the literal's table is set_merge's output, not the adds' -- and the two
# differ because the frozenset constant was built by adds into ITS own table
# and then re-placed into the destination's, which resized on a different
# schedule. Lython emits the adds, which is what CPython does for every
# literal that is not folded.
#
# MEASURED, and the window is narrow enough to be worth writing down. The two
# tables' masks coincide except while the fold's frozenset has resized and the
# destination has not yet caught up:
#
#   n = 3, 4 ....... agree (both mask 7; set_merge takes the wholesale path)
#   n = 5, 6 ....... DIFFER (frozenset mask 31, destination mask 15)
#   n >= 8 ......... agree again
#
# ⭐ AND THE CONDITION IS THE COMPILER'S, not the value's:
#
#   a = -20
#   {a, -15, -27, -23, -22, 2} ..... agrees (a Name blocks the fold; plain
#                                    BUILD_SET 6, which is what lyc emits)
#   {-20, -15, -27, -23, -22, 2} ... differs (all constants, folded)
#
# So the same six elements print in two different orders in CPython depending
# on whether one of them is spelled as a variable. Any repair has to reproduce
# that, which means the emitter has to decide, on the AST, whether CPython's
# optimizer would have folded -- including its own constant folding, since
# `{1+1, 2, 3}` reaches the check as three Constants.
#
# ⛔ Why NOT do it in the runtime: the runtime cannot see the difference. Both
# spellings arrive as the same sequence of add_box calls. The only thing the
# runtime would need is a "now merge yourself into a fresh empty set and swap
# bodies" primitive, which is three lines on top of what
# @__ly_set_raw_merge_set and @__ly_set_raw_swap_bodies already do -- the work
# is entirely in the emitter deciding when to call it, and in emitting a call
# to something that is not a Python method (a py dialect op, since the set
# literal is emitted as a py.pack plus synthesised `.add(...)` statements in
# EmitterExpressions.cpp:emitSetLiteral).
#
# differential: expect-wrong the point is the stdout divergence

print({0, 7, 14, 21, 28})
print({0, 7, 14, 21, 28, 35})
