# What this pins: a class attribute declared float and initialised with an int.
#
#     class P:
#         v: float = 1
#     print(P.v)
#     # RuntimeError: module global 'P.v' referenced before assignment
#
# A runtime failure, naming an internal cell, for a program whose problem is
# visible in its first two lines. The module-scope spelling of the same thing --
# `x: float = 1` -- has always refused it at emit, because `coerceValue`
# deliberately declines to retype between the numeric contracts ("int, float and
# bool share no representation") and the write says so. A class attribute's cell
# is the same storage under the same rule, and this channel had no such check:
# the store of an int into a float cell was dropped further down, leaving the cell
# unassigned.
#
# The refusal names the attribute and what to write instead. CPython prints 1
# here: its annotation is inert and the value stays an int, which this compiler
# cannot do for a cell whose representation is fixed by the declaration -- the
# same deviation the module global already documents.
class P:
    v: float = 1


print(P.v)
