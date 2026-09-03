# A module-level CONTAINER constant cannot be read from another module:
#
#     module 'simple' has no attribute 'NAMES' that resolves statically
#
# MEASURED (2026-09-02, RelWithDebInfo, today's tree), `simple.py` declaring one
# constant of each kind and a second module reading it:
#
#   COUNT = 3 ......... correct
#   TEXT = "hi" ....... correct
#   FLAG = True ....... correct
#   RATIO = 1.5 ....... CORRECT as of 2026-09-04 (the literal channel gained a
#                       float arm, and `widenLiteral` learned to read a float
#                       SPELLING -- a decimal point, an exponent, or a
#                       non-finite name -- instead of taking every unquoted
#                       spelling for an int)
#   NEG = -1 .......... CORRECT as of 2026-09-04 (a negative literal is a
#                       UnaryOp over a Constant, which missed the channel for
#                       ints as much as for floats)
#   NAMES = ["a"] ..... the message above
#   TABLE = {"k": 1} .. the message above
#   PAIR = (1, 2) ..... the message above
#   NOTHING = None .... CORRECT as of 2026-09-03 (the literal channel already
#                       carried the None spelling; only this producer left it
#                       out)
#
# and every one of them is readable from the module's own body.
#
# ⭐ WHAT WORKS IS WHAT A LITERAL TYPE CAN SPELL. `bindSourceModuleNamespace`
# binds functions and class names, and a constant rides the literal channel:
# its TYPE carries the value, so the importer materializes it with no module
# state at all. A container has no compile-time literal form, so it is not
# bound and the attribute does not resolve. That is now the only gap left in
# this list.
#
# ⛔ NOT the same question as a module GLOBAL in one file, which is a cell: a
# cross-module read has no cell to point at, because the importing program and
# the imported module do not share one module object. What the shape needs is
# the imported module's top-level binding emitted as a global the importer can
# read -- which is the machinery `collectModuleGlobals` builds for the main
# module and nothing builds for an imported one.
import a_module_with_constants as constants

print(constants.COUNT, constants.TEXT, constants.FLAG)
print(constants.RATIO)  # correct as of 2026-09-04
print(constants.NAMES, constants.TABLE, constants.PAIR)
print(constants.NOTHING is None)  # correct; the four above are not
