# A module-level CONTAINER or float constant cannot be read from another
# module:
#
#     module 'simple' has no attribute 'NAMES' that resolves statically
#
# MEASURED (2026-09-02, RelWithDebInfo, today's tree), `simple.py` declaring one
# constant of each kind and a second module reading it:
#
#   COUNT = 3 ......... correct
#   TEXT = "hi" ....... correct
#   FLAG = True ....... correct
#   RATIO = 1.5 ....... the message above
#   NAMES = ["a"] ..... the message above
#   TABLE = {"k": 1} .. the message above
#   PAIR = (1, 2) ..... the message above
#
# and every one of them is readable from the module's own body.
#
# ⭐ THE THREE THAT WORK ARE THE THREE THE MANIFEST CHANNEL CARRIES.
# `bindSourceModuleNamespace` binds functions and class names, and a constant
# rides the int/str/bool constant channels the manifest modules use
# (`ly.typing.int_constant_names` and its siblings). A float has no such
# channel and a container has no compile-time constant form at all, so neither
# is bound and the attribute does not resolve.
#
# ⛔ NOT the same question as a module GLOBAL in one file, which is a cell: a
# cross-module read has no cell to point at, because the importing program and
# the imported module do not share one module object. What the shape needs is
# the imported module's top-level binding emitted as a global the importer can
# read -- which is the machinery `collectModuleGlobals` builds for the main
# module and nothing builds for an imported one.
import a_module_with_constants as constants

print(constants.COUNT, constants.TEXT, constants.FLAG)
print(constants.NAMES, constants.TABLE, constants.PAIR, constants.RATIO)
