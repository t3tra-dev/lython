# WHAT: module-level constants of another module computed from its other
# constants -- integer arithmetic with floor division, a negative modulo,
# shifts and bit operations, and string concatenation and repetition --
# read by that module's function and from here.
#
# WHY THIS IS RUN: the values are computed as Python computes them, and a
# floor division or modulo of a negative number is where a fold can differ.
# Unannotated, each was refused ("its type is not one a module global can
# hold ... Annotate it") while the literal constants beside them resolved.
import a_computed_constant_of_an_imported_module_lib as lib

print(lib.show())
print(lib._DIFF + 1, lib._PREFIX.upper(), len(lib._RULE))
