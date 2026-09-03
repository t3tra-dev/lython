# OPEN, and now REFUSED where the program can read the reason. An Enum class in
# an imported module:
#
#     # a_module_of_enum_colors.py
#     from enum import Enum
#     class Color(Enum):
#         RED = 1
#
#     import a_module_of_enum_colors as c
#     print(c.Color.RED.name)
#
# used to reach the dialect verifier as "'py.class' op unknown base class
# 'Enum'" -- the compiler's own sentence for a class CPython has no trouble
# with. The same class in the MAIN module works, and an imported class deriving
# from a BUILTIN base (`class E(Exception)`) works too, so neither the import
# nor the inheritance is the problem.
#
# ⛔ WHY. `desugarEnumClasses` runs on the main module's node only, and it is
# what turns `class Color(Enum)` into a plain class whose members are
# constructed instances -- including dropping the `Enum` base the verifier
# cannot resolve. An imported module never reaches it.
#
# ⛔ AND DESUGARING IT THERE IS NOT ENOUGH. A desugared member is an INSTANCE
# held as a class attribute, and an imported class attribute is carried by the
# LITERAL channel -- its type IS the value -- which has no form for an object.
# That is the same wall the imported container constant stands behind
# ([[wb_a_container_constant_in_an_imported_module]]), and it is the mechanism
# both need: an imported module's top-level binding emitted as something the
# importer can read at run time rather than materialize at compile time.
#
# Measured 2026-09-04.
import a_module_of_enum_colors as c

print(c.Color.RED.name, c.Color.RED.value)
