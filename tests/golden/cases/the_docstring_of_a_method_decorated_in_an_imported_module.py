# WHAT: `__doc__` of a method an imported module decorates -- a decorator that
# returns the function, and one that returns a wrapper with its own docstring
# -- read through a computed name, as unittest's TestCase reads the test
# method's, and a plain method's beside them.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the read raised Lython's
# TypeError at run time ("is decorated in an imported module, whose cell is not
# read from here"), and which docstring a decorated name answers is the one
# its decorator returned.
import the_docstring_of_a_method_decorated_in_an_imported_module_lib as lib

t = lib.Thing()
for name in ["kept", "wrapped", "plain"]:
    print(name, repr(getattr(t, name).__doc__))
print(t.kept(), t.wrapped(), t.plain())
