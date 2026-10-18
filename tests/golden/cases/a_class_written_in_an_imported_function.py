# WHAT: a class written in a function of an imported module is that module's:
# `__module__` is the module's name, `__qualname__` names the function, and
# the default repr prints both.
#
# WHY THIS IS RUN: the value's static type is the module's base class, so the
# names are read off the class's type object at run time.
from a_class_written_in_an_imported_function_lib import make

b = make()
print(b.tag(), type(b).__name__, type(b).__qualname__, type(b).__module__)
r = repr(b)
print(r[: r.index(" object at")])
