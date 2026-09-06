# What: a plain `@decorator` on a def in an IMPORTED module. It was refused
# outright -- "an imported module's body does not run, so the decorator would
# never be applied and the undecorated function would answer under its name" --
# which was true until an imported module had a place to run things. It has one
# now: the module-global initializer queue, at the start of `__main__` in
# import order, and `f = d(f)` is exactly such an initializer.
#
# Running it is the whole evidence, because the failure mode the refusal named
# is a WRONG ANSWER and not an error: the undecorated body answers 2 where the
# wrapper answers 20. The driver inside the library is here because the name
# must reach the wrapper from that side too, and the stacked pair because the
# order the decorators apply in is what the result depends on.
#
# ⛔ The decorated TYPE is folded from the decorator's own SIGNATURE, not
# inferred from the call: a signature is computable with nothing bound, and
# inferring `d(f)` needs the module's own scope -- one pass later than where
# the importer's binders hand out the name. A decorator that is not a plain
# NAME declared in the same module keeps the refusal for that reason.
import a_module_of_decorated_functions as lib
from a_module_of_decorated_functions import both, scaled

print("qualified", lib.scaled(1), lib.scaled(3))
print("by name", scaled(1), both(1))
print("from inside the library", lib.driver(2))
