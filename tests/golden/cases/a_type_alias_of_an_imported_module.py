# WHAT: another module spells its parameters, fields and results through
# type aliases of its own -- a tuple of a class, an exception and a
# traceback, a tuple, a list, and a PEP 695 `type` alias -- and this module
# calls its functions and methods with values of those types.
#
# WHY THIS IS RUN: the values cross the module boundary and are read back
# at run time. The aliases were read as classes of their names in the other
# module ("static type builtins.Info does not provide manifest method
# '__getitem__'"); written in this module they always worked.
import sys
import a_type_alias_of_an_imported_module_lib as lib



def fail(kind: str) -> None:
    if kind == "key":
        raise KeyError(kind)
    raise ValueError(kind)


print(lib.pair_text((1, "a")))
r = lib.Recorder()
for kind in ["key", "value", "key"]:
    try:
        fail(kind)
    except Exception:
        print(lib.first_name(sys.exc_info()), r.record(sys.exc_info()))
print(r.counts)
