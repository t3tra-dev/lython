# WHAT: an exception class written in a function is raised out of the program,
# and the traceback names it by its qualified name, as CPython does.
#
# WHY THIS IS RUN: the traceback is printed by the running program, from the
# type object of what was raised.
def fail() -> None:
    class LocalError(ValueError):
        pass

    raise LocalError("bad thing")


fail()
