# `traceback.format_exception_only(e)` spells the class as CPython's
# TracebackException does: its qualified name, after its module unless that
# is __main__ or builtins. An imported module's exception printed as
# `LibError: lib`, where CPython prints `<module>.LibError: lib`, and a class
# written in a function lost its `make.<locals>.` -- in unittest's failure
# reports, which end with this line. The uncaught-exception printer already
# wrote both right.
# Why run it: the line is built at run time from the class the exception has.
import traceback

import format_exception_only_spells_the_class_as_cpython_does_lib as lib


class MainError(Exception):
    pass


def make() -> BaseException:
    class Local(Exception):
        pass
    return Local("local")


excs: list[BaseException] = [lib.LibError("lib"), MainError("main"), make(), KeyError("k")]
for exc in excs:
    print(traceback.format_exception_only(exc), end="")
print()
