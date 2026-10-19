# `type(e).__name__` and `repr(e)` of an exception whose class lives in a
# module with a long name print the whole class name. They copied the
# module-qualified name into a 64-byte buffer and took what followed the last
# dot: under a module name of 59 characters `LibError` printed as `LibE`, and
# past 64 the module's own name came out. A plain class's name, read the same
# way without the cap, was right.
# Why run it: the name is read from the class the exception has at run time.
import an_exception_class_of_a_long_named_module_keeps_its_whole_name_lib as lib


def name_of(e: BaseException) -> str:
    return type(e).__name__


try:
    raise lib.LibError("boom")
except Exception as e:
    print(name_of(e), repr(e), type(e).__name__)
print(repr(lib.LibError()), repr(lib.LibError("x")))
