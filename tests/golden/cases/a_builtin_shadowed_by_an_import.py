# What: a builtin shadowed by `from module import <name>`. The test for "does
# the program bind this name itself" asked only about the three things the
# module writes -- its values, its defs and its classes -- so an IMPORT left
# every builtin fast path visible and the BUILTIN ran:
#
#     # helpers.py: def len(v): return 99
#     from helpers import len
#     print(len("abc"))     # printed 3; CPython prints 99
#
# Five measured wrong the same way, all silently, and the one-file spelling of
# every one was already right -- which is what says the import binding is the
# gap and not the shadowing rule. A canonical binding is what an import leaves
# behind and nothing else in a main module makes one.
#
# Only running it can catch this: the program compiles either way and the
# answer is the difference.
from a_module_of_shadowing_builtins import abs, len, max, repr, sum


def called_in_a_body() -> int:
    return len("abcd") + abs(-1)


print("len", len("abc"))
print("abs", abs(-3))
print("sum", sum([1, 2, 3]))
print("max", max(1, 2))
print("repr", repr(3))
print("inside a function", called_in_a_body())
