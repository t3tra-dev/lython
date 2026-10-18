# WHAT: print(), str(), repr(), ascii(), format(), str.format, %-formatting and
# f-strings with each conversion render a class -- a builtin, a program class,
# and the result of type() -- as CPython does, alone and inside containers.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the text is built at run
# time by `type`'s own `__repr__` from the type object the class value
# references. Every one of these spellings used to be refused ("cannot render
# a class"), each through a path of its own.
class A:
    pass


xs = [1]
print(int, type([1]), str(int), repr(int), str(type(xs)))
print(f"{int}", f"{type(xs)}", str(A), repr(A), f"{A!r}", f"{int!a}", f"{A!s}")
print([int, A], (A,), {A: int})
print("%s %r" % (A, int))
print(ascii(A), format(int), "{}".format(A))
