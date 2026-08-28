# WHAT: a module-level name bound to a FUNCTION VALUE, read from inside a
# function body -- as a call, as an argument, and out of a list.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the call has to reach the
# function the name currently holds. A name that resolved to the wrong
# function compiles and prints a number of the right shape.
#
# ⛔ THE CELL IS DECLARED FROM THE STATIC TYPE, and a callable's is spelled
# `py.callable` rather than as a contract -- which is why the declaration pass
# skipped it and every such read was "unresolved name". The VALUE is an
# ordinary `builtins.function` object; only the type spelling differed.
def double(n: int) -> int:
    return n * 2


def add_one(fn):
    def wrapper(n: int) -> int:
        return fn(n) + 1
    return wrapper


CALLBACK = double
WRAPPED = add_one(double)


def through_callback(n: int) -> int:
    return CALLBACK(n)


def through_wrapper(n: int) -> int:
    return WRAPPED(n)


def apply_twice(n: int) -> int:
    return CALLBACK(CALLBACK(n))


print(CALLBACK(3), through_callback(3))
print(WRAPPED(3), through_wrapper(3))
print(apply_twice(3))
print([through_wrapper(v) for v in [1, 2, 3]])
