# WHAT: print() writes each argument's str() and each separator in turn, as
# CPython's builtin_print_impl does, and evaluates `sep=` once; a starred
# argument is its members, each a piece of its own.
#
# WHY THIS IS RUN: what is observed is the order of the bytes on the
# process's own stdout -- a `__str__` that prints writes between the pieces,
# and one that raises leaves the pieces before it written. A replaced
# sys.stdout takes another path and cannot see it.
class Loud:
    def __str__(self) -> str:
        print("inner")
        return "loud"


class Broken:
    def __str__(self) -> str:
        raise ValueError("no text")


seps = 0


def sep() -> str:
    global seps
    seps += 1
    return "-"


print("a", Loud(), "b")
try:
    print("before", Broken(), "after")
except ValueError:
    print("|caught")
print(1, 2, 3, sep=sep())
print("sep evaluated", seps, "time(s)")
xs = [1, 2, 3]
print(*xs)
print(*xs, sep=", ", end=".\n")
print("x", *xs, "y", sep="/")
print(*[], "only")
print(*"ab", *range(2))
print(*[Loud(), Loud()])
