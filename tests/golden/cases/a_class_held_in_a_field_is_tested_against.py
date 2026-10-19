# WHAT: `isinstance` and `issubclass` against a class value held in a field
# or returned by a call -- a class with no subclasses and one with -- alone
# and in a tuple, on either side.
#
# WHY THIS IS RUN: the class tested against is a value read at run time. A
# class value that is not spelled as a class name, but whose type alone names
# a class with no subclasses, was refused by issubclass ("requires statically
# resolvable class names"), while the same value of a class with subclasses
# was tested at run time.
class Holder:
    def __init__(self, expected: type[KeyError], wide: type[LookupError]) -> None:
        self.expected = expected
        self.wide = wide


def pick(flag: bool) -> type[KeyError]:
    return KeyError


h = Holder(KeyError, LookupError)
e: Exception = KeyError("k")
print(isinstance(e, h.expected), isinstance(ValueError("v"), h.expected), isinstance(e, h.wide))
print(issubclass(KeyError, h.expected), issubclass(ValueError, h.expected), issubclass(h.expected, h.wide))
print(issubclass(pick(True), LookupError), issubclass(KeyError, (h.expected, ValueError)))
