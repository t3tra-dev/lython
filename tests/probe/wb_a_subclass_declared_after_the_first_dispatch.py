# A subclass declared AFTER the first use of the base method that dispatches to
# it is refused:
#
#     'Loud.name' is used before 'Loud' is defined; a method of a class
#     declared later in the module cannot be resolved here, so move the class
#     above this use
#
# MEASURED (2026-09-02, RelWithDebInfo, today's tree):
#
#   every subclass declared before the first call ..... correct
#   one subclass declared after it .................... the message above
#   the same subclass in an IMPORTED module, with the
#     base's body emitted at import time .............. the same message, whose
#                                                       advice cannot be
#                                                       followed (the class is
#                                                       in another file)
#
# ⭐ THE BASE'S BODY IS EMITTED AT ITS FIRST USE, and the dispatcher it needs
# has to name every subclass that overrides the inner method. Subclasses
# declared after that point are not there to name. The refusal is honest and
# LOUD -- before the dispatch existed at all the same program silently ran the
# base's method -- but its advice is written for the one-file case.
#
# ⛔ The repair is the one `emitClassNow` already makes for a construction: pull
# the named classes forward before emitting the body that needs them. The
# dispatcher's candidate list is exactly that set of names, so it can ask for
# the same thing -- but it is built from `declaredClassBases`, which is a map of
# NAMES, and the puller works on the statements of the module being emitted.
class Shape:
    def name(self) -> str:
        return "shape"

    def describe(self) -> str:
        return self.name()


def describe(s: "Shape") -> str:
    return s.describe()


print(describe(Shape()))


class Loud(Shape):
    def name(self) -> str:
        return "loud"


print(describe(Loud()))
