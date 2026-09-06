# What: `__name__` of a function that came from another file. The fold accepted
# only a name this module's own `def` bound, so `from lib import compute` was
# refused -- "needs the name the `def` gave" -- while `lib.compute.__name__`
# one line over folded fine. One question, two spellings.
#
# The DEF's name is the leaf of the canonical symbol the import bound, which is
# stricter than the refusal's own worry rather than looser: an `as` alias
# answers the def's name here, which is what CPython answers, where a local
# `g = f` could not.
#
# Only running it can catch what the first attempt did: a StringRef into a
# temporary folded seven NUL bytes for "compute" -- the right length and no
# content, printed with no diagnostic.
import a_module_of_named_functions as lib
from a_module_of_named_functions import compute
from a_module_of_named_functions import other as renamed


def local(n: int) -> int:
    return n


class Shape:
    def area(self) -> int:
        return 1


print("by name", compute.__name__)
print("qualified", lib.compute.__name__)
print("aliased", renamed.__name__)
print("local", local.__name__, Shape.area.__name__, Shape.__name__)
print("in a list", [f for f in [compute.__name__, renamed.__name__]])
print("concatenated", compute.__name__ + "/" + str(compute(1, 2)))
