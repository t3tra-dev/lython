# DEVIATION, deliberate and measured. A guard proves a field is not None; a
# call between the guard and the read replaces it with None; the read RAISES
# where CPython returns the None.
#
#     def use(self) -> str:
#         if self.v is not None:
#             self.clear()          # sets self.v = None
#             return self.v         # CPython: None. Here: AttributeError.
#
# The guard's proof is about the past, and the field is re-read at every use.
# The read is therefore CHECKED against the proof rather than assuming it --
# see cases/a_guard_over_a_field for the half this makes work. Where the check
# fails the two honest answers are to raise, or to hand back a value the
# declared type says the caller cannot receive; this raises.
#
# ⛔ NOT the same as dropping the narrowing at the call. That would refuse the
# read outright, which is a worse answer for the program that never changes the
# field -- and the check costs one branch, which is what buys the BST idiom.
#
# Measured 2026-09-03: CPython prints None, this raises AttributeError naming
# the field and saying it changed.
class Box:
    def __init__(self) -> None:
        self.v = None

    def fill(self) -> None:
        self.v = "x"

    def clear(self) -> None:
        self.v = None

    def use(self) -> str:
        if self.v is not None:
            self.clear()
            return self.v
        return "-"


b = Box()
b.fill()
print(b.use())
