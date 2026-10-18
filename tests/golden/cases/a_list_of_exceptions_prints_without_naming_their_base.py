# WHAT: a list typed by a builtin exception class prints its elements, one of
# them a program exception, in a program that never names BaseException.
#
# WHY THIS IS RUN: the list renders each element through its class's slot at
# run time; the slot every exception falls back on is BaseException's, and a
# program that did not name it had that slot pruned as unreachable -- the
# print stopped the program.
class Coded(Exception):
    def __init__(self, code: int) -> None:
        super().__init__("code " + str(code))
        self.code = code

    def __repr__(self) -> str:
        return "Coded(" + str(self.code) + ")"


errs: list[Exception] = [Coded(5), ValueError("v")]
print(errs)
