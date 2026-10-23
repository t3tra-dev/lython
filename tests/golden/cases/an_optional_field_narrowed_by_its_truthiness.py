# `if self.w:` proves an optional field is not None on its true side, as
# `if x:` does for a local. The field's read in the arm was refused as a
# union ("'upper' runs on whichever member").
# WHY THIS IS RUN: the empty string is falsy and takes the other arm, which
# only the run tells apart from None.
class C:
    def __init__(self, w: str | None) -> None:
        self.w = w
    def go(self) -> str:
        if self.w:
            return self.w.upper()
        return "-"
def loc(x: str | None) -> str:
    if x:
        return x.upper()
    return "-"
print(C("a").go(), C(None).go(), C("").go(), loc("b"), loc(None))
