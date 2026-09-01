# An Optional FIELD is not narrowed by a check on it, though the same check on
# a local is:
#
#   if self.v is None: return 0
#   return self.v + 1
#   # static type !py.union<int, None> does not provide manifest method
#   # '__add__'
#
# The local spelling works, and so does the ternary over a local:
#
#   def f(v: "int | None") -> int:
#       if v is None: return 0
#       return v + 1            # fine
#
# ⭐ NARROWING IS KEYED ON A NAME. `optionalNoneComparison` asks
# `nameComparedWithNone`, which accepts only a Name node, and
# `applyBranchNarrowing` finds the SSA value to unwrap in `values[fact.name]`.
# An attribute read has neither: `self.v` is re-read from the object at every
# use.
#
# ⛔ THE MISSING PIECE IS INVALIDATION, not the narrowing. Recording the
# narrowed type per path ("self.v") and unwrapping the read is four lines, and
# unsound on its own -- the field can be written between the check and the
# read, and an unwrap to the wrong member is a SILENT mis-execution:
#
#   if self.v is not None:
#       self.reset()       # sets self.v = None
#       print(self.v)      # CPython prints None
#
# A sound version has to drop the fact at every point that can run user code
# or re-enter: a call (including an inlined operator dunder, which runs a
# source __add__), an attribute store, a loop back edge, a handler, a yield or
# an await. Each of those is a separate emission site, and a missed one is
# exactly the silent class this compiler exists to avoid -- so it is recorded
# rather than half-built.
class C:
    def __init__(self) -> None:
        self.v: "int | None" = 1

    def get(self) -> int:
        if self.v is None:
            return 0
        return self.v + 1


print(C().get())
