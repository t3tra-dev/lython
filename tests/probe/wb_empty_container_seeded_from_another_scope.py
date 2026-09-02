# An empty container keeps its erased element type when the operations that
# fill it are in a DIFFERENT scope from the assignment. Every position where
# the two are in the same scope now works.
#
# MEASURED (2026-09-02, RelWithDebInfo, today's tree), `xs = []` then a decode
# of an element:
#
#   in the same suite ................................ correct
#   in a for / while / try / with body, filled after .. correct
#   in one branch of an if, filled after ............. correct
#   filled by extend / insert / += / update /
#     setdefault / |= ................................ correct
#   a class FIELD, filled from another method ........ static type
#                                                      `builtins.object` does
#                                                      not provide ...
#   a module GLOBAL, filled inside a function ........ same
#   an outer local, filled inside a NESTED def ....... same
#
# ⭐ WHY THE LINE IS THERE: the seed scan (`emptyLiteralSeedType`) is a forward
# look over the suites the emitter is currently walking, and it stops at
# `suiteStackFloor` -- the callable boundary -- because the same name in an
# enclosing function is a different binding. The three failures are exactly the
# cases where the fill is on the other side of that floor, so they are not a
# wider scan of the same walk: the answer has to come from a pass over the
# whole class or module before any of it is emitted.
#
# ⛔ The field case is NOT the one `setField` answers. That rule refines a field
# whose FIRST assignment was empty when a LATER assignment in `__init__` gives a
# real one; here `__init__` has only the empty assignment and the element type
# exists only in another method's body.
class Bag:
    def __init__(self) -> None:
        self.xs = []

    def put(self, n: int) -> int:
        self.xs.append(n)
        return self.xs[0] + 1


b = Bag()
print(b.put(1))
