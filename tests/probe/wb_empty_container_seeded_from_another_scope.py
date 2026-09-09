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
#   a class FIELD, filled from another method ........ CORRECT NOW
#   a module GLOBAL, filled inside a function ........ static type
#                                                      `builtins.object` does
#                                                      not provide ...
#   an outer local, filled inside a NESTED def ....... same
#
# ⭐ WHY THE LINE IS THERE: the seed scan (`emptyLiteralSeedTypeIn`) is a
# forward look over the suites it is handed, and a caller hands it the ones it
# is walking -- which stop at the callable boundary, because the same name in
# an enclosing function is a different binding. The two failures are exactly
# the cases where the fill is on the other side of that boundary.
#
# ⭐ THE FIELD CASE WAS NOT (2026-09-09). Every method of a class is available
# to `collectClassFields` before any of it is emitted, so the class-wide pass
# the note below asked for already existed; what was missing was a scan that
# could be asked about `self.xs` rather than about a bare name. It takes a
# receiver now, and is asked once per method with that method's parameters in
# scope. Two methods that disagree leave the field erased.
#
# ⛔ What is left needs a pass over the whole MODULE before any of it is
# emitted, which is a different thing: a global is filled by functions whose
# bodies are typed against the global, and an outer local by a nested def whose
# own signature the enclosing walk has already answered for.
#
# ⛔ The field case was NOT the one `setField` answers. That rule refines a
# field whose FIRST assignment was empty when a LATER assignment gives a real
# one; a fill through `append` is not an assignment.
XS = []


def put(n: int) -> None:
    XS.append(n)


put(1)
print(XS[0] + 1)
