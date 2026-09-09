# OPEN. The pre-pass that infers unannotated parameters from module-level call
# sites runs after classes are PREDECLARED but before their members are
# registered, so an argument expression that reads a field or calls a method of
# a user class types as nothing and the parameter stays a variable:
#
#     class Box:
#         def __init__(self, v: str) -> None:
#             self.v = v
#     def show(s):
#         return s + "!"
#     print(show(Box("a").v))
#     # function parameter 's' requires an annotation
#
# MEASURED 2026-09-09, RelWithDebInfo, both before and after the module-scope
# repair beside it (cases/a_module_name_passed_to_an_inferred_parameter) --
# every row below reads the same on both, so this is a separate mechanism and
# not a residue of that one. Each row is a program CPython runs.
#
#   the INSTANCE itself as the argument ..................... correct
#   `Box()` (no `__init__` arguments) as the argument ....... correct
#   `Box("a").v` as the argument ............................ refused
#   `b = Box("a")` then `b.v` as the argument ............... refused
#   `Box().get()` as the argument ........................... refused
#   `b = Box("a")` then `b.get()` as the argument ........... refused
#   a `with C() as v` target as the argument ................ refused
#   the class declared BELOW the def ........................ refused
#
# ⭐ THE INSTANCE TRAVELS, ITS MEMBERS DO NOT. Passing `Box("a")` binds the
# parameter to the contract, and the body walk then reads `b.v` off it without
# trouble -- so the members are known by the time a BODY is typed, and not by
# the time a module-level ARGUMENT is. The one walk that sees them last is the
# one this pre-pass uses.
#
# ⛔ The direct form is the useful witness, not the two-statement one: with
# `show(Box("a").v)` written at the call site there is no module name in the
# program at all, which is what separates this from the module-scope defect.
#
# ⭐ AND IT REACHES THE CONTAINER SEEDING, which is where it costs most. The
# scan that decides an empty container's element type runs inside the signature
# walk too, so a grouping whose ELEMENT is a user class cannot be typed there --
# while the same grouping over strings can (2026-09-09):
#
#     def by_bin(parts: "list[Part]"):
#         out = {}
#         for p in parts:
#             k = p.bin_id            <- the member read, unresolvable here
#             if k not in out:
#                 out[k] = []
#             out[k].append(p)
#         return out
#     print(len(by_bin([Part("a", "A1")])["A1"]))
#     # builtins.object does not provide manifest method '__len__'
#
#   the same function over `list[str]`, keyed on `w[0]` .......... correct
#   the same function with `out` annotated ....................... correct
#   a FLAT list of the same class instances ...................... correct
#   a list of lists of them (no member read in the key) .......... correct
#
# ⛔ Recomputing the signature at declaration time, which is what the generator
# arm of `emitTopLevelDeclarations` does for the same reason, does not work
# here: the fixpoint has already BOUND the result variable to the answer the
# early walk gave, and the second answer would collide with it rather than
# replace it.
#
# ⛔ Not repaired here because it is an ORDERING change in the emitter
# (EmitterCore's "after class/import predeclaration ... before any body is
# typed"), and moving `registerModule` past member registration moves every
# signature the fixpoint resolves along with it. Scoped, not built.
class Box:
    def __init__(self, v: str) -> None:
        self.v = v

    def get(self) -> str:
        return self.v


def show(s):
    return s + "!"


print(show(Box("a").v))
print(show(Box("a").get()))
