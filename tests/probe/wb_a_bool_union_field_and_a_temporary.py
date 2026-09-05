# OPEN 2026-09-05. A `bool | None` FIELD, narrowed by `is None` and then read
# again inside the guard, is refused when the receiver is a TEMPORARY and its
# owned-local marker does not stand in the entry block:
#
#     error: owned resource from builtin.unrealized_conversion_cast result 0 is
#     still owned when 'LyAttributeError_Raise' unwinds out of the function;
#     the exception path must release, transfer, or return it
#
# The raise is the CHECKED read the narrowing emits (`lython-where-a-proof-is-
# spent`), so the enclosing frame holds the instance across a may-unwind call
# and no cleanup is placed for it.
#
# ⭐ WHAT SEPARATES THE PASSING SHAPES FROM THIS ONE, all measured:
#   - `int | None` and `str | None` fields pass every shape below. Only bool
#     fails, because only bool's union payload rides in SSA: the instance's
#     owned-local marker has three results (header, tag, truth bit) where the
#     others have one.
#   - ONE temporary passes: `print(Box(True).show())`.
#   - Two NAMED receivers pass: `b1 = Box(True); b2 = Box(None); b1.show() ...`
#   - Two temporaries in separate statements fail, two temporaries of DIFFERENT
#     classes fail, and one temporary inside a `for` fails. What they share is a
#     marker that is not in the entry block.
#   - The same method as a FREE FUNCTION taking the instance passes, and so does
#     a body that narrows without re-reading the field.
#
# ⛔ NOT the deallocator arity guard in `collectOwnedLocalObjectGroups`: a
# source class's deallocator takes the whole three-lane group, so the exact-
# width test it applies already passes here. Loosening it to a prefix rule
# fixed none of the programs below and was dropped.
class Box:
    def __init__(self, v: bool | None) -> None:
        self.v: bool | None = v

    def show(self) -> str:
        if self.v is None:
            return "n"
        return "t" if self.v else "f"


print(Box(True).show())
print(Box(None).show())
for i in range(2):
    print(Box(i == 0).show())
