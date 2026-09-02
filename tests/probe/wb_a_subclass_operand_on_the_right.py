# CPython gives the RIGHT operand priority when its type is a proper subclass
# of the left's and overrides the reflected method. Lython always dispatches on
# the left, so a comparison between two base-typed values answers with the
# base's body where CPython answers with the subclass's:
#
#     a: Base = Sub()
#     b: Base = Base()
#     print(b == a)      # prints True; CPython prints False
#
# MEASURED (2026-09-03, RelWithDebInfo, today's tree). The shape is narrow, and
# what makes it narrow is WHICH method reflects onto which:
#
#   `a == b` (subclass on the LEFT) .................. correct
#   `b == a`, Sub overrides __eq__ ................... this file
#   `b != a`, Sub overrides __eq__ ................... the same shape
#   `b < a`,  Sub overrides __gt__ ................... the same shape
#   `b < a`,  Sub overrides __lt__ only .............. correct (`<` reflects
#                                                      onto `__gt__`, which the
#                                                      subclass does not have)
#   `b + a`,  Sub overrides __add__ only ............. correct (reflects onto
#                                                      __radd__)
#   the subclass overrides nothing ................... correct
#
# ⭐ `__eq__` IS ITS OWN REFLECTION, which is why equality is where this shows
# up and arithmetic mostly does not. A program that compares a heterogeneous
# list of base-typed values hits it; one that adds them does not.
#
# ⛔ The repair needs BOTH operands' runtime classes and a test of "is the
# right one a proper subclass of the left one", then a swap -- a two-operand
# priority, where the dispatcher this compiler has is a one-receiver test
# (`virtualDispatcherFor`). Refusing instead would take `x == y` over two
# base-typed values with it, which is the shape `x in xs` compiles to.
class Base:
    def __eq__(self, other: object) -> bool:
        return True

    def __hash__(self) -> int:
        return 1


class Sub(Base):
    def __eq__(self, other: object) -> bool:
        return False

    def __hash__(self) -> int:
        return 2


a: Base = Sub()
b: Base = Base()
print(a == b, b == a)
