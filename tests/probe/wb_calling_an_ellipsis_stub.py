# DEVIATION, deliberate and recorded. A method whose body is `...` and whose
# declared result cannot hold None RAISES NotImplementedError when it is
# actually called; CPython returns None.
#
# The declaration itself compiles now (cases/a_method_declared_with_an_ellipsis
# _body), which is the whole point of the repair -- `def area(self) -> int: ...`
# in a base class is how Python declares an abstract method, and it used to be
# refused outright.
#
# ⛔ WHY NOT CPython'S ANSWER. The body has no value to fall through with: the
# annotation says `int` and None is not one, so returning None would be a value
# the ABI cannot expand into the declared result's lanes. The two honest
# choices are to refuse the DECLARATION -- which is the defect this replaced --
# or to refuse the CALL, loudly, at run time. `raise NotImplementedError` is
# also the spelling a hand-written stub uses for exactly this, and it has
# always compiled.
#
# ⛔ AND ONLY WHERE None DOES NOT FIT. `def before(self) -> None: ...` returns
# None here and in CPython; the golden pins that half.
#
# Measured 2026-09-03: CPython prints None, this prints the raise.
class Shape:
    def area(self) -> int: ...


try:
    print(Shape().area())
except NotImplementedError as e:
    print("stub:", str(e))
