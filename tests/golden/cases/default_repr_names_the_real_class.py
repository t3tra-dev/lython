# What this pins: the default repr of an instance held in a base-typed name.
#
#     class A: pass
#     class B(A): pass
#     x: A = B()
#     print(x)      # printed <__main__.A object at 0x...>
#                   # CPython prints <__main__.B object at 0x...>
#
# The prefix was baked in at compile time from the STATIC contract, so the value
# reported the class it was HELD as rather than the class it IS. Nothing could
# see it: the address differs between runs, so no output comparison reads that
# far, and there is no diagnostic to assert on -- the compiler was sure.
#
# The instance header carries its class id in word 1 (the word isinstance
# reads), and the lowering synthesizes a name table for every class the program
# declares, so the repr now reads them at run time.
#
# Why this must run: the whole defect is a string only the runtime produces.
# The address is split off because it is not reproducible; what is left is
# exactly the part that was wrong.
#
# ⛔ Manifest objects keep the compile-time prefix: their header word 1 is not a
# class id, and the contracts that reach the default repr have no subclass to be
# wrong about.
#
# ⛔ An instance passed through an `object` parameter still prints
# `<object object at ...>`: that path boxes the value and renders it through the
# erased manifest dispatch, which reads the BOX rather than the instance header.
# Same defect, different reader, and it is why this golden does its splitting
# inline instead of through a helper.


class A:
    pass


class B(A):
    pass


class C:
    def __init__(self, v: int) -> None:
        self.v = v


x: A = B()
y = A()
z = C(1)
print(str(x).split(" at ")[0], str(y).split(" at ")[0])
print(str(z).split(" at ")[0], repr(B()).split(" at ")[0])
print(type(x).__name__, type(y).__name__)
