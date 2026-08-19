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
# A CONTAINER of such instances used to ABORT -- "repr: boxed element has no
# conforming __repr__" -- because the element dispatch asserted instead of
# falling back. It falls back now, and what stood in the way was not the repr:
# the fallback helper returns an OWNED result, and a hand-written manifest
# helper without any ly.runtime.* attribute is treated as USER code by the
# refcount pass, which inserted a release for the hook result on the path that
# does not use it. The hook's miss returns poison, so that release freed
# garbage and the program aborted inside malloc.
#
# The ERASED reader is here too: an instance passed through an `object`
# parameter, or held in a list[object], is boxed and rendered by the manifest
# dispatch -- which reads the BOX. The box carries the class id in the same word,
# so it names the real class as well, and an int or a str going the same way
# still prints its own repr rather than an address.


class A:
    pass


class B(A):
    pass


class C:
    def __init__(self, v: int) -> None:
        self.v = v


def erased(v: object) -> str:
    return str(v).split(" at ")[0]


x: A = B()
y = A()
z = C(1)
print(str(x).split(" at ")[0], str(y).split(" at ")[0])
print(str(z).split(" at ")[0], repr(B()).split(" at ")[0])
print(type(x).__name__, type(y).__name__)
print(erased(x), erased(y))
print(erased(1), erased("s"), erased([1]))
print(str([A()]).split(" at ")[0] + "]", str((A(), 1)).split(" at ")[0])
print(str({"k": A()}).split(" at ")[0] + "}", str({A()}).split(" at ")[0] + "}")
