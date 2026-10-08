# WHAT: an instance held as `object` answers repr, str and print as the class
#   it is, not the class it was made as: a `def mk() -> A` that returns a B
#   prints B's repr, and a class's own __repr__ is not passed over for the
#   default one.
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: which __repr__ answers is
#   decided by the class the instance points at, at run time.
class A:
    def __repr__(self) -> str:
        return "A()"


class B(A):
    def __repr__(self) -> str:
        return "B()"


class Plain:
    pass


def mk(flag: bool) -> A:
    if flag:
        return B()
    return A()


def main() -> None:
    x: object = mk(True)
    print(repr(x))
    print(x)
    print(str(x))
    z: object = A()
    print(repr(z), str(z))
    if isinstance(x, B):
        print("B", repr(x))
    p: object = Plain()
    print(repr(p).startswith("<__main__.Plain object at 0x"))


main()
