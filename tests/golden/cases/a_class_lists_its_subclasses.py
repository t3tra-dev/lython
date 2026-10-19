# `C.__subclasses__()` lists the program's classes that name C as a base, in
# the order they were created -- an imported module's before this one's --
# whether C is the class itself or a class VALUE that only the run time
# knows. It was refused ("type[Base] does not provide manifest method
# '__subclasses__'"). The whole program's classes are known when it is
# compiled, so the answer is: unittest's `main()` finds its test classes
# this way.
# Why run it: a class value's answer depends on the class it holds, which a
# recursive walk asks about one level at a time.
import a_class_lists_its_subclasses_lib as lib


class Base:
    pass


class A(Base):
    pass


class B(Base):
    pass


class C(A):
    pass


class D(A):
    pass


class E(C):
    pass


class Outer(lib.Root):
    pass


class Third(lib.Inner):
    pass


def every(cls: type[Base]) -> list[type[Base]]:
    out: list[type[Base]] = []
    for sub in cls.__subclasses__():
        out.append(sub)
        out.extend(every(sub))
    return out


print([c.__name__ for c in Base.__subclasses__()], [c.__name__ for c in A.__subclasses__()])
print([c.__name__ for c in every(Base)])
print([c.__name__ for c in every(A)], [c.__name__ for c in every(E)])
held: type[Base] = A
print([c.__name__ for c in held.__subclasses__()], B.__subclasses__() == [])
print([c.__name__ for c in lib.Root.__subclasses__()], [c.__module__ for c in lib.Root.__subclasses__()])
print([c.__name__ for c in lib.Inner.__subclasses__()])
