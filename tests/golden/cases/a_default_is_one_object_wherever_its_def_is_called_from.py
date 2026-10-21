# What: a non-literal default is evaluated once, when its def runs, and every
# call that leaves the argument out shares that one object -- for a function
# and a generic function of an imported module, for a method of an imported
# class (called directly, through a function, and through a subclass), and for
# a method of this module called from a function body emitted before its class
# statement.
# Why run: whether two calls share the list is what the lengths print.
import a_default_is_one_object_wherever_its_def_is_called_from_lib as lib


class C:
    def add(self, into: list[int] = []) -> int:
        into.append(1)
        return len(into)


class Sub(lib.B):
    pass


def through(x: C) -> int:
    return x.add()


def through_lib(b: lib.B) -> int:
    return b.add()


c = C()
print(c.add(), c.add(), through(c), through(c))
print(lib.f(), lib.f(), lib.g(1), lib.g("a"), lib.g(2))
b = lib.B()
s = Sub()
print(b.add(), through_lib(b), through_lib(s), s.add())
