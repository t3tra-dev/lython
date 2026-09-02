# What: a class hierarchy that lives in an IMPORTED module. Every question the
# compiler asks about a hierarchy -- does a subclass override this, is this
# value an instance of that subclass -- is asked by contract NAME, and an
# imported class's name is dotted like a builtin's. The decode is that each
# answer here is asked through a BASE-typed reference: a compiler that resolved
# the base's body would print the same shape of output with the wrong words in
# it.
import a_module_that_declares_a_hierarchy as shapes


def describe(s: "shapes.Shape") -> str:
    return s.describe()


print(describe(shapes.Shape(1)), describe(shapes.Square(2)))
print(describe(shapes.Circle(3)), describe(shapes.Marked(4)))

everything: "list[shapes.Shape]" = [
    shapes.Shape(1),
    shapes.Square(2),
    shapes.Circle(3),
    shapes.Marked(4),
]
print([s.name() for s in everything])
print([s.describe() for s in everything])


def kind(s: "shapes.Shape") -> str:
    if isinstance(s, shapes.Marked):
        return "marked"
    if isinstance(s, shapes.Square):
        return "square"
    return "other"


print([kind(s) for s in everything])

