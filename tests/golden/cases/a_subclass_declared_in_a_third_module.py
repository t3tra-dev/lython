# What: a class hierarchy whose base and subclass live in DIFFERENT imported
# modules. What a class derives from is recorded by NAME before anything is
# emitted, and a base that lives in another module is spelled either dotted
# (`shapes.Shape`) or from-imported (`Square`) -- the first was skipped and the
# second was qualified with the importing module, so both subclasses were
# recorded as deriving from nothing. The decode is that every answer here is
# asked through a reference typed as the BASE, and both spellings appear: a
# compiler that resolved the base's body would print the same shape of output
# with the base's words in it, and `isinstance` would answer False for a value
# that is one.
import a_module_that_declares_a_hierarchy as shapes
import a_module_that_extends_another_module as more

everything: "list[shapes.Shape]" = [
    shapes.Shape(1),
    shapes.Square(2),
    more.Rounded(3),
    more.Sharp(4),
]
print([s.name() for s in everything])
print([s.describe() for s in everything])
print([isinstance(s, more.Rounded) for s in everything])
print([isinstance(s, shapes.Square) for s in everything])
