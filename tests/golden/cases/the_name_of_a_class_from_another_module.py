# What: the class NAME a program reads back for a class that lives in ANOTHER
# module. CPython answers two different strings from one class: `__name__` and
# `repr(exception)` are the LEAF, while the default repr of a plain object
# carries the module. The decode is that the module half and the leaf half are
# printed apart: a compiler that answered the qualified name everywhere would
# print a dotted `__name__`, and one that answered the leaf everywhere would
# print an empty module.
import a_module_that_declares_a_hierarchy as shapes


def caught() -> str:
    try:
        raise shapes.Fault("boom")
    except shapes.Trouble as e:
        return type(e).__name__ + " " + repr(e)


print(caught())
head = repr(shapes.Square(2)).split(" object at ")[0]
print(head.rsplit(".", 1)[0], "|", head.rsplit(".", 1)[1])
s: "shapes.Shape" = shapes.Marked(3)
print(type(s).__name__)
