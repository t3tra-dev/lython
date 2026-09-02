# What: a class attribute of an IMPORTED class, read through an instance. The
# class spelling and the instance spelling are one question, and only the class
# spelling could answer it: an imported class has no attribute cell (its
# module's body never runs to fill one), so the instance read fell past the
# cell channel into a field lookup that finds nothing. The decode is that both
# spellings are printed on the same line: a compiler answering one and not the
# other cannot print the line at all, and one answering a stale value would
# print a different word beside the same one.
import a_module_that_declares_a_hierarchy as shapes

print(shapes.Shape.kind, shapes.Shape(1).kind)
print(shapes.Shape.sides + shapes.Shape(2).sides)
s: "shapes.Shape" = shapes.Square(3)
print(s.kind, s.sides, s.describe())
