# OPEN. The class-attribute dispatcher does not reach an IMPORTED class:
#
#     # a_module_of_shadowing_subclasses.py
#     class Shape:
#         kind = "shape"
#         def describe(self) -> str: return self.kind
#     class Square(Shape):
#         kind = "square"
#
#     import a_module_of_shadowing_subclasses as m
#     shapes: list[m.Shape] = [m.Shape(), m.Square()]
#     for shape in shapes: print(shape.describe())
#     # 'kind' is overridden by a subclass of '...Shape', so this call cannot
#     # be resolved from the static type of the receiver
#
# The one-file spelling of exactly this IS fixed
# (cases/a_class_attribute_a_subclass_redeclares), so the boundary is the
# defect and not the shape.
#
# ⛔ WHY THE SAME REPAIR DOES NOT REACH. The dispatcher reads the attribute
# through the narrowed receiver, and that read needs the attribute to have
# STORAGE -- a cell named `<Class>.<attr>`. `classAttrSlots` is filled only for
# main-module classes (`if (symbolName.empty())` in EmitterClasses.cpp), so an
# imported class attribute lives on the constant channel, which
# re-materializes a value per read and therefore has one value per CLASS, not
# per receiver. `resolveClassAttrSlot` answering nothing is what declines the
# dispatcher, and giving imported classes cells is the mechanism this needs --
# not a change to the dispatcher.
#
# Measured 2026-09-03: refused at the `self.kind` inside the imported module's
# own method, so the import fails before the main module runs. Only the
# BASE-TYPED receiver reaches it -- `m.Square().describe()` names its class and
# compiles, and so does reading the attribute through the class
# (`m.Square.kind`), which was never in doubt.
import a_module_of_shadowing_subclasses as m

shapes: list[m.Shape] = [m.Shape(), m.Square()]
for shape in shapes:
    print(shape.describe())
