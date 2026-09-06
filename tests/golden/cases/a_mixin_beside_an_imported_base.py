# What: which class's attribute a subclass reads when one of its bases is in
# another file. Only a main-module class gets slot storage -- an imported one
# keeps its attributes on the constant channel -- and the MRO walk that looks
# for a slot SKIPPED a class that had none, so it handed back a class further
# down the MRO. `class BaseFirst(Shape, Loud)` read Loud's `kind` where Python
# reads Shape's, silently, and the same three classes in one file were right.
#
# Both base orders are here because which one comes first is the whole answer,
# and the method beside the attribute is what shows the two are resolved the
# same way.
from a_module_of_importable_shapes import Shape


class Loud:
    kind: str = "loud"

    def name(self) -> str:
        return "loud"


class MixinFirst(Loud, Shape):
    pass


class BaseFirst(Shape, Loud):
    pass


items: list[Shape] = [Shape(1), MixinFirst(2), BaseFirst(3)]
print("attribute", [i.kind for i in items])
print("method", [i.name() for i in items])
print("through the base", [i.describe() for i in items])
