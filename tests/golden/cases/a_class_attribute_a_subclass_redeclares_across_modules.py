# The class-attribute dispatcher, over an IMPORTED hierarchy. Its arms used to
# read through the narrowed receiver, which needs the storage cell only a
# main-module class gets -- so every imported base method reading `self.kind`
# was refused, which is where a library puts one. Each arm reads through the
# CLASS instead: inside the arm the runtime class IS that candidate, and the
# read is available on both channels.
import a_module_of_shadowing_shapes as shapes

items: list[shapes.Shape] = [shapes.Shape(), shapes.Square(), shapes.Tilted()]
for item in items:
    print(item.describe())
print(shapes.Shape.kind, shapes.Square.kind, shapes.Tilted.kind)


def widen(s: shapes.Shape) -> str:
    if isinstance(s, shapes.Square):
        return "sq" + str(s.side)
    return "other"


print([widen(i) for i in items])
