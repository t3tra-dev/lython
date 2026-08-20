# Which body a method call runs was decided from the STATIC receiver type, so
# an override behind a base-typed reference was refused outright ("'name' is
# overridden by a subclass of 'A', so this call cannot be resolved from the
# static type of the receiver") -- and that took every base-typed collection,
# parameter and declared binding with it. Must run: the refusal is what
# regresses, but only the printed values say the RIGHT body ran, and a
# dispatcher that always picked the base would compile just as quietly.


class Shape:
    def name(self) -> str:
        return "shape"

    def sides(self) -> int:
        return 0

    def label(self, prefix: str) -> str:
        return prefix + self.name()


class Rect(Shape):
    def name(self) -> str:
        return "rect"

    def sides(self) -> int:
        return 4


class Square(Rect):
    def name(self) -> str:
        return "square"


class Circle(Shape):
    def name(self) -> str:
        return "circle"


# A base-typed collection, the shape that made this unavoidable.
shapes: list[Shape] = [Shape(), Rect(), Square(), Circle()]
for s in shapes:
    print(s.name(), s.sides())

# A base-typed parameter.
def describe(s: Shape) -> str:
    return s.name() + "/" + str(s.sides())


print(describe(Square()), describe(Shape()), describe(Circle()))

# A base-typed binding of a subclass value.
one: Shape = Rect()
print(one.name(), one.sides())

# A method with a parameter, dispatched the same way.
print(describe(Rect()), Rect().label("a "), describe(Square()))
labels: list[str] = []
for s in shapes:
    labels.append(s.label("<"))
print(labels)

# Most-derived first: Square overrides name() but inherits Rect's sides().
sq: Shape = Square()
print(sq.name(), sq.sides())

# A base method calling a virtual one on ITSELF, reached through the base type.
holder: Shape = Circle()
print(holder.label("["))
