# WHAT: a class whose `__init__` annotates a field with a subclass of the
# class itself -- the child a node points to, unittest's current subtest --
# and the subclass, which inherits every field, stored in it and read back.
#
# WHY THIS IS RUN: the field holds a subclass instance at run time. Naming
# the subclass in the base's own body emitted the subclass first, before the
# base had any fields, and it had none ("class Sub has no field 'name'").
class Base:
    def __init__(self, name: str = "x") -> None:
        self.name = name
        self.sub: Sub | None = None

    def attach(self, name: str) -> "Sub":
        child = Sub(name)
        self.sub = child
        return child


class Sub(Base):
    def describe(self) -> str:
        return "sub " + self.name


b = Base()
c = b.attach("s")
c.attach("t")
print(b.name, b.sub.describe() if b.sub is not None else "-")
print(c.sub.name if c.sub is not None else "-", Sub("u").sub is None)
