# WHAT: `hasattr(cls, name)` on a class value whose class has subclasses,
# for a name no class of the line declares, one only a subclass declares,
# and one the class itself declares, asked of the class and of each
# subclass.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the first was refused
# whenever the class had any subclass ("may define that attribute"); the
# second is answered by the class the value has at run time, which only the
# printed answers show.
class Base:
    def shared(self) -> int:
        return 1


class Leaf(Base):
    pass


class Runner(Base):
    def runTest(self) -> None:
        pass


def probe(cls: type[Base]) -> str:
    return "%s %s %s %s" % (cls.__name__, hasattr(cls, "missing"),
                            hasattr(cls, "runTest"), hasattr(cls, "shared"))


for cls in [Base, Leaf, Runner]:
    print(probe(cls))
