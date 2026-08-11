# Why execution: the failure was an abort with no output -- "repr: boxed
# element has no conforming __repr__" -- so only running the program shows
# either the crash or the string it should have printed.
#
# A subclass that declares no methods of its own had no entry in the
# boxed-method dispatch, so a container holding one found nothing to call.
# repr(Kid()) written directly was always fine: that path resolves through the
# emitter's MRO walk. The manifest's exception subclasses already had a rescue
# for the same gap; source classes did not.


class Base:
    def __repr__(self) -> str:
        return "Base()"


class Inherits(Base):
    pass


class Overrides(Base):
    def __repr__(self) -> str:
        return "Overrides()"


class TwoLevels(Inherits):
    pass


class Stringy:
    def __str__(self) -> str:
        return "stringy"


class InheritsStr(Stringy):
    pass


def main() -> None:
    print([Inherits()])
    print([Overrides()])
    print([TwoLevels()])
    print([Base(), Inherits(), Overrides()])
    print(repr(Inherits()))
    print(InheritsStr())


main()
