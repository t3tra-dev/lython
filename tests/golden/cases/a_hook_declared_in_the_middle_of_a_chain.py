# What: a class that declares `__init_subclass__` of its own still gets its
# PARENT's hook run for it, and the `cls` that hook receives is the declaring
# class -- not the parent. The order of the three lines is the only evidence
# the parent's body ran at the right moment, and `cls.label()` is the only
# evidence of which class `cls` names: it answers the middle class's override
# while the parent's hook is running.
class Base:
    @classmethod
    def label(cls) -> str:
        return "base"

    @classmethod
    def __init_subclass__(cls) -> None:
        print("base hook for", cls.__name__, cls.label())


class Middle(Base):
    @classmethod
    def label(cls) -> str:
        return "middle"

    @classmethod
    def __init_subclass__(cls) -> None:
        print("middle hook for", cls.__name__, cls.label())


class Leaf(Middle):
    @classmethod
    def label(cls) -> str:
        return "leaf"


class Plain(Base):
    pass


print("done")
