# What: the keywords in a class header are `__init_subclass__`'s arguments.
# They were dropped: the class compiled, ran, and printed nothing of what the
# hook prints -- so the only evidence is the output, and the output is what a
# registry built this way depends on.
#
# The four lines cover what the header can say and the hook can declare: two
# keywords at once, a keyword-only parameter, a class that names none (its
# defaults apply), and a middle class that passes one to its PARENT's hook
# while declaring a hook of its own.
class Registry:
    @classmethod
    def __init_subclass__(cls, tag: str = "none", weight: int = 0) -> None:
        print("registry", cls.__name__, tag, weight)


class Tagged(Registry, tag="t", weight=2):
    pass


class Bare(Registry):
    pass


class Strict:
    @classmethod
    def __init_subclass__(cls, *, mode: str = "off") -> None:
        print("strict", cls.__name__, mode)


class Middle(Strict, mode="on"):
    @classmethod
    def __init_subclass__(cls, *, mode: str = "off") -> None:
        print("middle", cls.__name__, mode)


class Leaf(Middle, mode="deep"):
    pass


print("done")
