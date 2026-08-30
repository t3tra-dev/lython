# What: `__init_subclass__` runs at class DEFINITION, so the output order is
# the only evidence it ran at all -- and the `cls` it receives is the new
# class, which the registry below records by name. Nothing about either is
# visible without running it.
class Registry:
    seen: list[str] = []

    @classmethod
    def __init_subclass__(cls) -> None:
        Registry.seen.append(cls.__name__)


print("before", Registry.seen)


class Alpha(Registry):
    pass


class Beta(Registry):
    pass


class Gamma(Alpha):
    pass


print("after", Registry.seen)


class Plain:
    pass


class Quiet(Plain):
    pass


print("no hook", Registry.seen)
