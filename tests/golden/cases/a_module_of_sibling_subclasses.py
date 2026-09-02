# What: the imported half of `sibling_subclasses_from_another_module`. It is a
# case of its own because the golden runner globs every .py here -- running it
# alone declares three classes and prints nothing, which is what its empty
# expectation says. The two subclasses each call the base-typed method the
# other overrides, which is the pair no declaration order can satisfy one class
# at a time.
class Base:
    def show(self) -> str:
        return "?"


class Left(Base):
    def __init__(self, inner: Base) -> None:
        self.inner = inner

    def show(self) -> str:
        return "L" + self.inner.show()


class Right(Base):
    def __init__(self, inner: Base) -> None:
        self.inner = inner

    def show(self) -> str:
        return "R" + self.inner.show()
