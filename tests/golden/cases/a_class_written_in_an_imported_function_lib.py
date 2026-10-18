class Base:
    def tag(self) -> str:
        return "base"


def make() -> Base:
    class Impl(Base):
        def tag(self) -> str:
            return "impl"

    return Impl()
