# What: a field initialised by calling a method of the class being defined --
# the class's own annotated method, an inherited one, and a staticmethod --
# keeps the type that method's annotation promised, so later reads of the field
# reach the methods of that type instead of `object`.


class Rows:
    def __init__(self, n: int) -> None:
        self.n = n
        self.values = self.build()
        self.index = self.catalogue()
        self.label = self.name_of()

    def build(self) -> list[int]:
        return [self.n, self.n * 2]

    def catalogue(self) -> dict[str, int]:
        return {"n": self.n}

    def name_of(self) -> str:
        return "rows-" + str(self.n)

    def show(self) -> str:
        return (
            str(self.values)
            + " "
            + str(sorted(self.index.items()))
            + " "
            + self.label.upper()
            + " "
            + str(len(self.values))
        )


class Factory:
    def seed(self) -> list[str]:
        return ["a", "b"]


class Derived(Factory):
    def __init__(self) -> None:
        self.items = self.seed()
        self.extra = self.spare()

    @staticmethod
    def spare() -> dict[str, str]:
        return {"k": "v"}

    def show(self) -> str:
        return ",".join(self.items) + " " + str(sorted(self.extra.items()))


print(Rows(3).show())
print(Derived().show())
