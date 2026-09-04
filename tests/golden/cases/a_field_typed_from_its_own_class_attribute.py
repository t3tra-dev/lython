# A field initialized from the class's OWN class attribute. The field walk types
# the initializer with `inferExpr` before any body is emitted, and the class's
# attributes were registered only further down -- so `Config.DEFAULTS` answered
# `object`, the FIELD became object, and its reads carried nothing:
#
#     builtins.object does not provide manifest method '__getitem__'
#
# The same line reading ANOTHER class's attribute compiled, which is what said
# the SELF reference was the defect rather than the shape.


class Config:
    DEFAULTS = {"a": 1, "b": 2}
    LIMIT = 5
    NAME = "cfg"

    def __init__(self) -> None:
        self.values = dict(Config.DEFAULTS)
        self.room = Config.LIMIT
        self.title = Config.NAME + "!"

    def get(self, k: str) -> int:
        return self.values[k]

    def put(self, k: str, v: int) -> None:
        self.values[k] = v

    def summary(self) -> str:
        return self.title + "/" + str(self.room) + "/" + str(len(self.values))


c = Config()
print(c.get("a"), c.summary())
c.put("c", 3)
print(sorted(c.values.items()), c.get("c"))
print(Config.LIMIT, Config.NAME, sorted(Config.DEFAULTS.items()))


class Sizes:
    ROWS = 2
    COLS = 3

    def __init__(self) -> None:
        self.grid = [[0] * Sizes.COLS for _ in range(Sizes.ROWS)]

    def total(self) -> int:
        return sum(sum(row) for row in self.grid)


s = Sizes()
s.grid[1][2] = 7
print(s.grid, s.total())
