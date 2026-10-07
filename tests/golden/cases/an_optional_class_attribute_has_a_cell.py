# WHAT: a class attribute annotated with a union gets the cell a plain one
# has: it reads back None and its value, a write through the class or `cls`
# reaches it, a guard or a store narrows it, an empty list written into a
# `list[str] | None` attribute is a list of str, and `+=` on a proved
# attribute adds.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the attribute used to be
# read as a constant with no union tag and its writes were refused at the
# lowering; what is shown is the values the cell holds across writes, and an
# empty list stored into it is read back as the list it is.

class Config:
    debug: bool | None = None
    name: str | None = "cfg"
    cache: dict[str, int] | None = None
    @classmethod
    def get(cls) -> dict[str, int]:
        if cls.cache is None:
            cls.cache = {"k": 1}
        return cls.cache
print(Config.debug, Config.name)
Config.debug = True
print(Config.debug)
if Config.name is not None:
    print(Config.name.upper())
print(Config.get(), Config.get())
def f() -> None:
    Config.name = None
    print(Config.name)
f()
print(Config().name)
class Reg2:
    items: list[str] | None = None
    last: "Reg2 | None" = None
    count: int | None = None
    def __init__(self, n: str) -> None:
        self.n = n
def register(s: str) -> None:
    if Reg2.items is None:
        Reg2.items = []
    Reg2.items = Reg2.items + [s]
    Reg2.last = Reg2(s)
    if Reg2.count is None:
        Reg2.count = 0
    Reg2.count += 1
for i in range(50):
    register("x" + str(i))
print(len(Reg2.items) if Reg2.items is not None else 0, Reg2.count)
if Reg2.last is not None:
    print(Reg2.last.n)
Reg2.items = None
Reg2.last = None
print(Reg2.items, Reg2.last)
