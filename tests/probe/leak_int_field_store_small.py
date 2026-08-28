# WHAT: an int field written in a loop. The field is box-fronted, so each store
# puts a fresh heap int in the slot and has to release the one it replaces; the
# values are large enough that each box owns a multi-limb payload past the
# probe floor.
class Cell:
    v: int

    def __init__(self) -> None:
        self.v = 0


c = Cell()
i = 0
base = 10 ** 400
while i < 400:
    c.v = base + i
    i += 1
print(c.v % 1000)
