# What: a stored function reads back as the one runtime function contract, so
# only calling the value a lookup returns shows whether the call reached the
# body the key names -- every one here answers differently.
def upper() -> str:
    return "UP"


def lower() -> str:
    return "down"


table = {"u": upper, "l": lower}
print(table["u"](), table["l"]())

for key in sorted(table):
    print(key, table[key]())


def scale(n: int) -> int:
    return n * 3


def shift(n: int) -> int:
    return n + 3


arith = {"scale": scale, "shift": shift}
print(arith["scale"](4), arith["shift"](4))

nested = {"outer": {"inner": upper}}
print(nested["outer"]["inner"]())
