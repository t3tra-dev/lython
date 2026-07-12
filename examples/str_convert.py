class Money:
    def __init__(self, amount: int) -> None:
        self.amount = amount

    def __str__(self) -> str:
        return "$" + str(self.amount)


def risky(n: int) -> str:
    try:
        if n < 0:
            raise ValueError("bad value")
    except ValueError as err:
        return str(err)
    return str(n)


print(str(42))
print(str(3.5))
print(str("already"))
print(str(True))
print(str(-7) + "!")
print(len(str(1200)))
print(str(Money(25)))
print(risky(-1))
print(risky(9))
