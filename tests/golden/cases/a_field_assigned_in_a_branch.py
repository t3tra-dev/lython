# What: a field assigned only inside an if, a loop, a try or a with is still a
# field of the class, and the value each construction ends up holding is the
# only thing that shows which branch's assignment reached the slot.
class Mode:
    def __init__(self, flag: bool) -> None:
        if flag:
            self.n = 1
            self.tag = "on"
        else:
            self.n = 2
            self.tag = "off"


print(Mode(True).n, Mode(True).tag, Mode(False).n, Mode(False).tag)


class Counted:
    def __init__(self, upto: int) -> None:
        for i in range(upto):
            self.last = i


print(Counted(3).last)


class Tried:
    def __init__(self, text: str) -> None:
        try:
            self.value = int(text)
        except ValueError:
            self.value = -1


print(Tried("7").value, Tried("x").value)


class Optionally:
    def __init__(self, xs: "list[int] | None" = None) -> None:
        if xs is None:
            self.xs = []
        else:
            self.xs = xs


print(Optionally().xs, Optionally([1, 2]).xs)


class Deep:
    def __init__(self, n: int) -> None:
        if n > 0:
            if n > 5:
                self.size = "big"
            else:
                self.size = "small"
        else:
            self.size = "none"


print(Deep(9).size, Deep(1).size, Deep(0).size)
