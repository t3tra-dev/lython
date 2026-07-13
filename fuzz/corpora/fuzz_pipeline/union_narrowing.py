class Cat:
    def __init__(self, name: str) -> None:
        self.name = name


class Dog:
    def __init__(self, tag: int) -> None:
        self.tag = tag


def voice(p: Cat | Dog | None) -> str:
    if p is None:
        return "silent"
    if isinstance(p, Cat):
        return "meow:" + p.name
    if isinstance(p, Dog):
        return "woof:" + str(p.tag)
    return "?"


print(voice(Cat("tama")))
print(voice(Dog(7)))
print(voice(None))
