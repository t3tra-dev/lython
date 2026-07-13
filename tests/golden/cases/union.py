def describe(value: int | str) -> str:
    if isinstance(value, str):
        return "str:" + value
    return "int"


print(describe("hello"))
print(describe(42))


def pick(flag: int) -> int | str | None:
    if flag <= 0:
        return None
    if flag == 1:
        return "one"
    return flag


x: int | str | None = pick(1)
if x is None:
    print("none")
elif isinstance(x, str):
    print("got str: " + x)
else:
    print("got int")


class Cat:
    def __init__(self, name: str) -> None:
        self.name = name


class Dog:
    def __init__(self, tag: int) -> None:
        self.tag = tag


def pet(flag: int) -> Cat | Dog:
    if flag <= 0:
        return Cat("tama")
    return Dog(flag)


def voice(p: Cat | Dog) -> str:
    if isinstance(p, Cat):
        return "meow:" + p.name
    return "woof"


print(voice(pet(0)))
print(voice(pet(7)))


class Animal:
    def __init__(self, name: str) -> None:
        self.name = name


class NamedCat(Animal):
    def __init__(self, name: str, lives: int) -> None:
        self.name = name
        self.lives = lives


class NamedDog(Animal):
    def __init__(self, name: str, tag: int) -> None:
        self.name = name
        self.tag = tag


def animal_name(value: NamedCat | NamedDog) -> str:
    if isinstance(value, Animal):
        return value.name
    return "unknown"


def maybe_animal(flag: int) -> Animal | None:
    if flag <= 0:
        return None
    return NamedCat("nora", 7)


def lives(value: Animal | None) -> int:
    if isinstance(value, NamedCat):
        return value.lives
    return 0


print(animal_name(NamedCat("mika", 9)))
print(animal_name(NamedDog("pochi", 11)))
print(lives(maybe_animal(1)))
print(lives(maybe_animal(0)))
