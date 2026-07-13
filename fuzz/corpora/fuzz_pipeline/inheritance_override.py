class Animal:
    def __init__(self, name: str) -> None:
        self.name = name

    def sound(self) -> str:
        return "..."


class Bird(Animal):
    def __init__(self, name: str, pitch: int) -> None:
        self.name = name
        self.pitch = pitch

    def sound(self) -> str:
        return "tweet" + str(self.pitch)


print(Animal("x").sound())
print(Bird("y", 3).sound())
