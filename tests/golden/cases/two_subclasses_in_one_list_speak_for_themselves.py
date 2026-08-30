# What: the elements are two different subclasses, so the call has to reach the
# body each object's own class declares -- running it is the only way to see
# which one answered. The list has no annotation and neither does the
# conditional, which is where the type comes from.
class Animal:
    def __init__(self, name: str) -> None:
        self.name = name

    def speak(self) -> str:
        return self.name + " says nothing"


class Dog(Animal):
    def speak(self) -> str:
        return self.name + " says woof"


class Cat(Animal):
    def speak(self) -> str:
        return self.name + " says meow"


for animal in [Dog("rex"), Cat("tom"), Animal("blob")]:
    print(animal.speak())

chosen = Dog("d") if len("ab") == 2 else Cat("c")
print(chosen.speak())


def loudest(flag: int) -> str:
    pick = Cat("c") if flag > 0 else Dog("d")
    if isinstance(pick, Dog):
        return "dog:" + pick.speak()
    return "other:" + pick.speak()


print(loudest(1), loudest(-1))
