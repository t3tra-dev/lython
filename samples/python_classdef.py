class Person:
    def __init__(self, name: str) -> None:
        self.name = name
        self.age = 10

    def print_name(self) -> None:
        print(self.name)


person1 = Person("Alice")


print(person1.name)
print(person1.age)
person1.print_name()
