# The subclass lives in a module emitted AFTER the base's, because it imports
# it. Its class attributes are registered when its own module is emitted, so
# while the base's method is being written they do not exist yet -- and a
# dispatcher built over them read an attribute nothing could answer, reporting
# "'pets.Bird' object has no attribute 'sound'" for a class that plainly has
# one. The dispatcher now declines where an arm cannot read, and the program
# resolves the way it did before the attribute mode existed.
import a_module_of_animals as base
import a_module_of_pets as pets

zoo: list[base.Animal] = [base.Animal("x"), pets.Dog("d"), pets.Bird("b")]
for a in zoo:
    print(a.speak(), a.legs, a.home())
zoo[1].adopt("sam")
print(zoo[1].home())


def count_legs(animals: list[base.Animal]) -> int:
    return sum(a.legs for a in animals)


print(count_legs(zoo))
