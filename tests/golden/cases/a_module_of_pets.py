# Helper for a_class_attribute_a_subclass_in_a_third_module_redeclares.
import a_module_of_animals as base


class Dog(base.Animal):
    sound = "woof"


class Bird(base.Animal):
    legs = 2
    sound = "tweet"

    def speak(self) -> str:
        return super().speak() + "!"
