# Helper for a_class_attribute_a_subclass_in_a_third_module_redeclares.
class Animal:
    legs = 4
    sound = "..."

    def __init__(self, name: str) -> None:
        self.name = name
        self.owner = None

    def speak(self) -> str:
        return self.name + " says " + self.sound

    def adopt(self, who: str) -> None:
        self.owner = who

    def home(self) -> str:
        if self.owner is None:
            return "stray"
        return self.owner.upper()
