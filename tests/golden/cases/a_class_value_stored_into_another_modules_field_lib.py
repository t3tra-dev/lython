# The classes for cases/a_class_value_stored_into_another_modules_field.py.
class Base:
    pass


class Prog:
    def __init__(self, runner: type[Base] | None = None) -> None:
        self.runner = runner

    def runner_name(self) -> str:
        cls = self.runner
        if cls is None:
            return "default"
        return cls.__name__
