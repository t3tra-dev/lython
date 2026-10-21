# The classes for cases/a_class_value_of_another_module_called_with_keywords.py.
# Imported modules' bodies do not run, so the `assert` on the constants is
# decided when the module is compiled.
_WIDTH = 80
_MARGIN = _WIDTH - (5 + 12 + 5 + 12 + 5)
assert _MARGIN >= 0 and _WIDTH // 8 == 10


class Runner:
    def __init__(self, verbosity: int = 1, *, durations: int | None = None) -> None:
        self.verbosity = verbosity
        self.durations = durations


def margin() -> int:
    return _MARGIN
