# WHAT: `self.failureException(msg)` constructs through a
#   `type[BaseException]` class value. The dispatch type-checks the
#   constructor of EVERY exception class of the program, so an unrelated
#   `Carrier(payload: list[int])` makes the call refused ("argument
#   'payload' ... gives it str"). CPython raises a TypeError only if that
#   class is the one actually called.
# KIND: バグ (bug) -- loud: well-typed code refused
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   argument 'payload' of '__init__' is declared !py.contract<"builtins.list", [!py.contract<"builtins.int">]> and this call gives it !py.contract<"builtins.str">
# CPython 3.14 expects: AssertionError: boom / [1]
class Carrier(Exception):
    def __init__(self, payload: list[int]) -> None:
        super().__init__()
        self.payload = payload


class Checker:
    failureException: type[BaseException] = AssertionError

    def fail(self, msg: str) -> None:
        raise self.failureException(msg)


try:
    Checker().fail("boom")
except AssertionError as e:
    print("AssertionError:", e)
print(Carrier([1]).payload)
