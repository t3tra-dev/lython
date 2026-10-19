# WHAT: `self.failureException(msg)` constructs through a
#   `type[BaseException]` class value. The dispatch type-checks the
#   constructor of EVERY exception class of the program, so an unrelated
#   `Carrier(payload: list[int])` makes the call refused ("argument
#   'payload' ... gives it str"). CPython reaches Carrier's __init__ only
#   if that class is the one actually called.
# KIND: バグ (bug) -- loud: well-typed code refused
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# PARTLY FIXED: 2026-10-20 -- Carrier's arm no longer refuses the call; it raises
#   if Carrier is the class called (wb_a_class_value_call_whose_argument_the_init
#   _cannot_hold). What still stops this program is
#   ut_bug_class_value_class_attribute_through_instance.
# CLASSIFICATION @ lang-gaps c8463f9d + the fix: 3 loud 拒否 (診断)
#   class Checker has no field 'failureException'
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
