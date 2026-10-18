# WHAT: a method whose parameter is declared `object` tests its truth as an
# object's -- `assertTrue(self, expr: object)` with an int, a float (0.0,
# -0.0, nan), a str and None -- the same answer a free function with that
# parameter gives, while an `int` parameter keeps Lython's rule that its truth
# is written as a comparison.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the answers are the truth
# of run-time values. An inlined method binds the argument as it came, so
# `self.assertTrue(3)` saw an int and was refused ("implicit truthiness of int
# is rejected"), where the method not inlined answered.
class TestCase:
    def assertTrue(self, expr: object, msg: str = "") -> str:
        if not expr:
            return "FAIL " + repr(expr) + " is not true"
        return "ok"

    def assertFalse(self, expr: object) -> str:
        if expr:
            return "FAIL " + repr(expr) + " is not false"
        return "ok"


def truth(expr: object) -> bool:
    return not not expr


t = TestCase()
nan = float("nan")
print(t.assertTrue(3), t.assertTrue(0), t.assertTrue(2.5), t.assertTrue(0.0))
print(t.assertFalse(0), t.assertFalse(-0.0), t.assertFalse(nan),
      t.assertFalse(""), t.assertFalse(None), t.assertTrue("x"))
print(truth(3), truth(0), truth(0.0), truth(nan))
