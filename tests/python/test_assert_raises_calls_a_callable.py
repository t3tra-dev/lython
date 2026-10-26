# assertRaises, assertRaisesRegex, assertWarns and assertWarnsRegex given a
# callable call it with the rest of the arguments and return None, as
# CPython's do; given none they return the context manager.
import unittest
import warnings


def divide(a: int, b: int) -> int:
    return a // b


def warn(text: str) -> None:
    warnings.warn(text, UserWarning)


class Box:
    def pop(self, key: str) -> int:
        raise KeyError(key)


def check(text: str, msg: str = "") -> int:
    if msg == "bad":
        raise ValueError(text)
    return 0


def warn_if(msg: str = "") -> None:
    if msg == "w":
        warnings.warn("careful", UserWarning)


class CallableFormTest(unittest.TestCase):

    def test_assert_raises_calls_the_callable(self) -> None:
        self.assertIsNone(self.assertRaises(ZeroDivisionError, divide, 1, 0))
        self.assertRaises(ZeroDivisionError, divide, 1, b=0)
        self.assertRaises((KeyError, ZeroDivisionError), divide, 1, 0)
        self.assertRaises(KeyError, Box().pop, "k")
        self.assertRaises(ValueError, int, "x")
        self.assertRaises(ValueError, chr, -1)
        self.assertRaises(ZeroDivisionError, lambda: divide(2, 0))

    def test_not_raised_names_the_callable(self) -> None:
        with self.assertRaises(AssertionError) as cm:
            self.assertRaises(ZeroDivisionError, divide, 4, 2)
        self.assertEqual(str(cm.exception), "ZeroDivisionError not raised by divide")

    def test_context_form_still_works(self) -> None:
        with self.assertRaises(ZeroDivisionError) as cm:
            divide(1, 0)
        self.assertIsInstance(cm.exception, ZeroDivisionError)
        with self.assertRaises(AssertionError) as outer:
            with self.assertRaises(KeyError, msg="custom"):
                pass
        self.assertEqual(str(outer.exception), "KeyError not raised : custom")

    def test_regex_callable_form(self) -> None:
        self.assertRaisesRegex(ValueError, "invalid literal", int, "x")
        with self.assertRaises(AssertionError) as cm:
            self.assertRaisesRegex(ValueError, "nope", int, "x")
        self.assertEqual(str(cm.exception),
                         "\"nope\" does not match \"invalid literal for int() with base 10: 'x'\"")
        with self.assertRaisesRegex(KeyError, "k"):
            Box().pop("k")

    def test_warns_callable_form(self) -> None:
        self.assertIsNone(self.assertWarns(UserWarning, warn, "w"))
        self.assertWarnsRegex(UserWarning, "^w", warn, "wx")
        with self.assertRaises(AssertionError) as cm:
            self.assertWarns(UserWarning, divide, 4, 2)
        self.assertEqual(str(cm.exception), "UserWarning not triggered by divide")
        with self.assertWarns(UserWarning) as wcm:
            warn("ctx")
        self.assertEqual(str(wcm.warning), "ctx")

    def test_msg_is_passed_to_the_callable(self) -> None:
        self.assertRaises(ValueError, check, "x", msg="bad")
        self.assertRaisesRegex(ValueError, "^x$", check, "x", msg="bad")
        self.assertWarns(UserWarning, warn_if, msg="w")
        self.assertWarnsRegex(UserWarning, "care", warn_if, msg="w")
        with self.assertRaises(AssertionError) as cm:
            self.assertRaises(ValueError, check, "x", msg="fine")
        self.assertEqual(str(cm.exception), "ValueError not raised by check")

    def test_msg_of_the_context_form_is_the_assertion_message(self) -> None:
        with self.assertRaises(AssertionError) as raised:
            with self.assertRaisesRegex(ValueError, "a", msg="m3"):
                pass
        self.assertEqual(str(raised.exception), "ValueError not raised : m3")
        with self.assertRaises(AssertionError) as warned:
            with self.assertWarns(UserWarning, msg="w2"):
                pass
        self.assertEqual(str(warned.exception), "UserWarning not triggered : w2")
        with self.assertRaises(AssertionError) as warned_regex:
            with self.assertWarnsRegex(UserWarning, "a", msg="m4"):
                pass
        self.assertEqual(str(warned_regex.exception), "UserWarning not triggered : m4")


if __name__ == "__main__":
    unittest.main()
