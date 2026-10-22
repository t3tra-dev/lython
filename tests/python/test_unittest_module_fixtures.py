# WHAT: a module's setUpModule runs once before its first test and before
# any setUpClass, a module cleanup registered there runs after the module's
# tests, and tearDownModule has not run while the module's tests do -- as
# CPython's unittest does for the module unittest.main() runs.
import unittest

events: list[str] = []


def setUpModule() -> None:
    events.append("setUpModule")
    unittest.addModuleCleanup(events.append, "moduleCleanup")


def tearDownModule() -> None:
    events.append("tearDownModule")


class First(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        events.append("setUpClass")

    def test_module_was_set_up_first(self) -> None:
        self.assertEqual(events[:2], ["setUpModule", "setUpClass"])


class Second(unittest.TestCase):
    def test_module_set_up_once(self) -> None:
        self.assertEqual(events.count("setUpModule"), 1)
        self.assertNotIn("tearDownModule", events)
        self.assertNotIn("moduleCleanup", events)


if __name__ == '__main__':
    unittest.main()
