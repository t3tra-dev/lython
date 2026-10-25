# A bool stored where a module-level name or container declares int is kept
# as the bool: the name's cell and the container's element hold its object, so
# a read from anywhere sees True with its own type while adding as 1, and an
# int stored back makes the name an int again. A value that is a bool or an int
# (a read of `{1: 5, 4: True}`) is stored as whichever it holds.
import unittest

FLAG: int = 0
FLAG = True
ANNOTATED: int = True
COUNT: int = 3
TABLE = {1: 5, 4: True}
CHOSEN: int = 0
CHOSEN = TABLE[4]
OTHER: int = 0
OTHER = TABLE[1]
LIST: list[int] = [1, 2]
LIST[0] = True
LIST.append(False)
BY_NAME: dict[str, int] = {"a": 1}
BY_NAME["b"] = True


def set_count(flag: bool) -> None:
    global COUNT
    COUNT = flag


def set_count_big() -> None:
    global COUNT
    COUNT = 2 ** 70


def facts(v: int) -> tuple[str, bool, bool, str, int]:
    return (type(v).__name__, isinstance(v, bool), v is True, repr(int(v)),
            v + 1)


class ModuleNameTest(unittest.TestCase):

    def test_a_store_at_module_scope(self) -> None:
        self.assertEqual(facts(FLAG), ("bool", True, True, "1", 2))
        self.assertEqual(facts(ANNOTATED), ("bool", True, True, "1", 2))
        self.assertEqual((repr(FLAG), str(FLAG), f"{FLAG}"),
                         ("True", "True", "True"))

    def test_a_store_from_a_function(self) -> None:
        set_count(False)
        self.assertEqual((repr(COUNT), type(COUNT).__name__, COUNT + 1),
                         ("False", "bool", 1))
        set_count(True)
        self.assertEqual(facts(COUNT), ("bool", True, True, "1", 2))

    def test_an_int_stored_back(self) -> None:
        set_count(True)
        set_count_big()
        self.assertEqual((type(COUNT).__name__, COUNT - 2 ** 70), ("int", 0))

    def test_a_bool_or_an_int(self) -> None:
        self.assertEqual(facts(CHOSEN), ("bool", True, True, "1", 2))
        self.assertEqual(facts(OTHER), ("int", False, False, "5", 6))


class ModuleContainerTest(unittest.TestCase):

    def test_a_list_element(self) -> None:
        self.assertEqual(repr(LIST), "[True, 2, False]")
        self.assertEqual([type(v).__name__ for v in LIST],
                         ["bool", "int", "bool"])
        self.assertEqual((LIST[0] + 1, sum(LIST)), (2, 3))

    def test_a_dict_value(self) -> None:
        self.assertEqual(repr(BY_NAME), "{'a': 1, 'b': True}")
        self.assertEqual(facts(BY_NAME["b"]), ("bool", True, True, "1", 2))


if __name__ == "__main__":
    unittest.main()
