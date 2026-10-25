# A bool kept where an int is declared -- an int field, an element of a
# list[int] or a tuple[int, ...], a value of a dict[str, int] -- is still the
# bool: its type is bool, isinstance(v, bool) holds, it is True (the
# singleton), int(v) is the plain int of its value, and it prints as True
# while adding as 1 -- after a function returns it, too. Storing an int back
# makes the slot an int again.
import unittest


class Counter:
    def __init__(self, n: int) -> None:
        self.n = n

    def flag(self) -> None:
        self.n = True


class Annotated:
    def __init__(self, n: int) -> None:
        self.n: int = n


def described(x: int) -> str:
    return (repr(x) + ":" + type(x).__name__ + ":" +
            str(isinstance(x, bool)) + ":" + str(x is True))


def ident(x: int) -> int:
    return x


def facts(v: int) -> tuple[str, bool, bool, bool, str, int]:
    return (type(v).__name__, isinstance(v, bool), v is True, v is False,
            repr(int(v)), v + 1)


class FieldTest(unittest.TestCase):

    def test_a_store_from_outside(self) -> None:
        c = Counter(3)
        c.n = True
        self.assertEqual(facts(c.n), ("bool", True, True, False, "1", 2))
        self.assertEqual((repr(c.n), str(c.n), f"{c.n}"),
                         ("True", "True", "True"))

    def test_a_store_from_a_method(self) -> None:
        c = Counter(3)
        c.flag()
        self.assertEqual(described(c.n), "True:bool:True:True")

    def test_an_annotated_field(self) -> None:
        a = Annotated(4)
        a.n = False
        self.assertEqual(facts(a.n), ("bool", True, False, True, "0", 1))

    def test_an_int_stored_back(self) -> None:
        c = Counter(3)
        c.n = True
        c.n = 9
        self.assertEqual(facts(c.n), ("int", False, False, False, "9", 10))

    def test_choices_and_containers(self) -> None:
        c = Counter(3)
        c.n = True
        self.assertEqual((repr(max(c.n, 0)), repr(min(c.n, 0))), ("True", "0"))
        ys: list[int] = [7]
        ys.append(c.n)
        self.assertEqual(repr(ys), "[7, True]")
        self.assertEqual(type(ys[1]).__name__, "bool")


class ContainerTest(unittest.TestCase):

    def test_a_list_element(self) -> None:
        xs: list[int] = [5]
        xs.append(True)
        self.assertEqual(facts(xs[1]), ("bool", True, True, False, "1", 2))
        self.assertEqual(facts(xs[0]), ("int", False, False, False, "5", 6))
        self.assertEqual((repr(max(xs[1], 0)), repr(min(xs[1], 7))),
                         ("True", "True"))

    def test_a_list_given_to_a_function(self) -> None:
        def kinds(xs: list[int]) -> list[str]:
            return [type(v).__name__ for v in xs]
        xs: list[int] = [5, 2]
        xs[0] = True
        xs.append(False)
        self.assertEqual(kinds(xs), ["bool", "int", "bool"])
        self.assertEqual([repr(int(v)) for v in xs], ["1", "2", "0"])

    def test_a_dict_value(self) -> None:
        d: dict[str, int] = {}
        d["t"] = True
        d["n"] = 4
        self.assertEqual(facts(d["t"]), ("bool", True, True, False, "1", 2))
        self.assertEqual(described(d["n"]), "4:int:False:False")

    def test_literals_given_to_a_function(self) -> None:
        d: dict[str, int] = {"a": True, "b": 3}
        self.assertEqual(facts(d["a"]), ("bool", True, True, False, "1", 2))
        self.assertEqual(facts(d["b"]), ("int", False, False, False, "3", 4))
        self.assertEqual(repr(max(d["a"], 0)), "True")
        t: tuple[int, int] = (True, 5)
        self.assertEqual(facts(t[0]), ("bool", True, True, False, "1", 2))

    def test_returned_by_a_function(self) -> None:
        xs: list[int] = [5]
        xs.append(True)
        c = Counter(3)
        c.n = False
        self.assertEqual(facts(ident(xs[1])), ("bool", True, True, False, "1", 2))
        self.assertEqual(facts(ident(c.n)), ("bool", True, False, True, "0", 1))
        self.assertEqual(facts(ident(xs[0])), ("int", False, False, False, "5", 6))

    def test_type_objects(self) -> None:
        xs: list[int] = [5]
        xs.append(True)
        self.assertTrue(type(xs[1]) is bool)
        self.assertFalse(type(xs[1]) is int)
        self.assertEqual((type(5).__name__, int.__name__), ("int", "int"))


if __name__ == "__main__":
    unittest.main()
