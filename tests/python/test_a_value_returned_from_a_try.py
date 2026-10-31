# A union returned from inside a try, or from its handler, is the member it
# was given, annotated or not, in a function, a method and a loop; a handler
# that falls through leaves the return after the try to answer. A read the
# compiler decides will raise -- a missing key of a dict display, a position
# past a tuple or list display -- raises as CPython does, with the rest of
# its statement building an object that is never built. An object that
# dies on both arms of a branch inside a try is released on each.
import unittest
from typing import Optional


class Node:
    def __init__(self, v: int) -> None:
        self.v = v


class Failure:
    def __init__(self, message: str) -> None:
        self.message = message + "!"


def parse(s: str) -> int | str:
    try:
        return int(s)
    except ValueError:
        return s


def parse_unannotated(s: str):
    try:
        return int(s)
    except ValueError:
        return s


def parse_falling_through(s: str) -> int | str:
    try:
        return int(s)
    except ValueError:
        pass
    return s.upper()


def convert(s: str):
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def nested(s: str) -> int | str | None:
    try:
        try:
            return int(s)
        except ValueError:
            if s == "":
                return None
            raise
    except ValueError:
        return s.upper()


def make(i: int) -> Node | Failure:
    try:
        if i % 3 == 0:
            raise KeyError(i)
        return Node(i)
    except KeyError as e:
        return Failure(str(e))


def first_number(xs: list[str]) -> int | str:
    for x in xs:
        try:
            return int(x)
        except ValueError:
            continue
    return "none"


def numbers(xs: list[str]) -> list[int] | dict[str, int]:
    out: list[int] = []
    try:
        for x in xs:
            out.append(int(x))
        return out
    except ValueError:
        pass
    return {x: len(x) for x in xs}


class Table:
    def get(self, key: str, d: dict[str, int]) -> int | str:
        try:
            return d[key]
        except KeyError:
            return key


def missing_key() -> Failure:
    d = {"a": 1}
    return Failure(str(d["b"]))


def past_the_list() -> Failure:
    xs = [1, 2]
    return Failure(str(xs[5]))


def past_the_tuple() -> Failure:
    xs = (1, 2)
    return Failure(str(xs[5]))


def caught_missing_key() -> Optional[Failure]:
    try:
        d = {"a": 1}
        return Failure(str(d["b"]))
    except KeyError:
        return None


class Lookup:
    def get(self, key: str, d: dict[str, int]) -> str | None:
        try:
            if d[key] > 0:
                return None
            return None
        except KeyError:
            return key


def described(v: object) -> str:
    return type(v).__name__ + ":" + str(v)


class AUnionReturnedFromATry(unittest.TestCase):
    def test_the_member_returned_from_the_body_or_the_handler(self) -> None:
        self.assertEqual(described(parse("12")), "int:12")
        self.assertEqual(described(parse("ab")), "str:ab")
        self.assertEqual(described(parse_unannotated("7")), "int:7")
        self.assertEqual(described(parse_unannotated("x")), "str:x")

    def test_a_handler_that_falls_through(self) -> None:
        self.assertEqual(described(parse_falling_through("3")), "int:3")
        self.assertEqual(described(parse_falling_through("q")), "str:Q")
        self.assertEqual(
            [described(convert(s)) for s in ["1", "2.5", "zz"]],
            ["int:1", "float:2.5", "str:zz"])

    def test_nested_tries(self) -> None:
        self.assertEqual(
            [described(nested(s)) for s in ["4", "", "k"]],
            ["int:4", "NoneType:None", "str:K"])

    def test_objects_of_two_classes(self) -> None:
        kinds = [type(make(i)).__name__ for i in range(4)]
        self.assertEqual(kinds, ["Failure", "Node", "Node", "Failure"])
        failure = make(3)
        self.assertIsInstance(failure, Failure)
        if isinstance(failure, Failure):
            self.assertEqual(failure.message, "3!")

    def test_in_a_loop_and_a_method(self) -> None:
        self.assertEqual(described(first_number(["a", "5"])), "int:5")
        self.assertEqual(described(first_number(["a"])), "str:none")
        self.assertEqual(numbers(["1", "2"]), [1, 2])
        self.assertEqual(numbers(["x", "yy"]), {"x": 1, "yy": 2})
        table = {k: 9 for k in ["a"]}
        self.assertEqual(described(Table().get("a", table)), "int:9")
        self.assertEqual(described(Table().get("b", table)), "str:b")
        self.assertIsNone(Lookup().get("a", {"a": 9}))
        self.assertEqual(Lookup().get("b", {"a": 9}), "b")


class AReadDecidedToRaise(unittest.TestCase):
    def test_a_missing_key_of_a_display(self) -> None:
        with self.assertRaises(KeyError):
            missing_key()
        self.assertIsNone(caught_missing_key())

    def test_a_position_past_a_display(self) -> None:
        with self.assertRaises(IndexError) as caught:
            past_the_list()
        self.assertEqual(str(caught.exception), "list index out of range")
        with self.assertRaises(IndexError) as caught:
            past_the_tuple()
        self.assertEqual(str(caught.exception), "tuple index out of range")


if __name__ == "__main__":
    unittest.main()
