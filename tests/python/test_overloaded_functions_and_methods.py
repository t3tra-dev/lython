# A function or a method declared with `@overload`s runs its implementation,
# and each call has the type of the first overload that accepts it -- in
# this module and through an import.
import unittest
from typing import overload

import overload_shapes
from overload_shapes import parse


@overload
def describe(x: int) -> str: ...
@overload
def describe(x: str) -> int: ...
def describe(x: int | str) -> int | str:
    if isinstance(x, int):
        return "int " + str(x)
    return len(x)


@overload
def first[T](xs: list[T]) -> T: ...
@overload
def first[T](xs: list[T], default: T) -> T: ...
def first[T](xs: list[T], default: T | None = None) -> T | None:
    if xs:
        return xs[0]
    return default


def same(x: int | str) -> int | str:
    return x


class Shelf:
    @overload
    def take(self, n: int) -> list[str]: ...
    @overload
    def take(self) -> str: ...
    def take(self, n: int | None = None) -> list[str] | str:
        if n is None:
            return "one"
        return ["item"] * n


class OverloadTest(unittest.TestCase):

    def test_a_module_function(self) -> None:
        self.assertEqual(describe(3).upper(), "INT 3")
        self.assertEqual(describe("abc") + 1, 4)

    def test_a_generic_function(self) -> None:
        self.assertEqual(first([1, 2]) + 1, 2)
        none: list[int] = []
        self.assertEqual(first(none, 7) * 2, 14)
        self.assertEqual(first(["a"]).upper(), "A")

    def test_a_function_with_the_implementation_signature_is_its_own(self) -> None:
        self.assertEqual(same(1), 1)
        self.assertEqual(same("s"), "s")

    def test_a_method(self) -> None:
        shelf = Shelf()
        self.assertEqual(shelf.take().upper(), "ONE")
        self.assertEqual(len(shelf.take(2)), 2)

    def test_an_imported_function_and_method(self) -> None:
        self.assertEqual(overload_shapes.parse("4").bit_length(), 3)
        self.assertEqual(parse("8").bit_length(), 4)
        self.assertFalse(overload_shapes.parse("1.5", True).is_integer())
        store = overload_shapes.Store()
        self.assertEqual(store.get("b", 9).bit_length(), 4)
        self.assertIsNone(store.get("b"))


if __name__ == "__main__":
    unittest.main()
