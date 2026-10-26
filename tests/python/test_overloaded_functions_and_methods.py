# A function or a method declared with `@overload`s runs its implementation,
# and each call has the type of the first overload that accepts it -- in
# this module and through an import.
import unittest
from typing import overload

import overload_shapes
from overload_shapes import parse
from overload_shapes import parse as read


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


spell = describe

# A module name bound to an overloaded function on one path and to another
# function on the next is typed by what it holds at each read.
module_loop_results: list[str] = []
for use_same in (False, True):
    chosen = describe
    if use_same:
        chosen = same
    module_loop_results.append(repr(chosen(5)))
for _ in range(2):
    typed = describe
    module_loop_results.append(typed(6).upper())
looped = describe
for _ in range(2):
    module_loop_results.append(repr(looped(7)))
    looped = same
matched = describe
match len(module_loop_results):
    case 6:
        matched = same
module_loop_results.append(repr(matched(8)))


def call_rebound[T](x: T) -> str:
    return repr(rebound(3))


rebound = describe
rebound_results = [call_rebound(1)]
rebound = same
rebound_results.append(call_rebound(1))


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

    def test_a_name_bound_to_an_overloaded_function(self) -> None:
        self.assertEqual(spell(3).upper(), "INT 3")
        local = describe
        self.assertEqual(local("ab") + 1, 3)
        self.assertEqual(read("2").bit_length(), 2)
        through_module = overload_shapes.parse
        self.assertEqual(through_module("2").bit_length(), 2)

    def test_a_module_name_rebound_in_a_loop_or_a_branch(self) -> None:
        self.assertEqual(module_loop_results,
                         ["'int 5'", "5", "INT 6", "INT 6", "'int 7'", "7", "8"])

    def test_a_body_emitted_once_reads_what_the_module_name_holds(self) -> None:
        self.assertEqual(rebound_results, ["'int 3'", "3"])

    def test_a_local_of_the_same_name_is_not_the_overloaded_function(self) -> None:
        describe = same
        self.assertEqual(describe(1), 1)

    def test_a_name_rebound_on_one_path_calls_what_it_holds(self) -> None:
        results: list[str] = []
        for use_same in (False, True):
            chosen = describe
            if use_same:
                chosen = same
            results.append(repr(chosen(5)))
        self.assertEqual(results, ["'int 5'", "5"])


if __name__ == "__main__":
    unittest.main()
