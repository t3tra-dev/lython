# A name a loop reads at the top of its body and rebinds further in -- inside
# an inner loop, a `match` case, or loops nested deeper -- holds the new value
# on the loop's next trip.
import unittest


def by_inner_for() -> int:
    k = 1
    total = 0
    for i in range(2):
        total += k
        for g in [100]:
            k = g
    return total


def by_inner_while() -> int:
    k = 1
    total = 0
    for i in range(2):
        total += k
        j = 0
        while j < 1:
            k = 100
            j += 1
    return total


def by_inner_for_in_while() -> int:
    k = 1
    total = 0
    i = 0
    while i < 2:
        total += k
        for g in [100]:
            k = g
        i += 1
    return total


def by_match() -> int:
    k = 1
    total = 0
    for i in range(3):
        total += k
        match i:
            case 0:
                k = 100
            case _:
                pass
    return total


def by_nested_loops() -> int:
    k = 1
    total = 0
    for i in range(2):
        total += k
        for j in range(1):
            for m in range(1):
                k = 100
    return total


class InnerRebindingTest(unittest.TestCase):

    def test_an_inner_loop(self) -> None:
        self.assertEqual(by_inner_for(), 101)
        self.assertEqual(by_inner_while(), 101)
        self.assertEqual(by_inner_for_in_while(), 101)

    def test_a_match_case(self) -> None:
        self.assertEqual(by_match(), 201)

    def test_loops_nested_deeper(self) -> None:
        self.assertEqual(by_nested_loops(), 101)


if __name__ == "__main__":
    unittest.main()
