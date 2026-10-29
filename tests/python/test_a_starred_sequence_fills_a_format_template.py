# A starred sequence into a literal `str.format` template fills the fields it
# reaches, in order or by number; members past the last field are ignored,
# and a sequence too short for a field raises the IndexError CPython raises
# at that field.
import unittest


def row(cells: list[str]) -> str:
    return "{:>4}|{:>4}".format(*cells)


class StarredFormatTest(unittest.TestCase):
    def test_fields_in_order(self):
        parts = ["a", "b"]
        self.assertEqual("{}-{}".format(*parts), "a-b")
        self.assertEqual(row(["x", "yz"]), "   x|  yz")
        self.assertEqual("{}+{}".format(*range(3)), "0+1")

    def test_numbered_fields_and_written_positionals(self):
        words = ("one", "two")
        self.assertEqual("{1} {0} {1}".format(*words), "two one two")
        self.assertEqual("{} {} {}".format("lead", *words), "lead one two")

    def test_a_short_sequence_raises_at_the_first_field_it_cannot_fill(self):
        with self.assertRaises(IndexError) as raised:
            "{}-{}".format(*["a"])
        self.assertEqual(str(raised.exception),
                         "Replacement index 1 out of range for positional "
                         "args tuple")
        with self.assertRaises(IndexError) as raised:
            "{2}{0}".format(*[1, 2])
        self.assertEqual(str(raised.exception),
                         "Replacement index 2 out of range for positional "
                         "args tuple")


if __name__ == "__main__":
    unittest.main()
