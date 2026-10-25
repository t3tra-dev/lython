# str.count, str.replace and str.find answer what CPython's do over latin-1
# and wider strings: single-character and longer needles, non-overlapping
# counting, ranges, empty needles, a count limit on replace, and needles that
# never occur or cannot occur in a narrower haystack.
import unittest


BASE = "the quick brown fox jumps over the lazy dog " * 20
WIDE = "héllo wörld héllo " * 3


def naive_count(s: str, sub: str, start: int, end: int) -> int:
    # Non-overlapping occurrences of a non-empty `sub` in s[start:end].
    count = 0
    i = start
    while i + len(sub) <= end:
        if s[i:i + len(sub)] == sub:
            count += 1
            i += len(sub)
        else:
            i += 1
    return count


class CountTest(unittest.TestCase):

    def test_against_a_scan_of_every_position(self) -> None:
        for sub in ["the", "o", "dog ", "zz", "e ", " t", "lazy dog the"]:
            self.assertEqual(BASE.count(sub),
                             naive_count(BASE, sub, 0, len(BASE)))
            self.assertEqual(BASE.count(sub, 5, 400),
                             naive_count(BASE, sub, 5, 400))

    def test_one_character_at_the_start_of_the_range(self) -> None:
        self.assertEqual("aaa".count("a"), 3)
        self.assertEqual("abcabc".count("a", 3), 1)
        self.assertEqual(BASE.count("t"), naive_count(BASE, "t", 0, len(BASE)))
        self.assertEqual(BASE.count("t", 31),
                         naive_count(BASE, "t", 31, len(BASE)))

    def test_occurrences_do_not_overlap(self) -> None:
        self.assertEqual("aaaa".count("aa"), 2)
        self.assertEqual("aaaaa".count("aa"), 2)
        self.assertEqual("abababab".count("aba"), 2)

    def test_edges(self) -> None:
        self.assertEqual("".count("a"), 0)
        self.assertEqual("a".count("a", 1), 0)
        self.assertEqual("abc".count(""), 4)
        self.assertEqual("abc".count("", 1, 2), 2)
        self.assertEqual("abc".count("abcd"), 0)
        self.assertEqual("abcabc".count("abc", -3), 1)
        self.assertEqual("xa".count("a", 0, 1), 0)

    def test_wide_strings(self) -> None:
        self.assertEqual(WIDE.count("héllo"), 6)
        self.assertEqual(WIDE.count("ö"), 3)
        self.assertEqual(WIDE.count("llo"), 6)
        self.assertEqual("xĀyĀ".count("Ā"), 2)
        self.assertEqual("abc".count("Ā"), 0)


class ReplaceTest(unittest.TestCase):

    def test_one_character_for_another(self) -> None:
        out = BASE.replace("o", "0")
        self.assertEqual(len(out), len(BASE))
        self.assertEqual(out.count("0"), BASE.count("o"))
        self.assertEqual(out.count("o"), 0)
        self.assertEqual(out[:30], "the quick br0wn f0x jumps 0ver")

    def test_a_count_limit(self) -> None:
        self.assertEqual("aaa".replace("a", "b", 2), "bba")
        self.assertEqual(BASE.replace("o", "0", 3)[:46],
                         "the quick br0wn f0x jumps 0ver the lazy dog th")
        self.assertEqual("abc".replace("a", "b", 0), "abc")
        self.assertEqual("xo".replace("o", "0", -1), "x0")

    def test_no_match_and_identity(self) -> None:
        self.assertEqual("abc".replace("z", "y"), "abc")
        self.assertEqual("oooo".replace("o", "o"), "oooo")
        self.assertEqual("".replace("a", "b"), "")

    def test_a_replacement_that_spells_the_needle(self) -> None:
        self.assertEqual("ab".replace("ab", "ba"), "ba")
        self.assertEqual("aaa".replace("a", "aa"), "aaaaaa")

    def test_wide_strings(self) -> None:
        self.assertEqual("héllo".replace("l", "L"), "héLLo")
        self.assertEqual("hello".replace("l", "Ā"), "heĀĀo")
        self.assertEqual(WIDE.replace("ö", "o")[:11], "héllo world")


class FindTest(unittest.TestCase):

    def test_find_from_a_position(self) -> None:
        for start in range(0, 50, 7):
            self.assertEqual(BASE.find("lazy", start), 35 if start <= 35 else 79)
        self.assertEqual(BASE.find("cat"), -1)
        self.assertEqual(WIDE.find("wörld"), 6)


if __name__ == "__main__":
    unittest.main()
