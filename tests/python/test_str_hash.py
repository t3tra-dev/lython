# A str's hash is kept once taken: it agrees with the hash of an equal str
# built another way, including after the str was appended to in place, and
# dict and set lookups find keys by it.
import unittest


def built(parts: list[str]) -> str:
    return "".join(parts)


class StrHashTest(unittest.TestCase):

    def test_equal_strs_hash_alike(self) -> None:
        a = built(["ab", "cd"])
        b = built(["a", "bcd"])
        self.assertEqual(hash(a), hash(b))
        self.assertEqual(hash(a), hash(a))
        self.assertEqual(hash(a), hash("abcd"))

    def test_appending_after_hashing_forgets_the_hash(self) -> None:
        # Hashed between appends, so the later appends can land in the room
        # the first one left rather than in a fresh str.
        s = built(["x", "y"])
        for i in range(20):
            before = hash(s)
            s += "z"
            self.assertNotEqual(hash(s), before)
        self.assertEqual(hash(s), hash(built(["xy", "z" * 20])))

    def test_dict_and_set_find_keys_by_cached_hash(self) -> None:
        keys = [built(["k", str(i)]) for i in range(200)]
        d = {k: i for i, k in enumerate(keys)}
        for i in range(200):
            self.assertEqual(d[built(["k", str(i)])], i)
        seen = set(keys)
        self.assertTrue(built(["k", "7"]) in seen)
        self.assertFalse(built(["k", "200"]) in seen)

    def test_wide_and_empty_strs(self) -> None:
        self.assertEqual(hash(built(["", ""])), 0)
        wide = built(["Ā", "\U0001f600"])
        self.assertEqual(hash(wide), hash("Ā\U0001f600"))
        self.assertEqual({wide: 1}[built(["Ā\U0001f600"])], 1)


if __name__ == "__main__":
    unittest.main()
