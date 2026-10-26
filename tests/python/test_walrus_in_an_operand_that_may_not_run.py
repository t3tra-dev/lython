# A name bound by `:=` inside an operand that may not run -- an operand of
# `and` / `or` after the first, an arm of a conditional expression -- is the
# function's local afterwards: read after an `if`, in its body, in a `while`
# body, and raising UnboundLocalError when the operand did not run.
import unittest


def after_if(xs: list[str]) -> str:
    if len(xs) > 5 and (tag := xs[0]):
        pass
    else:
        tag = "w"
    return tag


def in_the_body(n: int) -> int:
    if n > 0 and (m := n * 2) > 5:
        return m
    return -1


def after_or(n: int) -> int:
    ok = n < 0 or (k := n + 1) > 3
    return k if not ok or n >= 3 else -1


def in_both_arms(n: int) -> int:
    v = (a := n * 3) if n > 2 else (a := n - 1)
    return v + a


def in_a_while(xs: list[int]) -> list[int]:
    out: list[int] = []
    pos = 0
    while pos < len(xs) and (cur := xs[pos]) != 0:
        out.append(cur * 2)
        pos += 1
    return out


def unbound(s: str) -> int:
    if len(s) > 0 and (z := len(s)) > 1:
        pass
    return z


class WalrusTest(unittest.TestCase):

    def test_read_after_the_expression(self) -> None:
        self.assertEqual((after_if(["a"]), after_if(["b"] * 6)), ("w", "b"))
        self.assertEqual((in_the_body(1), in_the_body(4)), (-1, 8))
        self.assertEqual(after_or(5), 6)
        self.assertEqual((in_both_arms(4), in_both_arms(1)), (24, 0))
        self.assertEqual(in_a_while([3, 0, 7]), [6])

    def test_an_operand_that_did_not_run(self) -> None:
        self.assertEqual(unbound("ab"), 2)
        with self.assertRaises(UnboundLocalError):
            unbound("")


if __name__ == "__main__":
    unittest.main()
