# A module-level name bound more than once -- rebound under an `if`, in a
# loop, by an augmented assignment, as a `for` target, to values of two types
# -- is read by a function as the value it has when the function runs, and a
# function declaring it `global` writes it.
import sys
import unittest

tag = "global"
if len(sys.argv) > 0:
    tag = "rebound"
mixed = 1
if len(sys.argv) > 0:
    mixed = "a"
data = [1, 2]
data = data + [3]
total = 0
last = -1
for last in range(4):
    total += last
mode = "slow"


def read_all() -> str:
    return " ".join([tag, repr(mixed), str(len(data)), str(total), str(last)])


def bump() -> None:
    global total
    total = total + 100


def current_mode() -> str:
    return mode


class GlobalTest(unittest.TestCase):

    def test_read_from_a_function(self) -> None:
        self.assertEqual(read_all(), "rebound 'a' 3 6 3")

    def test_written_by_a_function_and_rebound_later(self) -> None:
        global mode
        bump()
        mode = "fast"
        self.assertEqual((total, current_mode()), (106, "fast"))


if __name__ == "__main__":
    unittest.main()
