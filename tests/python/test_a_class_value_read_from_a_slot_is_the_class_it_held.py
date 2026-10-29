# A class value read back out of a container is the class that was stored,
# whichever member of a union of class values it is: every class value's
# slot holds a type object, and the type object says which class it is --
# a subclass of a member included.
import unittest


class Base:
    pass


class Derived(Base):
    pass


def name_at(classes: tuple[type[KeyError], type[IndexError]], i: int) -> str:
    return classes[i].__name__


class ClassValueSlotTest(unittest.TestCase):
    def test_a_list_of_two_builtin_classes(self):
        xs: list[type[int] | type[str]] = [int, str]
        self.assertEqual([c.__name__ for c in xs], ["int", "str"])
        self.assertEqual([c("5") for c in xs], [5, "5"])

    def test_a_tuple_read_at_a_run_time_index(self):
        pair = (KeyError, IndexError)
        self.assertEqual([name_at(pair, i) for i in range(2)],
                         ["KeyError", "IndexError"])

    def test_a_subclass_of_a_member(self):
        xs: list[type[Base] | type[int]] = [Derived, int, Base]
        self.assertEqual([c.__name__ for c in xs], ["Derived", "int", "Base"])


if __name__ == "__main__":
    unittest.main()
