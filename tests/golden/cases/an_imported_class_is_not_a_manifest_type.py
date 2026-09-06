# What: six questions this compiler answers by asking "is this contract one of
# MY classes or a manifest one", asked about a class that came from another
# file. The test was the NAME's shape -- a dot means manifest -- and an
# imported class's contract name is `mod.Bag`, so every one of them answered
# "manifest" for a program's own class. Five were refusals; the last two were
# silent, folding a live branch away.
#
# Each line needs running: the four operator forms print the value the class's
# own body computes, and the two hierarchy forms print which branch survived.
# Written in ONE file every one of them was already right, which is what says
# the boundary is the cause.
from a_module_of_operator_classes import Bag, Big, MyErr, Num, Slot


def classify(e: Exception) -> str:
    if isinstance(e, MyErr):
        return "mine"
    return "other"


print("in", 2 in Bag(), 9 in Bag())
print("reflected <", Big(1) < Big(2), Big(2) < Big(1))
print("eq both ways", Num(1) == 1, 1 == Num(1))
print("radd", 5 + Num(1))
print("index", [10, 20, 30][Slot(1)])
print("isinstance", classify(MyErr("x")), classify(ValueError("y")))
print("issubclass", issubclass(MyErr, Exception), issubclass(MyErr, ValueError))
