# WHAT: `return NotImplemented` -- the documented way to write a comparison
# dunder that does not handle an operand -- and what each of the six spellings
# falls back to when it is returned.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the fallback is an ANSWER.
# `==` falls back to identity and `!=` to its negation, so a class that
# declines an int must still say False and True for them; the four orderings
# fall back to a TypeError whose message a reader compares against CPython's
# word for word.
#
# ⛔ THE MESSAGE NAMES THE DUNDER'S OWN OPERAND ORDER. CPython builds it from
# the operator as written, so a REFLECTED ordering whose both sides decline
# could print the pair the other way round. A class that defines `__lt__`
# alone -- the idiom -- is dispatched directly and cannot reach that.
class Money:
    amount: int

    def __init__(self, amount: int) -> None:
        self.amount = amount

    def __repr__(self) -> str:
        return "Money(" + str(self.amount) + ")"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Money):
            return NotImplemented
        return self.amount == other.amount

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Money):
            return NotImplemented
        return self.amount < other.amount

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, Money):
            return NotImplemented
        return self.amount >= other.amount


m = Money(5)
print(m == Money(5), m == Money(6), m == m)
print(m == 5, m != 5, m == "five", m != "five")
print(Money(1) < Money(2), Money(2) < Money(1))
print(Money(2) >= Money(2), Money(1) >= Money(2))
print(sorted([Money(3), Money(1), Money(2)]))
try:
    print(m < 5)
except TypeError as e:
    print("TypeError:", e)
try:
    print(m >= "x")
except TypeError as e:
    print("TypeError:", e)


# --- identity between disjoint types is False, not a refusal ---------------
class Tag:
    pass


t = Tag()
n = 5
word = "s"
print(t is t, t is Tag(), t is n, t is word, n is word)
