# An instance of a SOURCE class crosses the generator suspension ABI on the
# object-family value lane, in all three positions the lane machinery has:
#
#   - yielded (the value lane carries the instance's whole physical span: the
#     object header plus one box per boxed field, not just a handle),
#   - held across a yield (a frame lane in the generator's storage),
#   - passed in (an argument lane the generator retains at creation).
#
# Read back in the loop body, a field borrows out of that span, so the
# resumer's release of the yielded instance has to wait for the borrow's
# retain: `label.text` is the value under test, not just the fact that the
# program compiles.
from typing import Iterator


class Label:
    def __init__(self, text: str, weight: int) -> None:
        self.text: str = text
        self.weight: int = weight


class Counter:
    def __init__(self, start: int) -> None:
        self.n: int = start


# One boxed field and nothing else: the box word loads ARE the instance's last
# physical use, so the borrow they assemble is retained after them.
class Word:
    def __init__(self, text: str) -> None:
        self.text: str = text


def one() -> Iterator[Label]:
    yield Label("alpha", 1)


def pair(first: Label, second: Label) -> Iterator[Label]:
    yield first
    yield second


def held(first: Label) -> Iterator[Label]:
    kept = Label("kept", 99)
    yield first
    yield kept


def counted(seed: Counter) -> Iterator[Counter]:
    yield Counter(seed.n * 2)


def words() -> Iterator[Word]:
    yield Word("solo")


for label in one():
    print(label.text, len(label.text), label.weight)

for label in pair(Label("beta", 22), Label("gamma", 333)):
    print(label.text, len(label.text), label.weight)

for label in held(Label("delta", 4)):
    print(label.text, label.weight)

for counter in counted(Counter(5)):
    print(counter.n)

for word in words():
    print(word.text)

g = one()
print(next(g).text)

kept = []
for label in pair(Label("epsilon", 6), Label("zeta", 7)):
    kept.append(label.text)
print(kept)
