# WHAT: a field read before any store raises CPython's AttributeError, named
# for the instance's own class: a field only declared, one a method fills
# later, one a subclass's __init__ leaves to a super().__init__ it never calls,
# one a constructor reads (through a method) before storing it, and a bool
# field assigned on one branch. Reads that follow a store -- after the method
# ran, through super().__init__, after __init__ finished -- answer as usual,
# and every one of them is caught by an `except AttributeError`.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: whether a store came first
# is a run time fact for these shapes; the empty slot was a segfault for an
# object field and silently False for a bool one.

class A:
    x: str

class Lazy:
    label: str
    def __init__(self) -> None:
        self.n = 1
    def start(self) -> None:
        self.label = "started" * 2

class Base:
    def __init__(self) -> None:
        self.tag = "base"

class NoSuper(Base):
    def __init__(self) -> None:
        self.other = "o"

class WithSuper(Base):
    def __init__(self) -> None:
        super().__init__()
        self.other = "w"

class Early:
    def __init__(self) -> None:
        self.peek()
        self.v = "late"
    def peek(self) -> None:
        try:
            print(self.v)
        except AttributeError as e:
            print("early:", e)

def tag_of(b: Base) -> str:
    return b.tag

def attempt(k: int) -> None:
    try:
        if k == 0:
            print(A().x)
        elif k == 1:
            print(Lazy().label)
        elif k == 2:
            print(tag_of(NoSuper()))
    except AttributeError as e:
        print(type(e).__name__, e)

for k in range(3):
    attempt(k)
z = Lazy()
z.start()
print(z.label, tag_of(WithSuper()), tag_of(Base()))
e = Early()
print(e.v)


class Flagged:
    flag: bool
    def __init__(self, set_it: bool) -> None:
        if set_it:
            self.flag = False

def read_flag(f: Flagged) -> str:
    try:
        return str(f.flag)
    except AttributeError as e:
        return "AE " + str(e)

print(read_flag(Flagged(True)), read_flag(Flagged(False)))
class Both:
    def __init__(self) -> None:
        self.on = True
        self.off = False
both = Both()
print(both.on, both.off, not both.off)
