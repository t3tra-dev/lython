# What this pins: an annotated module global of a container type is storage-
# backed, so a function reads the module's current value and a mutation
# anywhere is visible everywhere. Every read below used to be
# "unresolved name 'T'".
#
# Why this needs to run rather than assert on a diagnostic: the property is
# that the cell tracks the object across a REALLOCATION. A list grows past its
# initial capacity here (50 appends from empty), and the read that follows goes
# through a fresh load of the cell -- if the cell held a stale description that
# read would fetch freed storage, and if the mutation were dropped it would
# fetch the pre-mutation contents. Both compile; only the values separate them.
# The counts are also why the case is in the leak gate: the global holds one
# retained reference and a rebinding must release exactly the one it replaces.
#
# Every expected line is python3.14's.


# --- read from a function, every container shape ---------------------------
TABLE: dict[str, int] = {"a": 1, "b": 2}
ITEMS: list[int] = [10, 20, 30]
PAIR: tuple[int, str] = (7, "seven")
MARKS: set[int] = {1, 2, 3}
FROZEN: frozenset[int] = frozenset([4, 5])


def look(k: str) -> int:
    return TABLE[k]


def item(i: int) -> int:
    return ITEMS[i]


def sizes() -> str:
    return str(len(TABLE)) + "/" + str(len(ITEMS)) + "/" + str(len(MARKS)) + \
        "/" + str(len(FROZEN))


print(look("a"), look("b"))
print(item(0), item(2), len(ITEMS))
print(PAIR[0], PAIR[1])
print(3 in MARKS, 9 in MARKS, 4 in FROZEN)
print(sizes())


# --- a growth from inside a function, read back from another ---------------
GROWN: list[int] = []


def fill(n: int) -> None:
    i = 0
    while i < n:
        GROWN.append(i * i)
        i += 1


def at(i: int) -> int:
    return GROWN[i]


fill(50)
print(len(GROWN), at(0), at(7), at(49), GROWN[25])
GROWN.append(-1)
print(len(GROWN), at(50))


# --- dict insert from a function, past its initial capacity ---------------
COUNTS: dict[int, str] = {}


def put(k: int) -> None:
    COUNTS[k] = str(k * 3)


def get(k: int) -> str:
    return COUNTS[k]


j = 0
while j < 60:
    put(j)
    j += 1
print(len(COUNTS), get(0), get(59), COUNTS[30])


# --- the mutators that carry a rebind result ------------------------------
ORDER: list[int] = [3, 1, 2]


def sort_it() -> None:
    ORDER.sort()


def insert_front(v: int) -> None:
    ORDER.insert(0, v)


def splice() -> None:
    ORDER[1:3] = [99]


sort_it()
print(ORDER)
insert_front(0)
print(ORDER)
splice()
print(ORDER)

SEEN: set[int] = {0}


def collect(n: int) -> None:
    i = 0
    while i < n:
        SEEN.add(i % 7)
        i += 1


collect(40)
print(len(SEEN), 6 in SEEN, 7 in SEEN)


# --- extend / pop, and a rebinding assignment through `global` -------------
WORDS: list[str] = ["a"]


def extend_it() -> None:
    WORDS.extend(["b", "c"])


def pop_it() -> str:
    return WORDS.pop()


extend_it()
extend_it()
print(WORDS, len(WORDS))
print(pop_it(), WORDS)


def replace() -> None:
    global WORDS
    WORDS = ["z"]


replace()
print(WORDS)


# --- a container of containers --------------------------------------------
NESTED: list[list[int]] = [[1], [2]]


def push() -> None:
    NESTED.append([3])


def inner(i: int, j: int) -> int:
    return NESTED[i][j]


push()
print(NESTED, inner(0, 0), inner(2, 0))
