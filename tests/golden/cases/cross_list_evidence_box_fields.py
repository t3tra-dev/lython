# Cross-track: wave25/defects settles a list's evidence across branch and loop
# merges (the cross-block demote list needed alongside dict), and wave25/abi
# moved reference-typed fields behind a stable box so a field read now goes
# through the box rather than reading inline words. A field read INSIDE a merged
# block feeds the demoted list, so the two meet on every append of a field.
#
# The rows travel as parameters: a module-level list/instance global is not
# visible inside a function body (loudly diagnosed, unrelated to this cross).


# Two fields only: a three-field Row expands past the payload box's handle
# budget and is loudly rejected as a container element (wave25/abi), so the
# derived text is a method rather than a third field.
class Row:
    def __init__(self, name: str, score: int) -> None:
        self.name = name
        self.score = score

    def tag(self) -> str:
        return self.name + "#" + str(self.score)


def make() -> list[Row]:
    return [Row("a", 3), Row("b", 1), Row("c", 2)]


# Field reads inside an if/else merge feeding one list built before the branch.
def split(rows: list[Row], limit: int) -> list[str]:
    out: list[str] = []
    for row in rows:
        if row.score >= limit:
            out.append(row.name)
        else:
            out.append(row.tag())
    return out


print(split(make(), 2))
print(split(make(), 9))
print(split(make(), 0))


# A list mutated in both arms of a branch and again after the merge: the
# merged evidence must survive, and every element is a field read. (Rebinding
# the local to a fresh list in one arm instead is still rejected -- the
# pre-branch list leaks -- so this shape stays on the mutation path.)
def merged(rows: list[Row], pick: int) -> list[str]:
    acc: list[str] = ["seed"]
    if pick == 1:
        acc.append(rows[0].name)
        acc.append(rows[1].name)
    else:
        acc.append(rows[2].tag())
    acc.append(rows[0].tag())
    return acc


print(merged(make(), 1))
print(merged(make(), 2))


# A loop-carried list of field reads, with a break that leaves the loop before
# the last append.
def until(rows: list[Row], stop: str) -> list[str]:
    found: list[str] = []
    for row in rows:
        if row.name == stop:
            break
        found.append(row.name)
        found.append(row.tag())
    return found


print(until(make(), "c"))
print(until(make(), "a"))
print(until(make(), "z"))


# A while loop whose body mutates a str field and re-reads it through the box.
def rename(times: int) -> list[str]:
    row = Row("x", 0)
    seen: list[str] = []
    i = 0
    while i < times:
        row.name = row.name + str(i)
        seen.append(row.name)
        i = i + 1
    seen.append(row.name)
    return seen


print(rename(3))
print(rename(0))


# The nested dict shape: field reads as dict keys and values, read back after
# the merge that built them.
def index(rows: list[Row], flip: int) -> dict[str, int]:
    table: dict[str, int] = {}
    for row in rows:
        if flip == 1:
            table[row.name] = row.score
        else:
            table[row.tag()] = row.score + 10
    return table


print(index(make(), 1))
print(index(make(), 0))
print(index(make(), 1)["b"], index(make(), 0)["c#2"])
