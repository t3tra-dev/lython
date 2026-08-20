# Five bare spellings were claimed by manifest contracts whether or not
# anything imported them -- Task, Future, AbstractEventLoop, CancelledError,
# Context -- and fourteen more by the protocol table, so `class Task` followed
# by `def top(ts: list[Task])` typed the parameter as asyncio's Task and the
# call was refused with "arguments do not match Callable contract for function
# target top", naming neither the class nor the collision. With the annotation fixed, `Task` then took asyncio's
# class ID 15 from a namespace guess in the lowering and the program
# SEGFAULTED. Must run: a refusal is what the annotation half regresses to, but
# the class-id half is a crash, and only running it says the instances are
# tagged as themselves.


class Task:
    def __init__(self, name: str, pri: int) -> None:
        self.name = name
        self.pri = pri

    def label(self) -> str:
        return self.name + "/" + str(self.pri)


class AbstractEventLoop:
    def __init__(self, items: list[int]) -> None:
        self.items = items

    def total(self) -> int:
        t = 0
        for v in self.items:
            t = t + v
        return t


class Future:
    def __init__(self, v: int) -> None:
        self.v = v


class Context:
    def __init__(self, name: str) -> None:
        self.name = name


class CancelledError:
    def __init__(self, n: int) -> None:
        self.n = n


def top(tasks: list[Task], n: int) -> list[str]:
    ordered = sorted(tasks, key=lambda t: (-t.pri, t.name))
    out: list[str] = []
    for t in ordered[:n]:
        out.append(t.label())
    return out


ts = [Task("a", 1), Task("b", 5), Task("c", 5), Task("d", 3)]
print(top(ts, 3))

print(AbstractEventLoop([1, 2, 3]).total())
print(Future(7).v, Context("c").name, CancelledError(9).n)

# The class id is what a dict key and an isinstance test read, so both are
# asked here: a class tagged with asyncio's id answers them against the wrong
# schema.
print(isinstance(ts[0], Task), isinstance(Future(1), Task))
seen: dict[str, Task] = {}
for t in ts:
    seen[t.name] = t
print(sorted(seen), seen["b"].label())
