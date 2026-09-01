# What: a `def` inside an if, a loop or a try binds its name for the whole
# scope in CPython, and only calling the name after the region shows which
# definition ran -- each of these answers differently.
flag = True
if flag:
    def go() -> int:
        return 1
else:
    def go() -> int:
        return 2


print(go())


def pick(mode: str) -> int:
    if mode == "a":
        def step() -> int:
            return 10
    else:
        def step() -> int:
            return 20

    return step()


print(pick("a"), pick("b"))


def from_a_loop() -> int:
    for _ in range(1):
        def once() -> int:
            return 5

    return once()


print(from_a_loop())


def from_a_try() -> str:
    try:
        def name() -> str:
            return "tried"
    except ValueError:
        def name() -> str:
            return "caught"

    return name()


print(from_a_try())


def never_runs(flag: bool) -> int:
    if flag:
        def maybe() -> int:
            return 1

    if flag:
        return maybe()
    return -1


print(never_runs(True), never_runs(False))
