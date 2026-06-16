def describe(name: str | None) -> str:
    if name is None:
        return "anonymous"
    return "name=" + name


print(describe("Alice"))
print(describe(None))


def pick(flag: int) -> int | None:
    if flag <= 0:
        return None
    return flag * 10


x: int | None = pick(3)
if x is not None:
    print(x)

y: int | None = pick(0)
if y is None:
    print("picked none")


class User:
    def __init__(self, name: str) -> None:
        self.name = name


def find_user(flag: int) -> User | None:
    if flag <= 0:
        return None
    return User("Bob")


u: User | None = find_user(1)
if u is not None:
    print(u.name)

v: User | None = find_user(-1)
if v is None:
    print("no user")
